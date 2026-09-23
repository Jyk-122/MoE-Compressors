"""Packed NF4 backbone checkpoints; the source directory supplies code and processor."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from prefetch.backbone.quantization import NF4Experts
from prefetch.backbone.structure import find_moe_blocks


logger = logging.getLogger(__name__)


def cpu_tensors(values):
    # Clone also separates shared codebooks/tied weights for safetensors.
    return {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}


def save_nf4_checkpoint(model, directory, source_model, blocksize=64, shard_bytes=512 * 2**20):
    """Write each MoE's packed weights/state, then shard the remaining native tensors."""
    import bitsandbytes as bnb
    from safetensors.torch import save_file

    if any("lora_A" in name or "prerouter" in name for name, _ in model.named_parameters()):
        raise ValueError("Export the base model before installing LoRA or prerouters")
    blocks = find_moe_blocks(model)
    if any(not isinstance(block.experts, NF4Experts) for _, block in blocks):
        raise ValueError("Export requires an experts_nf4 base model")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(format="prefetch.experts_nf4.v1", source_model=str(source_model),
                    blocksize=blocksize, bitsandbytes_version=bnb.__version__,
                    torch_version=str(torch.__version__), experts=[], native_shards=[],
                    quantized_source_parameters=0)
    if hasattr(model, "generation_config"):
        metadata["generation_config"] = model.generation_config.to_dict()
    for index, (name, block) in enumerate(blocks):
        for linear in list(block.experts.gate_up) + list(block.experts.down):
            state = linear.weight.quant_state
            if state.blocksize != blocksize or state.quant_type != "nf4" or not state.nested:
                raise ValueError("Expert quantization state differs from the requested NF4 checkpoint")
            metadata["quantized_source_parameters"] += state.shape.numel()
        filename = f"experts-{index:03d}.safetensors"
        save_file(cpu_tensors(block.experts.state_dict()), str(directory / filename))
        metadata["experts"].append(dict(name=name, file=filename))
        logger.info("Saved NF4 %s", name)

    prefixes = tuple(f"{name}.experts." for name, _ in blocks)
    shard, size = {}, 0

    def flush():
        filename = f"native-{len(metadata['native_shards']):03d}.safetensors"
        save_file(shard, str(directory / filename))
        metadata["native_shards"].append(filename)
        shard.clear()

    for name, value in model.state_dict().items():
        if name.startswith(prefixes):
            continue
        nbytes = value.numel() * value.element_size()
        if shard and size + nbytes > shard_bytes:
            flush()
            size = 0
        shard.update(cpu_tensors({name: value}))
        size += nbytes
    if shard:
        flush()
    # Written last: its presence marks a completed export.
    (directory / "nf4_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def read_nf4_metadata(directory, source_model, blocksize):
    metadata = json.loads((Path(directory) / "nf4_config.json").read_text(encoding="utf-8"))
    if metadata["format"] != "prefetch.experts_nf4.v1":
        raise ValueError("Unsupported NF4 checkpoint format")
    if metadata["source_model"] != str(source_model):
        raise ValueError("NF4 checkpoint source_model differs from model.path")
    if metadata["blocksize"] != blocksize:
        raise ValueError("NF4 checkpoint blocksize differs from model.nf4_blocksize")
    return metadata


def restore_nf4_weights(model, directory, metadata, device):
    """Fill a meta-parameter skeleton without reading or requantizing BF16 experts."""
    import bitsandbytes as bnb
    from safetensors.torch import load_file

    directory = Path(directory)
    blocks = dict(find_moe_blocks(model))
    if set(blocks) != {entry["name"] for entry in metadata["experts"]}:
        raise ValueError("NF4 checkpoint MoE modules differ from the source model")
    prefixes = tuple(f"{name}.experts." for name in blocks)
    expected = {name for name in model.state_dict() if not name.startswith(prefixes)}
    for entry in metadata["experts"]:
        block = blocks[entry["name"]]
        experts = NF4Experts(block.experts, device, metadata["blocksize"], quantize=False)
        values = load_file(str(directory / entry["file"]))
        for name, linear in experts.named_modules():
            if not isinstance(linear, bnb.nn.Linear4bit):
                continue
            key = f"{name}.weight"
            stats = {k[len(key) + 1:]: values.pop(k) for k in list(values) if k.startswith(key + ".")}
            weight = bnb.nn.Params4bit.from_prequantized(values.pop(key), stats, device=device, module=linear)
            if (tuple(weight.quant_state.shape) != (linear.out_features, linear.in_features)
                    or weight.blocksize != metadata["blocksize"] or weight.quant_type != "nf4"):
                raise ValueError(f"NF4 weight shape or quantization mismatch: {entry['name']}.{name}")
            linear.weight = weight
        if values:
            raise ValueError(f"Unexpected NF4 tensors in {entry['file']}: {list(values)}")
        block.experts = experts

    loaded = set()
    for filename in metadata["native_shards"]:
        values = load_file(str(directory / filename))
        for name, value in values.items():
            if name not in expected or name in loaded:
                raise ValueError(f"Unexpected or duplicate native tensor: {name}")
            path, _, leaf = name.rpartition(".")
            module = model.get_submodule(path)
            original = getattr(module, leaf)
            if original.shape != value.shape:
                raise ValueError(f"Native tensor shape differs: {name}")
            value = value.to(device)
            if leaf in module._parameters:
                value = nn.Parameter(value, requires_grad=False)
            setattr(module, leaf, value)
            loaded.add(name)
        del values
    if loaded != expected:
        raise ValueError(f"Missing native tensors: {sorted(expected - loaded)}")
    model.requires_grad_(False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    # Non-persistent buffers (e.g. RoPE frequencies) were constructed on CPU.
    model.to(device)
    return model


def load_nf4_checkpoint(base, device):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    metadata = read_nf4_metadata(base["nf4_checkpoint"], base["path"], base.get("nf4_blocksize", 64))
    model_config = AutoConfig.from_pretrained(base["path"], trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if base.get("attn_implementation"):
        kwargs["attn_implementation"] = base["attn_implementation"]
    # Keep buffers real: non-persistent RoPE buffers are absent from state_dict.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(model_config, **kwargs)
    restore_nf4_weights(model, base["nf4_checkpoint"], metadata, device)
    if "generation_config" in metadata:
        model.generation_config = GenerationConfig.from_dict(metadata["generation_config"])
    return model, metadata["quantized_source_parameters"]
