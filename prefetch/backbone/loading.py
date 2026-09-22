from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from prefetch.backbone.structure import find_moe_blocks


KEY_MAPPING = {
    r"^visual": "model.visual",
    r"^model(?!\.(language_model|visual|lm_head))": "model.language_model",
}


class LoRALinear(nn.Module):
    """Small attention adapter with FP32 trainable master weights."""
    def __init__(self, base, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        base.requires_grad_(False)

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            values = self.dropout(x)
            adapted = torch.nn.functional.linear(values if x.is_cuda else values.to(self.lora_A.dtype), self.lora_A)
            adapted = torch.nn.functional.linear(adapted, self.lora_B)
        base = self.base(x)
        return base + (adapted * self.scale).to(base.dtype)


def install_lora(model, config):
    # Identify the language decoder container from the actual MoE module path.
    paths = {name.rsplit(".", 2)[0] for name, _ in find_moe_blocks(model)}
    installed = []
    for path in sorted(paths):
        for index, layer in enumerate(model.get_submodule(path)):
            attn = layer.self_attn
            for projection in ("qkv_proj", "o_proj"):
                original = getattr(attn, projection, None)
                if not isinstance(original, nn.Linear):
                    raise ValueError(f"Expected language attention {path}.{index}.self_attn.{projection} Linear")
                setattr(attn, projection, LoRALinear(original, config["rank"], config["alpha"], config["dropout"]))
                installed.append(f"{path}.{index}.self_attn.{projection}")
    return installed


def lora_state(model):
    return {name: param.detach().cpu().contiguous() for name, param in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}


def save_lora(model, directory, config):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(lora_state(model), str(directory / "lora.safetensors"))
    (directory / "lora_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_lora(model, directory):
    from safetensors.torch import load_file
    values = load_file(str(Path(directory) / "lora.safetensors"))
    expected = {name for name, _ in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
    if expected != set(values):
        raise ValueError("LoRA checkpoint projections differ from this model")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in values:
                param.copy_(values[name])


def _load_one(config, device):
    from transformers import AutoModelForCausalLM, AutoProcessor
    from prefetch.backbone.quantization import quantize_experts
    base = config["model"]
    quantization = base.get("quantization", "none")
    checkpoint = base.get("nf4_checkpoint")
    blocksize = base.get("nf4_blocksize", 64)
    source_parameters = 0
    if quantization not in {"none", "experts_nf4"}:
        raise ValueError("quantization must be none or experts_nf4")
    if checkpoint:
        if quantization != "experts_nf4":
            raise ValueError("nf4_checkpoint requires model.quantization=experts_nf4")
        from prefetch.backbone.nf4_checkpoint import load_nf4_checkpoint
        model, source_parameters = load_nf4_checkpoint(base, device)
    else:
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                      device_map={"": "cpu"}, key_mapping=KEY_MAPPING)
        if base.get("attn_implementation"):
            kwargs["attn_implementation"] = base["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(base["path"], **kwargs)
        model.requires_grad_(False)
        if quantization == "experts_nf4":
            source_parameters = quantize_experts(model, device, blocksize)
        model.to(device)
    gc.collect()
    processor = AutoProcessor.from_pretrained(base["path"], trust_remote_code=True)
    lora = config.get("lora", {})
    if lora.get("enabled", False):
        adapter = lora.get("checkpoint")
        if adapter:
            stored = json.loads((Path(adapter) / "lora_config.json").read_text(encoding="utf-8"))
            for key in ("rank", "alpha", "dropout"):
                lora[key] = stored[key]
        install_lora(model, lora)
        if adapter:
            load_lora(model, adapter)
        if config.get("stage") != "lora":
            if not adapter:
                raise ValueError("Router/evaluation stage with LoRA enabled requires a trained lora.checkpoint")
            model.requires_grad_(False)
    elif config.get("stage") == "lora":
        raise ValueError("stage=lora requires lora.enabled=true")
    model.eval()
    report = dict(quantization=quantization, quantized_source_parameters=source_parameters,
                  nf4_blocksize=blocksize if quantization == "experts_nf4" else None,
                  nf4_checkpoint=checkpoint,
                  allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    print(json.dumps({"loading": report}), flush=True)
    return model, processor, report


def load_model(config, device):
    """BF16 loading can be serial; packed NF4 checkpoints load independently per rank."""
    base = config["model"]
    if dist.is_initialized() and base.get("serial_load", True) and not base.get("nf4_checkpoint"):
        result = None
        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                result = _load_one(config, device)
            dist.barrier()
        return result
    return _load_one(config, device)
