"""Calibrate channel moments and directed substitution costs on native layer inputs."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ..data import make_loader, to_device
from ..utils import load_model_and_processor
from .qwen3_5_moe.patch_hf import expert_output, moe_blocks


FORMAT = "transfer_moe"
FIELDS = ("mean", "std", "error")


def save_calibration(states, path, base_model=""):
    from safetensors.torch import save_file

    tensors, records = {}, {}
    for index, state in states.items():
        prefix = f"layers.{index}"
        for name in FIELDS:
            tensors[f"{prefix}.{name}"] = state[name].detach().cpu().contiguous().clone()
        records[str(index)] = {name: state[name] for name in ("samples", "top_k", "epsilon")}
    metadata = {
        "format": FORMAT,
        "layers": records,
        "base_model": str(base_model)
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"transfer": json.dumps(metadata, ensure_ascii=False)})


def read_calibration(path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["transfer"])
    states = {}
    for index, record in metadata["layers"].items():
        states[int(index)] = {**record, **{name: tensors[f"layers.{index}.{name}"] for name in FIELDS}}
    return states, metadata


@torch.no_grad()
def calibrate_layer(block, hidden, epsilon=1e-5):
    """Use the same [samples, hidden] inputs for every expert.

    Compute each expert once and reuse its outputs for population moments and
    D[source, target]. The squared residual is expanded exactly into
    matrix products, preserving the target's per-channel variance weighting.
    This avoids an [experts, experts, samples, hidden] intermediate.
    """
    experts = block.experts
    hidden = hidden.to(experts.gate_up_proj)
    count = experts.gate_up_proj.shape[0]
    outputs = torch.stack([expert_output(experts, hidden, i).float() for i in range(count)])
    variance, mean = torch.var_mean(outputs, dim=1, correction=0)
    std = variance.sqrt().clamp_min(epsilon)

    centered = outputs - mean.unsqueeze(1)
    standardized = centered / std.unsqueeze(1)
    # ||u_j - sigma_j*z_i||² = ||u_j||² + ||sigma_j*z_i||² - 2<z_i, sigma_j*u_j>.
    source_square = standardized.square().sum(1) @ std.square().T
    target_square = centered.square().sum((1, 2)).unsqueeze(0)
    cross = standardized.flatten(1) @ (centered * std.unsqueeze(1)).flatten(1).T
    error = source_square.double() + target_square.double() - 2 * cross.double()
    error = (error / hidden.shape[0]).float().clamp_min_(0)
    error.fill_diagonal_(0)
    return {"mean": mean.cpu(), "std": std.cpu(), "error": error.cpu(),
            "samples": hidden.shape[0], "top_k": block.gate.top_k, "epsilon": epsilon}


@torch.no_grad()
def calibrate_model(model, loader, epsilon=1e-5):
    """Collect native MoE inputs, then calibrate one layer at a time.

    loader yields single-example processor batches with attention_mask.
    All non-padding tokens are used; no assistant-label masks are used.
    """
    blocks = moe_blocks(model)
    samples = {index: [] for index in blocks}
    handles = []
    was_training = model.training
    model.eval()
    try:
        for index, block in blocks.items():
            def capture(module, args, index=index):
                hidden = args[0].reshape(-1, args[0].shape[-1])
                hidden = hidden[mask.to(hidden.device)]
                samples[index].append(hidden.cpu())
            handles.append(block.register_forward_pre_hook(capture))

        device = model.get_input_embeddings().weight.device
        for batch in tqdm(loader, desc="Collecting layer inputs"):
            inputs = to_device(batch, device)
            mask = inputs["attention_mask"].reshape(-1).bool()
            model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    result = {}
    for index, block in tqdm(blocks.items(), desc="Calibrating expert pairs"):
        hidden = torch.cat(samples.pop(index))
        result[index] = calibrate_layer(block, hidden, epsilon)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="JSONL files with messages and optional images")
    parser.add_argument("--output", default="transfer_moe/outputs/calibration.safetensors")
    parser.add_argument("--device", default="cuda:0", help="Torch device or an HF device map such as auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-image-side", type=int, default=672)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    model, processor = load_model_and_processor(args.model, args.device, args.dtype)
    loader = make_loader(processor, args.data, args.samples, args.max_length, args.max_image_side)
    states = calibrate_model(model, loader, args.epsilon)
    save_calibration(states, args.output, base_model=args.model)
    print(f"Saved {len(states)} layers to {args.output}")


if __name__ == "__main__":
    main()
