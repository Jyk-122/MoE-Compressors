"""Fit directed scalar projectors from co-routed expert outputs."""

import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from tqdm import tqdm

from ..data import make_loader, to_device
from ..utils import load_model_and_processor
from .qwen3_5_moe.patch_hf import expert_output, moe_blocks


FIELDS = ("scale", "loss", "norm", "pair_counts")


class PairStatistics:
    """Weighted <u,u>, <u,v>, <v,v>; u is omitted, v is retained.

    Only co-routed pairs are observed. Each token's weight is ||u||, not
    its router score. FP64 accumulators retain sequence-by-sequence sums.
    """

    def __init__(self, experts, device):
        self.experts = experts
        self.moments = torch.zeros(experts * experts, 3, dtype=torch.float64, device=device)
        self.counts = torch.zeros(experts * experts, dtype=torch.int64, device=device)
        self.norm_sum = torch.zeros(experts, dtype=torch.float64, device=device)
        self.tokens = 0

    def update(self, ids, outputs):
        outputs = outputs.float()
        gram = outputs @ outputs.transpose(1, 2)
        square = gram.diagonal(dim1=1, dim2=2)
        norm = square.sqrt()
        pairs = (ids.unsqueeze(2) * self.experts + ids.unsqueeze(1)).flatten()
        weight = norm.unsqueeze(2)
        source = (weight * square.unsqueeze(2)).expand_as(gram)
        cross = weight * gram
        target = weight * square.unsqueeze(1)
        values = torch.stack((source, cross, target), -1).reshape(-1, 3).double()
        self.moments.index_add_(0, pairs, values)
        self.counts.index_add_(0, pairs, torch.ones_like(pairs))
        self.norm_sum.index_add_(0, ids.flatten(), norm.flatten().double())
        self.tokens += ids.shape[0]

    def finalize(self, ridge=1e-3, scalar_clip=4.0):
        source, cross, target = self.moments.T.reshape(3, self.experts, self.experts)
        scale = (cross / (target + ridge)).clamp(-scalar_clip, scalar_clip)
        loss = ((source - 2 * scale * cross + scale.square() * target)
                / (source + 1e-12)).clamp_min(0)
        counts = self.counts.view(self.experts, self.experts)
        loss.masked_fill_(counts == 0, 1e30)
        scale.fill_diagonal_(1)
        loss.fill_diagonal_(0)
        norm = self.norm_sum / counts.diagonal().clamp_min(1)
        return {"scale": scale.float().cpu(), "loss": loss.float().cpu(),
                "norm": norm.float().cpu(), "pair_counts": counts.cpu(),
                "samples": self.tokens}


def save_calibration(states, path, base_model="", *, ridge=1e-3, scalar_clip=4.0):
    from safetensors.torch import save_file

    tensors = {f"layers.{index}.{name}": state[name].detach().cpu().contiguous()
               for index, state in states.items() for name in FIELDS}
    metadata = {"format": "exfold", "base_model": str(base_model),
                "ridge": ridge, "scalar_clip": scalar_clip,
                "layers": {str(index): {key: state[key] for key in ("samples", "top_k")}
                           for index, state in states.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"exfold": json.dumps(metadata, ensure_ascii=False)})


def read_calibration(path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["exfold"])
    states = {int(index): {**record, **{name: tensors[f"layers.{index}.{name}"] for name in FIELDS}}
              for index, record in metadata["layers"].items()}
    return states, metadata


@torch.no_grad()
def calibrate_model(model, loader, *, ridge=1e-3, scalar_clip=4.0):
    """Stream sequences once; keep pair statistics instead of an output cache.

    The temporary expert forward collects unweighted routed outputs while
    performing HF's native eager accumulation. Every routed token/expert pair
    is executed once. Only the current sequence's [tokens, K, hidden] outputs
    are retained until its pair statistics have been accumulated.
    """
    blocks = moe_blocks(model)
    statistics = {index: PairStatistics(block.gate.num_experts, block.gate.weight.device)
                  for index, block in blocks.items()}
    originals = {index: block.experts.forward for index, block in blocks.items()}
    was_training = model.training
    model.eval()
    try:
        for index, block in blocks.items():
            def collect(experts, hidden, ids, weights, index=index):
                outputs = hidden.new_empty((*ids.shape, hidden.shape[-1]))
                routed = torch.zeros_like(hidden)
                for expert_id in ids.unique():
                    token, slot = torch.where(ids == expert_id)
                    output = expert_output(experts, hidden[token], expert_id)
                    outputs[token, slot] = output
                    routed.index_add_(0, token, (output * weights[token, slot, None]).to(hidden.dtype))
                valid = mask.to(ids.device)
                statistics[index].update(ids[valid], outputs[valid])
                return routed

            block.experts.forward = MethodType(collect, block.experts)
        device = model.get_input_embeddings().weight.device
        for batch in tqdm(loader, desc="Calibrating ExFold"):
            inputs = to_device(batch, device)
            mask = inputs["attention_mask"].reshape(-1).bool()
            model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for index, block in blocks.items():
            block.experts.forward = originals[index]
        model.train(was_training)
    return {index: {**stats.finalize(ridge, scalar_clip), "top_k": blocks[index].gate.top_k}
            for index, stats in statistics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", default="transfer_moe/outputs/skip_exfold/calibration.safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-image-side", type=int, default=672)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--scalar-clip", type=float, default=4.0)
    args = parser.parse_args()
    if args.ridge <= 0 or args.scalar_clip <= 0:
        parser.error("ridge and scalar-clip must be positive")
    model, processor = load_model_and_processor(args.model, args.device, args.dtype)
    loader = make_loader(processor, args.data, args.samples, args.max_length, args.max_image_side)
    states = calibrate_model(model, loader, ridge=args.ridge, scalar_clip=args.scalar_clip)
    save_calibration(states, args.output, args.model, ridge=args.ridge, scalar_clip=args.scalar_clip)
    print(f"Saved {len(states)} ExFold layers to {args.output}")


if __name__ == "__main__":
    main()
