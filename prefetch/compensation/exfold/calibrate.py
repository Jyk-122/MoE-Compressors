"""Fit directed scalar expert substitutions with norm-weighted least squares."""
from __future__ import annotations

import argparse
import json

import torch

from ..artifacts import save_artifact
from ..calibration import add_arguments, collect, load_calibration


class PairwiseStats:
    """CPU float64 accumulators; update consumes bounded [T,K,H] output chunks."""
    def __init__(self, experts):
        self.experts = experts
        self.dot = torch.zeros(experts, experts, dtype=torch.float64)
        self.target_sq = torch.zeros_like(self.dot)
        self.source_sq = torch.zeros_like(self.dot)
        self.count = torch.zeros(experts, experts, dtype=torch.long)

    @torch.no_grad()
    def update(self, source_ids, target_ids, source_outputs, target_outputs):
        u, v = source_outputs.double(), target_outputs.double()
        source_sq = u.square().sum(-1).unsqueeze(2)
        target_sq = v.square().sum(-1).unsqueeze(1)
        weight = source_sq.sqrt()
        dot = torch.bmm(u, v.transpose(1, 2))
        pairs = (source_ids.unsqueeze(2) * self.experts + target_ids.unsqueeze(1)).flatten().cpu()
        for accumulator, values in ((self.dot, weight * dot),
                                    (self.target_sq, weight * target_sq),
                                    (self.source_sq, (weight * source_sq).expand_as(dot))):
            accumulator.view(-1).index_add_(0, pairs, values.flatten().cpu())
        self.count.view(-1).index_add_(0, pairs, torch.ones_like(pairs))

    def finalize(self, ridge=1e-8, clip=4.0):
        if ridge < 0 or clip <= 0:
            raise ValueError("ridge must be nonnegative and clip must be positive")
        coefficient = (self.dot / (self.target_sq + ridge).clamp_min(1e-30)).clamp(-clip, clip)
        residual = self.source_sq - 2 * coefficient * self.dot + coefficient.square() * self.target_sq
        loss = (residual / self.source_sq.clamp_min(1e-30)).clamp_min(0)
        observed = (self.count > 0) & (self.source_sq > 0) & (self.target_sq > 0)
        coefficient = torch.where(observed, coefficient, torch.zeros_like(coefficient))
        loss = torch.where(observed, loss, torch.full_like(loss, 1e30))
        return dict(coefficients=coefficient.float(), loss=loss.float(), count=self.count.clone())


class ExFoldCalibration:
    def __init__(self, layers, sampling="prefetch"):
        self.sampling = sampling
        self.stats = {row["target_moe"]: PairwiseStats(row["experts"]) for row in layers}

    def update(self, target, native, predicted, native_outputs, candidate_outputs):
        candidates = native[0] if self.sampling == "co_routed" else predicted[0]
        self.stats[target].update(native[0], candidates, native_outputs, candidate_outputs)

    def result(self, ridge=1e-8, clip=4.0):
        tensors, coverage = {}, {}
        for target, stats in self.stats.items():
            values = stats.finalize(ridge, clip)
            tensors.update({f"{target}.{name}": value for name, value in values.items()})
            off_diagonal = ~torch.eye(stats.experts, dtype=torch.bool)
            valid = (values["loss"] < 1e30) & off_diagonal
            coverage[target] = dict(observed_pairs=int(valid.sum()),
                                    possible_pairs=int(off_diagonal.sum()))
        return tensors, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--sampling", choices=["prefetch", "co_routed"], default="prefetch")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=4.0)
    args = parser.parse_args()
    if not args.output:
        parser.error("ExFold calibration requires --output")
    model, processor, state, config, device, metadata = load_calibration(args)
    collector = ExFoldCalibration(state.layers, args.sampling)
    counts = collect(model, processor, state, config, device, args, collector)
    tensors, coverage = collector.result(args.ridge, args.clip)
    metadata.update(method="exfold", sampling=args.sampling, ridge=args.ridge, clip=args.clip,
                    tokens=counts, pair_coverage=coverage)
    save_artifact(args.output, tensors, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
