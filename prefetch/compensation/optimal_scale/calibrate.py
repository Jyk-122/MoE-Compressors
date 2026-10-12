"""Collect all MoE layers in one data pass, then solve per-expert channel scales."""
from __future__ import annotations

import argparse
import json
import logging

import torch

from ..artifacts import save_artifact
from ..calibration import add_arguments, collect, load_calibration


logger = logging.getLogger(__name__)


class ScaleStats:
    """Joint normal equations G[h] S[:,h] = q[h] for one layer."""
    sampling = "prefetch"

    def __init__(self, experts, hidden, channel_chunk_size=64):
        self.experts = experts
        self.channel_chunk_size = channel_chunk_size
        self.gram = torch.zeros(hidden, experts, experts, dtype=torch.float64)
        self.rhs = torch.zeros(hidden, experts, dtype=torch.float64)
        self.energy = self.baseline_error = 0.0
        self.tokens = 0

    @torch.no_grad()
    def update(self, target, native, predicted, native_outputs, predicted_outputs):
        _, native_weights = native
        ids, baseline_weights = predicted
        # Both routes already include the model's normalization and routed scaling.
        u = native_outputs.to(device="cpu", dtype=torch.float64)
        v = predicted_outputs.to(device="cpu", dtype=torch.float64)
        g = native_weights.to(device="cpu", dtype=torch.float64)
        b = baseline_weights.to(device="cpu", dtype=torch.float64)
        ids = ids.cpu()
        y = (u * g.unsqueeze(-1)).sum(1)
        features = v * b.unsqueeze(-1)
        self.energy += float(y.square().sum())
        self.baseline_error += float((features.sum(1) - y).square().sum())
        self.tokens += ids.shape[0]

        # Only the K selected columns of V are nonzero. Accumulate their K² cross terms.
        pairs = (ids.unsqueeze(2) * self.experts + ids.unsqueeze(1)).flatten()
        for start in range(0, y.shape[1], self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, y.shape[1])
            values = features[:, :, start:end].permute(2, 0, 1)  # [channels, tokens, K]
            products = values.unsqueeze(3) * values.unsqueeze(2)
            self.gram[start:end].view(end - start, -1).index_add_(1, pairs, products.flatten(1))
            rhs = values * y[:, start:end].T.unsqueeze(-1)
            self.rhs[start:end].index_add_(1, ids.flatten(), rhs.flatten(1))

    @torch.no_grad()
    def solve(self, ridge=1e-4, ridge_center=1.0):
        """Minimize ||V S - Y||² + ridge * ||S - ridge_center||²."""
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        scale = torch.empty_like(self.rhs)
        fitted_error = self.energy
        eye = torch.eye(self.experts, dtype=torch.float64)
        for start in range(0, self.rhs.shape[0], self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, self.rhs.shape[0])
            gram, rhs = self.gram[start:end], self.rhs[start:end]
            fitted = torch.linalg.solve(gram + ridge * eye,
                                        (rhs + ridge * ridge_center).unsqueeze(-1)).squeeze(-1)
            # Unobserved expert channels keep the predicted baseline.
            fitted = torch.where(gram.diagonal(dim1=-2, dim2=-1) > 0, fitted, 1.0)
            scale[start:end] = fitted
            fitted_error += float((fitted * torch.bmm(gram, fitted.unsqueeze(-1)).squeeze(-1)).sum()
                                  - 2 * (fitted * rhs).sum())
        if not torch.isfinite(scale).all():
            raise ValueError("OptimalScale solve produced non-finite scales")
        report = dict(tokens=self.tokens,
                      observed_experts=int((self.gram.diagonal(dim1=-2, dim2=-1).sum(0) > 0).sum()),
                      baseline_normalized_mse=self.baseline_error / max(self.energy, 1e-30),
                      normalized_mse=max(fitted_error, 0.0) / max(self.energy, 1e-30))
        return scale.T.float().contiguous(), report


class OptimalScaleCalibration:
    """Accumulate each target layer's normal equations during the same forward."""
    sampling = "prefetch"

    def __init__(self, shapes, channel_chunk_size=64):
        self.shapes = shapes
        self.channel_chunk_size = channel_chunk_size
        self.stats = {}

    def update(self, target, native, predicted, native_outputs, predicted_outputs):
        if target not in self.stats:
            self.stats[target] = ScaleStats(*self.shapes[target], self.channel_chunk_size)
        self.stats[target].update(target, native, predicted, native_outputs, predicted_outputs)


def calibrate_layers(model, processor, state, config, device, args, metadata):
    """Collect once, then solve, save and release each layer's G and q."""
    shapes = {row["target_moe"]: state.blocks[row["target_moe"]][1].gate.weight.shape
              for row in state.layers}
    stats_bytes = sum(hidden * experts * (experts + 1) * 8 for experts, hidden in shapes.values())
    logger.info("OptimalScale: collecting %s layers in one data pass, up to %s response tokens/layer; "
                "CPU FP64 G and q require %.2f GiB in total (plus temporary buffers)",
                len(shapes), args.max_tokens, stats_bytes / 1024**3)
    collector = OptimalScaleCalibration(shapes, args.channel_chunk_size)
    tokens = collect(model, processor, state, config, device, args, collector)

    tensors, completed, counts, metrics = {}, [], {}, {}
    report = dict(metadata, method="optimal_scale", ridge=args.ridge,
                  ridge_center=args.ridge_center, sampling="prefetch",
                  collection="single_pass", limit=args.limit, max_tokens=args.max_tokens)
    for row in state.layers:
        target = row["target_moe"]
        logger.info("OptimalScale layer %s: solving", target)
        stats = collector.stats.pop(target)
        scale, fitted = stats.solve(args.ridge, args.ridge_center)
        del stats  # Release this layer's G and q immediately after solving.
        tensors[f"{target}.scale"] = scale
        completed.append(row)
        counts[target] = tokens[target]
        metrics[target] = fitted
        report.update(layers=list(completed), tokens=dict(counts), fit_metrics=dict(metrics))
        save_artifact(args.output, tensors, report)
        logger.info("OptimalScale layer %s: %s", target, json.dumps(fitted))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(limit=32, max_tokens=65536)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--ridge-center", type=float, choices=[0.0, 1.0], default=1.0,
                        help="Regularize toward 1 (predicted baseline) or 0 (original ridge)")
    parser.add_argument("--channel-chunk-size", type=int, default=64,
                        help="Channels per statistics/solve chunk; CPU G and q cover all target layers")
    args = parser.parse_args()
    if not args.output:
        parser.error("OptimalScale requires --output")
    if args.ridge <= 0 or args.channel_chunk_size <= 0:
        parser.error("ridge and channel-chunk-size must be positive")
    model, processor, state, config, device, metadata = load_calibration(args)
    report = calibrate_layers(model, processor, state, config, device, args, metadata)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
