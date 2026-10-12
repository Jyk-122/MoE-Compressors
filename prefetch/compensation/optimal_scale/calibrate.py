"""Collect all MoE layers in one data pass, then solve per-expert channel scales."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from ..artifacts import load_artifact, save_artifact
from ..calibration import add_arguments, collect, load_calibration
from .shared import map_statistics, shared_statistics, statistics_lock


logger = logging.getLogger(__name__)


class ScaleStats:
    """Joint normal equations G[h] S[:,h] = q[h] for one layer."""
    sampling = "prefetch"

    def __init__(self, experts, hidden, channel_chunk_size=64, *, shared_dir=None, target=None):
        self.experts = experts
        self.channel_chunk_size = channel_chunk_size
        self.lock_path = None if shared_dir is None else Path(shared_dir) / f"{target}.lock"
        if shared_dir is None:
            self.gram = torch.zeros(hidden, experts, experts, dtype=torch.float64)
            self.rhs = torch.zeros(hidden, experts, dtype=torch.float64)
            self.totals = torch.zeros(3, dtype=torch.float64)
        else:
            self.gram, self.rhs, self.totals = map_statistics(shared_dir, target, experts, hidden)
        self.local_tokens = 0

    @property
    def tokens(self):
        return int(self.totals[2])

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
        totals = torch.tensor([float(y.square().sum()),
                               float((features.sum(1) - y).square().sum()), ids.shape[0]],
                              dtype=torch.float64)
        self.local_tokens += ids.shape[0]

        # Only the K selected columns of V are nonzero. Accumulate their K squared cross terms.
        pairs = (ids.unsqueeze(2) * self.experts + ids.unsqueeze(1)).flatten()
        for start in range(0, y.shape[1], self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, y.shape[1])
            values = features[:, :, start:end].permute(2, 0, 1)  # [channels, tokens, K]
            products = (values.unsqueeze(3) * values.unsqueeze(2)).flatten(1)
            rhs = (values * y[:, start:end].T.unsqueeze(-1)).flatten(1)
            with statistics_lock(self.lock_path):
                self.gram[start:end].view(end - start, -1).index_add_(1, pairs, products)
                self.rhs[start:end].index_add_(1, ids.flatten(), rhs)
        with statistics_lock(self.lock_path):
            self.totals += totals

    @torch.no_grad()
    def solve(self, ridge=1e-4, ridge_center=1.0):
        """Minimize ||V S - Y||² + ridge * ||S - ridge_center||²."""
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        scale = torch.empty_like(self.rhs)
        energy, baseline_error, _ = self.totals.tolist()
        fitted_error = energy
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
                      baseline_normalized_mse=baseline_error / max(energy, 1e-30),
                      normalized_mse=max(fitted_error, 0.0) / max(energy, 1e-30))
        return scale.T.float().contiguous(), report


class OptimalScaleCalibration:
    """Accumulate each target layer's normal equations during the same forward."""
    sampling = "prefetch"

    def __init__(self, shapes, channel_chunk_size=64, shared_dir=None):
        self.shapes = shapes
        self.shared_dir = shared_dir
        self.channel_chunk_size = channel_chunk_size
        self.stats = {}

    def layer_stats(self, target):
        if target not in self.stats:
            kwargs = {} if self.shared_dir is None else dict(shared_dir=self.shared_dir, target=target)
            self.stats[target] = ScaleStats(*self.shapes[target], self.channel_chunk_size, **kwargs)
        return self.stats[target]

    def update(self, target, native, predicted, native_outputs, predicted_outputs):
        self.layer_stats(target).update(target, native, predicted, native_outputs, predicted_outputs)


def shard_path(output, rank):
    output = Path(output)
    return output.with_name(f"{output.stem}.rank{rank}{output.suffix}")


def calibrate_layers(model, processor, state, config, device, args, metadata, *,
                     rank=0, world_size=1, shared_dir=None):
    """Forward this rank's samples into shared statistics, then solve global scales."""
    shapes = {row["target_moe"]: state.blocks[row["target_moe"]][1].gate.weight.shape
              for row in state.layers}
    stats_bytes = sum(hidden * experts * (experts + 1) * 8 for experts, hidden in shapes.values())
    logger.info("OptimalScale rank %s/%s: sample indices %s, %s, ... out of at most %s; "
                "CPU FP64 G and q require %.2f GiB in total (plus per-rank temporary buffers)",
                rank, world_size, rank, rank + world_size, args.limit, stats_bytes / 1024**3)
    collector = OptimalScaleCalibration(shapes, args.channel_chunk_size, shared_dir)
    started = perf_counter()
    collect(model, processor, state, config, device, args, collector, rank=rank, world_size=world_size)
    collection_seconds = perf_counter() - started
    logger.info("OptimalScale rank %s: collection finished in %.1f s", rank, collection_seconds)

    local_tokens = {target: stats.local_tokens for target, stats in collector.stats.items()}
    if world_size > 1:
        dist.barrier()  # Every sample's contributions are now in the shared G and q.
        owned = {row["target_moe"] for row in state.layers[rank::world_size]}
        collector.stats = {target: stats for target, stats in collector.stats.items() if target in owned}
        dist.barrier()  # Other workers have released the mappings this rank will solve.

    output = args.output if world_size == 1 else shard_path(args.output, rank)
    tensors, completed, counts, metrics = {}, [], {}, {}
    report = dict(metadata, method="optimal_scale", ridge=args.ridge,
                  ridge_center=args.ridge_center, sampling="prefetch", parallelism="samples",
                  collection="single_pass", limit=args.limit, max_tokens=args.max_tokens,
                  rank=rank, world_size=world_size, local_tokens=local_tokens,
                  collection_seconds=collection_seconds, statistics_storage="shared_cpu" if shared_dir else "cpu")
    for row in state.layers[rank::world_size]:
        target = row["target_moe"]
        logger.info("OptimalScale layer %s: solving", target)
        collector.layer_stats(target)
        stats = collector.stats.pop(target)
        if stats.tokens == 0:
            raise ValueError("Calibration needs valid response tokens for every target layer")
        started = perf_counter()
        scale, fitted = stats.solve(args.ridge, args.ridge_center)
        fitted["solve_seconds"] = perf_counter() - started
        del stats  # Release this layer's G and q immediately after solving.
        if shared_dir is not None:
            (Path(shared_dir) / f"{target}.bin").unlink()
        tensors[f"{target}.scale"] = scale
        completed.append(row)
        counts[target] = fitted["tokens"]
        metrics[target] = fitted
        report.update(layers=list(completed), tokens=dict(counts), fit_metrics=dict(metrics))
        save_artifact(output, tensors, report)
        logger.info("OptimalScale layer %s: %s", target, json.dumps(fitted))
    if not completed:
        report.update(layers=[], tokens={}, fit_metrics={})
        save_artifact(output, tensors, report)
    return report


def merge_shards(output, layers, world_size):
    """Combine disjoint layers' scales after every rank has finished solving."""
    tensors, counts, metrics, durations, local_tokens = {}, {}, {}, {}, {}
    for rank in range(world_size):
        values, report = load_artifact(shard_path(output, rank))
        if report["layers"] != layers[rank::world_size]:
            raise ValueError(f"Incomplete target layers in calibration rank {rank}")
        tensors.update(values)
        counts.update(report["tokens"])
        metrics.update(report["fit_metrics"])
        durations[rank] = report["collection_seconds"]
        local_tokens[rank] = report["local_tokens"]
    report.pop("rank")
    report.update(layers=layers, tokens=counts, fit_metrics=metrics,
                  collection_seconds=max(durations.values()), rank_collection_seconds=durations,
                  rank_local_tokens=local_tokens)
    report.pop("local_tokens")
    save_artifact(output, tensors, report)
    logger.info("OptimalScale: merged %s layers into %s", len(layers), output)
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
    parser.add_argument("--shared-stats-dir", help="Single-host shared statistics directory; Linux default: /dev/shm")
    args = parser.parse_args()
    if not args.output:
        parser.error("OptimalScale requires --output")
    if args.ridge <= 0 or args.channel_chunk_size <= 0:
        parser.error("ridge and channel-chunk-size must be positive")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
            parser.error("Shared statistics require all calibration workers on one host")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
    try:
        model, processor, state, config, device, metadata = load_calibration(args)
        shapes = {row["target_moe"]: state.blocks[row["target_moe"]][1].gate.weight.shape
                  for row in state.layers}
        storage = shared_statistics(shapes, args.shared_stats_dir) if world_size > 1 else nullcontext(None)
        with storage as directory:
            report = calibrate_layers(model, processor, state, config, device, args, metadata,
                                      rank=rank, world_size=world_size, shared_dir=directory)
            if world_size > 1:
                dist.barrier()
                if rank == 0:
                    report = merge_shards(args.output, state.layers, world_size)
        if rank == 0:
            print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
