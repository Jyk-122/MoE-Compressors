"""Shared data loop and bounded-memory expert-output collection."""
from __future__ import annotations

from collections import defaultdict
from itertools import islice
import json
import os
from pathlib import Path

import torch


@torch.no_grad()
def expert_outputs(block, hidden, indices):
    """Compute each unique (token, expert) once in a single backend call."""
    tokens, slots = indices.shape
    # Expert-major keys batch the same expert together, including route overlaps.
    keys = indices * tokens + torch.arange(tokens, device=indices.device).unsqueeze(1)
    pairs, inverse = torch.unique(keys.flatten(), return_inverse=True)
    token_ids = pairs.remainder(tokens)
    expert_ids = pairs.div(tokens, rounding_mode="floor").unsqueeze(1)
    ones = torch.ones_like(expert_ids, dtype=torch.float32)
    values = block.experts(hidden[token_ids], expert_ids, ones).float()
    return values[inverse].view(tokens, slots, hidden.shape[-1])


class CalibrationObserver:
    def __init__(self, collector, chunk_size=32, max_tokens=512, targets=None):
        self.collector = collector
        self.chunk_size, self.max_tokens = chunk_size, max_tokens
        self.targets = None if targets is None else set(targets)
        self.tokens = defaultdict(int)

    @torch.no_grad()
    def __call__(self, target, block, hidden, native, predicted):
        if self.targets is not None and target not in self.targets:
            return
        stop = min(hidden.shape[0], self.max_tokens - self.tokens[target])
        for start in range(0, stop, self.chunk_size):
            end = min(start + self.chunk_size, stop)
            actual = tuple(t[start:end] for t in native)
            proposed = tuple(t[start:end] for t in predicted)
            x = hidden[start:end]
            if self.collector.sampling == "co_routed":
                u = v = expert_outputs(block, x, actual[0])
            else:
                ids = torch.cat([actual[0], proposed[0]], dim=1)
                outputs = expert_outputs(block, x, ids)
                u, v = outputs.split([actual[0].shape[1], proposed[0].shape[1]], dim=1)
            self.collector.update(target, actual, proposed, u, v)
            self.tokens[target] += end - start


def add_arguments(parser):
    parser.add_argument("--config", help="Run YAML; defaults to checkpoint run_config.json")
    parser.add_argument("--checkpoint", required=True, help="Trained prerouter checkpoint")
    parser.add_argument("--sample-file", required=True, help="Filtered calibration JSONL")
    parser.add_argument("--limit", type=int, default=128, help="Maximum calibration examples")
    parser.add_argument("--max-tokens", type=int, default=512, help="Total response-token cap per layer across all examples")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", help="Optional OWA / required ExFold and OptimalScale safetensors artifact")


def load_calibration(args):
    from prefetch.backbone.loading import load_model
    from prefetch.evaluation.evaluate import experiment_config
    from prefetch.prerouter.patch import patch
    from prefetch.utils.logging import configure_logging
    configure_logging()
    if min(args.limit, args.max_tokens, args.chunk_size) <= 0:
        raise ValueError("limit, max-tokens and chunk-size must be positive")
    config = experiment_config(args.config, args.checkpoint)
    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    model, processor, _ = load_model(config, device)
    model.eval().requires_grad_(False)
    state = patch(model, config.get("prefetch"), checkpoint=args.checkpoint,
                  execution_mode="native", compensation={})
    metadata = dict(layers=state.layers, model=config["model"], lora=config.get("lora"),
                    predictor=str(args.checkpoint), prediction_mode=state.config.mode,
                    distance=state.config.distance, sample_file=str(args.sample_file),
                    scope="native_trajectory_response_inputs")
    return model, processor, state, config, device, metadata


def calibration_batches(collate, args, rank=0, world_size=1):
    """Assign JSONL records round-robin before image processing and model forward."""
    with Path(args.sample_file).open(encoding="utf-8") as source:
        for index, line in enumerate(islice(source, args.limit)):
            if index % world_size == rank:
                yield index, collate([json.loads(line)])


def distributed_batches(batches, state, max_tokens, world_size):
    """Allocate the serial response-token prefix across independently prepared shards."""
    import torch.distributed as dist
    batches = list(batches)
    local_counts = []
    for index, batch in batches:
        state.reset(router_mask=batch["router_mask"])
        state.begin_forward(batch["input_ids"], batch.get("attention_mask"))
        mask = state.valid_mask & batch["router_mask"][0, state.target_start:].bool()
        local_counts.append((index, int(mask.sum())))
    state.reset()
    counts = [None] * world_size
    dist.all_gather_object(counts, local_counts)
    allowances = {}
    remaining = max_tokens
    for index, count in sorted(item for shard in counts for item in shard):
        allowances[index] = min(count, remaining)
        remaining -= allowances[index]
    batches = [(index, batch) for index, batch in batches if allowances[index] > 0]
    return batches, sum(allowances[index] for index, _ in batches)


@torch.no_grad()
def collect(model, processor, state, config, device, args, collector, *, targets=None,
            rank=0, world_size=1):
    from tqdm import tqdm
    from prefetch.datasets.dataset import to_device
    from prefetch.training.runtime import make_collator
    targets = state.targets if targets is None else set(targets)
    collate = make_collator(processor, config["data"])
    batches = calibration_batches(collate, args, rank, world_size)
    max_tokens, total = args.max_tokens, args.limit
    if world_size > 1:
        batches, max_tokens = distributed_batches(batches, state, max_tokens, world_size)
        total = len(batches)
    observer = CalibrationObserver(collector, args.chunk_size, max_tokens, targets)
    previous = state.route_observer
    state.route_observer = observer
    description = "Compensation calibration" + (f" rank {rank}" if world_size > 1 else "")
    try:
        for _, batch in tqdm(batches, total=total, desc=description):
            batch = to_device(batch, device)
            state.reset(router_mask=batch.pop("router_mask"))
            batch.pop("labels", None)
            forward_kwargs = dict(config["model"].get("router_forward_kwargs") or {})
            forward_kwargs["use_cache"] = False
            model(**batch, **forward_kwargs)
            if all(observer.tokens[target] >= max_tokens for target in targets):
                break
    finally:
        state.route_observer = previous
        state.reset()
    if world_size == 1 and any(observer.tokens[target] == 0 for target in targets):
        raise ValueError("Calibration needs valid response tokens for every target layer")
    return dict(observer.tokens)
