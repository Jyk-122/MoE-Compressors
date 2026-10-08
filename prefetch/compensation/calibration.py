"""Shared data loop and bounded-memory expert-output collection."""
from __future__ import annotations

from collections import defaultdict
from itertools import islice
import json
from pathlib import Path

import torch


@torch.no_grad()
def expert_outputs(block, hidden, indices):
    """Unweighted [tokens, slots, hidden] outputs through the installed backend."""
    ones = torch.ones((hidden.shape[0], 1), device=hidden.device, dtype=torch.float32)
    return torch.stack([block.experts(hidden, indices[:, slot:slot + 1], ones).float()
                        for slot in range(indices.shape[1])], dim=1)


class CalibrationObserver:
    def __init__(self, collector, chunk_size=32, max_tokens=512):
        self.collector = collector
        self.chunk_size, self.max_tokens = chunk_size, max_tokens
        self.tokens = defaultdict(int)

    @torch.no_grad()
    def __call__(self, target, block, hidden, native, predicted):
        stop = min(hidden.shape[0], self.max_tokens - self.tokens[target])
        for start in range(0, stop, self.chunk_size):
            end = min(start + self.chunk_size, stop)
            actual = tuple(t[start:end] for t in native)
            proposed = tuple(t[start:end] for t in predicted)
            x = hidden[start:end]
            u = expert_outputs(block, x, actual[0])
            # co_routed ExFold uses native expert pairs only.
            v = (u if self.collector.sampling == "co_routed"
                 else expert_outputs(block, x, proposed[0]))
            self.collector.update(target, actual, proposed, u, v)
            self.tokens[target] += end - start


def add_arguments(parser):
    parser.add_argument("--config", help="Run YAML; defaults to checkpoint run_config.json")
    parser.add_argument("--checkpoint", required=True, help="Trained prerouter checkpoint")
    parser.add_argument("--sample-file", required=True, help="Filtered calibration JSONL")
    parser.add_argument("--limit", type=int, default=128, help="Maximum calibration examples")
    parser.add_argument("--max-tokens", type=int, default=512, help="Response tokens per target layer")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", help="Optional OWA / required ExFold safetensors artifact")


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
    device = torch.device("cuda", 0)
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


@torch.no_grad()
def collect(model, processor, state, config, device, args, collector):
    from tqdm import tqdm
    from prefetch.datasets.dataset import to_device
    from prefetch.training.runtime import make_collator
    observer = CalibrationObserver(collector, args.chunk_size, args.max_tokens)
    collate = make_collator(processor, config["data"])
    previous = state.route_observer
    state.route_observer = observer
    try:
        with Path(args.sample_file).open(encoding="utf-8") as source:
            for line in tqdm(islice(source, args.limit), total=args.limit, desc="Compensation calibration"):
                batch = to_device(collate([json.loads(line)]), device)
                state.reset(router_mask=batch.pop("router_mask"))
                batch.pop("labels", None)
                forward_kwargs = dict(config["model"].get("router_forward_kwargs") or {})
                forward_kwargs["use_cache"] = False
                model(**batch, **forward_kwargs)
                if all(observer.tokens[target] >= args.max_tokens for target in state.targets):
                    break
    finally:
        state.route_observer = previous
        state.reset()
    if any(observer.tokens[target] == 0 for target in state.targets):
        raise ValueError("Calibration needs valid response tokens for every target layer")
    return dict(observer.tokens)
