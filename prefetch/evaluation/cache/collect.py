"""Collect full decode traces once; simulate multiple cache capacities offline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from prefetch.backbone.loading import load_model
from prefetch.datasets.dataset import load_records, to_device
from prefetch.evaluation.cache.trace import DecodeTrace
from prefetch.evaluation.evaluate import experiment_config, generation_inputs
from prefetch.prerouter.checkpoint import read_metadata
from prefetch.prerouter.patch import patch
from prefetch.training.runtime import rank, seed_all, setup, world_size


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True, help="Directory for one JSONL per validation source and rank")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if read_metadata(args.checkpoint)["config"]["mode"] != "same_token":
        parser.error("Cache evaluation currently requires a same_token checkpoint")
    config = experiment_config(args.config, args.checkpoint)
    device = setup(config.get("seed", 42))
    model, processor, _ = load_model(config, device)
    state = patch(model, checkpoint=args.checkpoint)
    model.eval()
    trace = DecodeTrace(state, prediction_k=8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for source, spec in config["data"].get("validation", {}).items():
        dataset = load_records(spec["path"], spec.get("limit"))
        indices = range(rank(), len(dataset), world_size())
        path = output / f"{source}-rank{rank()}.jsonl"
        metadata = dict(type="metadata", format="moe-cache-trace-v1", mode="same_token", phase="decode",
                        prediction_k=trace.prediction_k, layers=trace.layers, source=source,
                        checkpoint=str(Path(args.checkpoint).resolve()), model=config["model"],
                        lora=config.get("lora", {}), prefetch=asdict(state.config),
                        generation=dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                        seed=config.get("seed", 42),
                                        max_image_side=config["data"].get("max_image_side", 672)),
                        rank=rank(), world_size=world_size(), expected_requests=len(indices))
        with path.open("x", encoding="utf-8") as file:
            file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for index in tqdm(indices, desc=f"Cache trace {source} rank {rank()}", unit="request"):
                example = dataset[index]
                seed_all(config.get("seed", 42) + index)
                inputs = to_device(generation_inputs(processor, example,
                                   config["data"].get("max_image_side", 672)), device)
                with trace.capture(model):
                    generated = model.generate(**inputs, use_cache=True, do_sample=False, num_beams=1,
                                               max_new_tokens=args.max_new_tokens)
                record = trace.request(id=example["id"], sample_index=index,
                                       prompt_tokens=inputs["input_ids"].shape[1],
                                       generated_tokens=generated.shape[1] - inputs["input_ids"].shape[1])
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
            file.write(json.dumps(dict(type="complete", requests=len(indices))) + "\n")
        print(f"Saved cache trace: {path}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
