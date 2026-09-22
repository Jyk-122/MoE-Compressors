"""Standalone teacher-forcing or autoregressive generation evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from prefetch.datasets.dataset import load_records, multimodal_messages, open_images, processor_call, to_device
from prefetch.backbone.loading import load_model
from prefetch.evaluation.metrics import save_report
from prefetch.prerouter.patch import patch
from prefetch.evaluation.plot import plot_report
from prefetch.evaluation.routing import RoutingMetrics, capture_generation
from prefetch.training.runtime import evaluate_task, rank, read_config, setup, world_size
from prefetch.training.train import TrainingTask


def experiment_config(config_path, checkpoint):
    saved = None
    if checkpoint and (Path(checkpoint) / "run_config.json").exists():
        saved = json.loads((Path(checkpoint) / "run_config.json").read_text(encoding="utf-8"))
    config = read_config(config_path) if config_path else saved
    if config is None:
        raise ValueError("Provide --config or a checkpoint containing run_config.json")
    if saved:
        for key in ("path", "quantization"):
            if config["model"].get(key) != saved["model"].get(key):
                raise ValueError(f"Checkpoint base model.{key} differs from evaluation config")
        if config.get("lora") != saved.get("lora"):
            raise ValueError("Use the same attention adapter as the predictor's training run")
    config["stage"] = "router"
    return config


def generation_inputs(processor, example, max_image_side=672):
    # Evaluate the first answer; the held-out reference answer is not in the prompt.
    messages = example["messages"]
    first_answer = next(i for i, message in enumerate(messages) if message["role"] == "assistant")
    paths = example.get("images") or []
    return processor_call(processor, multimodal_messages(messages[:first_answer], paths),
                          open_images(paths, max_image_side), generation=True)


@torch.no_grad()
def evaluate_generation(model, state, processor, config, device, output, max_new_tokens):
    model.eval()
    state.config.excluded_token_ids = list(set(state.config.excluded_token_ids) |
                                          set(processor.tokenizer.all_special_ids))
    meter = RoutingMetrics(state)
    for name, source in config["data"].get("validation", {}).items():
        dataset = load_records(source["path"], source.get("limit"))
        meter.reset()
        for index in range(rank(), len(dataset), world_size()):
            inputs = to_device(generation_inputs(processor, dataset[index], config["data"].get("max_image_side", 672)), device)
            with capture_generation(model, state, meter):
                model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                               max_new_tokens=max_new_tokens)
        report = meter.report(distributed=dist.is_initialized())
        report["evaluation"] = dict(source=name, examples=len(dataset), mode="first_answer_generation",
                                     max_new_tokens=max_new_tokens, prefill_tokens="all_text", decode_tokens="all_text")
        if rank() == 0:
            path = Path(output) / f"{name}-generation.json"
            save_report(report, path)
            plot_report(report, path.with_suffix(""))
            print(json.dumps({name: {p: v["global"] for p, v in report["phases"].items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["teacher_forcing", "generation"], default="teacher_forcing")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    config = experiment_config(args.config, args.checkpoint)
    device = setup(config.get("seed", 42))
    model, processor, _ = load_model(config, device)
    state = patch(model, checkpoint=args.checkpoint)
    if args.mode == "teacher_forcing":
        task = TrainingTask(model, state, "router", config["model"].get("router_forward_kwargs"))
        evaluate_task(task, processor, config, device, args.output, "final")
    else:
        evaluate_generation(model, state, processor, config, device, args.output, args.max_new_tokens)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
