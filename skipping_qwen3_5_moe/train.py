"""Train coefficient matrices and moment transport with reconstruction and LM KL."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path

import torch

from .data import to_device
from .config import read_config, load_model_and_processor
from .losses import CompensationTrainingModel
from .model import configure_activation_checkpointing, load_compensation, save_compensation
from .routing import validate_threshold


def make_accelerator(training, cpu=False, expected_world_size=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not cpu and torch.cuda.is_available() and world_size > 1:
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} exceeds the visible GPU count")
        torch.cuda.set_device(local_rank)
        os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{local_rank}"
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, InitProcessGroupKwargs
    accelerator = Accelerator(
        cpu=cpu, mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=training["gradient_accumulation"], sync_with_dataloader=False),
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False, gradient_as_bucket_view=True),
                         InitProcessGroupKwargs(timeout=timedelta(hours=1))],
    )
    expected = world_size if expected_world_size is None else expected_world_size
    if accelerator.num_processes != expected:
        raise RuntimeError(f"Requested {expected} training processes, initialized {accelerator.num_processes}; "
                           "launch with python -m torch.distributed.run")
    if not cpu and torch.cuda.is_available():
        if accelerator.device.type != "cuda":
            raise RuntimeError(f"GPU training initialized on {accelerator.device}; check Accelerate's CPU settings")
        if world_size > 1 and accelerator.device != torch.device(f"cuda:{local_rank}"):
            raise RuntimeError(f"Rank {local_rank} must use cuda:{local_rank}, got {accelerator.device}")
        torch.cuda.set_device(accelerator.device)
    return accelerator


def make_task(model, config):
    options = config["training"]
    configure_activation_checkpointing(model, options.get("activation_checkpointing", True))
    return CompensationTrainingModel(
        model, threshold=config["skipping"]["threshold"], feature_weight=options["feature_weight"],
        kl_weight=options["kl_weight"], temperature=options["temperature"])


def metric_values(result):
    feature_count = result["feature_positions"]
    prediction_count = result["prediction_positions"]
    return torch.stack((result["feature_loss"] * feature_count, feature_count,
                        result["kl_loss"] * prediction_count, prediction_count,
                        *result["counts"].unbind())).detach().double()


def metric_report(totals, training):
    feature_sum, feature_count, kl_sum, kl_count, kept, original, positions, mixed = totals.tolist()
    feature = feature_sum / feature_count if feature_count else 0.0
    kl = kl_sum / kl_count if kl_count else 0.0
    return {"loss": training["feature_weight"] * feature + training["kl_weight"] * kl,
            "feature_loss": feature, "kl_loss": kl,
            "feature_positions": int(feature_count), "prediction_positions": int(kl_count),
            "pruning_ratio": 1 - kept / original if original else 0.0,
            "mean_retained": kept / positions if positions else 0.0,
            "coefficient_positions": int(mixed), "routed_positions": int(positions)}


def cuda_memory(device):
    if device.type != "cuda":
        return {}
    return {"allocated": torch.cuda.memory_allocated(device) / 2 ** 30,
            "reserved": torch.cuda.memory_reserved(device) / 2 ** 30,
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 2 ** 30}


def report_runtime(accelerator, model, task, output):
    from importlib.metadata import version
    report = {"event": "training_start", "rank": accelerator.process_index,
              "local_rank": accelerator.local_process_index, "world_size": accelerator.num_processes,
              "device": str(accelerator.device),
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "ddp": isinstance(task, torch.nn.parallel.DistributedDataParallel),
              "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
              "base_dtype": str(model.get_input_embeddings().weight.dtype),
              "activation_checkpointing": model.skipping_controller.activation_checkpointing,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "transformers": version("transformers"),
              "accelerate": version("accelerate"), "cuda_memory_gib": cuda_memory(accelerator.device)}
    if accelerator.device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(accelerator.device)
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        report["cudnn_sdpa"] = torch.backends.cuda.cudnn_sdp_enabled()
    output.mkdir(parents=True, exist_ok=True)
    (output / f"runtime-rank-{accelerator.process_index}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


@torch.no_grad()
def evaluate(task, loaders):
    model = task.model
    device = model.get_input_embeddings().weight.device
    result = {}
    for name, loader in loaders.items():
        total = torch.zeros(8, device=device, dtype=torch.float64)
        for batch in loader:
            values = task(to_device(batch, device))
            total += metric_values(values).to(device)
        if total[1].item() == 0:
            raise ValueError(f"Validation set {name} has no valid reconstruction positions")
        result[name] = metric_report(total, {"feature_weight": task.feature_weight, "kl_weight": task.kl_weight})
    return result


def train(model, loader, validation, config, accelerator):
    import yaml
    options = config["training"]
    if len(loader) == 0:
        raise ValueError("Training loader is empty")
    model.skipping_controller.threshold = validate_threshold(config["skipping"]["threshold"])
    output = Path(config["output_dir"])
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                                 encoding="utf-8")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"], foreach=False)
    warmup, steps = options["warmup_steps"], options["max_steps"]

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    task, optimizer, loader = accelerator.prepare(make_task(model, config), optimizer, loader)
    if accelerator.num_processes > 1 and not isinstance(task, torch.nn.parallel.DistributedDataParallel):
        raise RuntimeError("Multi-process compensation training requires a DistributedDataParallel model")
    raw_task = accelerator.unwrap_model(task)
    raw_task.model.eval()
    report_runtime(accelerator, raw_task.model, task, output)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(8, device=accelerator.device, dtype=torch.float64)
    step = 0
    while step < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with accelerator.accumulate(task):
            try:
                result = task(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError("Non-finite compensation loss")
                accelerator.backward(result["loss"])
                total += metric_values(result).to(accelerator.device)
                del result
            except RuntimeError as error:
                report = {"event": "training_error", "rank": accelerator.process_index, "step": step,
                          "device": str(accelerator.device), "input_shape": list(batch["input_ids"].shape),
                          "error": str(error), "cuda_memory_gib": cuda_memory(accelerator.device)}
                (output / f"error-rank-{accelerator.process_index}.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report), flush=True)
                raise
            finally:
                raw_task.model.skipping_controller.feature_terms.clear()
            if not accelerator.sync_gradients:
                continue
            norm = accelerator.clip_grad_norm_(parameters, options["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite compensation gradient")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % options["log_every"] == 0 or step == steps:
            combined = accelerator.reduce(total, reduction="sum")
            memory = cuda_memory(accelerator.device)
            if memory:
                values = torch.tensor(list(memory.values()), device=accelerator.device)
                values = accelerator.gather(values).reshape(-1, len(memory)).max(0).values.tolist()
                memory = dict(zip(memory, values))
            if accelerator.is_main_process:
                report = {"step": step, "threshold": config["skipping"]["threshold"],
                          "learning_rate": scheduler.get_last_lr()[0], "world_size": accelerator.num_processes,
                          **metric_report(combined, options)}
                if memory:
                    report["cuda_memory_gib_max"] = memory
                print(json.dumps(report), flush=True)
                with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report) + "\n")
            total.zero_()
            if memory:
                torch.cuda.reset_peak_memory_stats(accelerator.device)
        if validation and (step % options["eval_every"] == 0 or step == steps):
            if accelerator.is_main_process:
                report = evaluate(raw_task, validation)
                (output / "metrics").mkdir(exist_ok=True)
                (output / "metrics" / f"step-{step}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps({"step": step, "validation": report}), flush=True)
            accelerator.wait_for_everyone()
        if step % options["save_every"] == 0:
            if accelerator.is_main_process:
                save_compensation(raw_task.model, output / f"checkpoint-{step}.safetensors")
            accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_compensation(raw_task.model, output / "compensation.safetensors")
    accelerator.wait_for_everyone()
    return step


def main():
    from accelerate.utils import set_seed
    from .data import make_loader, validation_loaders
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Calibration or trained compensation artifact used for initialization")
    parser.add_argument("--expected-world-size", type=int, help="Verify the number of training processes")
    args = parser.parse_args()
    config = read_config(args.config)
    accelerator = make_accelerator(config["training"], expected_world_size=args.expected_world_size)
    print(json.dumps({"event": "process_ready", "rank": accelerator.process_index,
                      "local_rank": accelerator.local_process_index, "world_size": accelerator.num_processes,
                      "device": str(accelerator.device)}), flush=True)
    set_seed(config["seed"])
    model, processor = load_model_and_processor(config, accelerator.device)
    load_compensation(model, args.checkpoint or config["calibration"]["artifact"],
                      threshold=config["skipping"]["threshold"])
    with accelerator.main_process_first():
        loader = make_loader(processor, config["data"])
        validation = validation_loaders(processor, config["data"])
    train(model, loader, validation, config, accelerator)
    accelerator.end_training()


if __name__ == "__main__":
    main()
