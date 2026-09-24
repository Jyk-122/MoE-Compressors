"""Distributed setup and validation shared by the CLIs."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
import yaml

from prefetch.datasets.dataset import SFTCollator, load_records, to_device
from prefetch.evaluation.metrics import save_report


logger = logging.getLogger(__name__)


def format_duration(seconds):
    """Format whole elapsed seconds as HHH:MM:SS, with accumulated hours."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:03d}:{minutes:02d}:{seconds:02d}"


def estimate_eta(elapsed_seconds, completed_steps, remaining_steps):
    """Estimate time to the last optimizer step using this launch's mean step time."""
    if completed_steps <= 0:
        return dict(eta_seconds=None, eta_time=None)
    seconds = elapsed_seconds / completed_steps * max(remaining_steps, 0)
    return dict(eta_seconds=seconds, eta_time=format_duration(math.ceil(seconds)))


def format_training_log(entry):
    """Format console fields while preserving numeric values in the JSONL entry."""
    display = {key: value for key, value in entry.items()
               if key not in {"elapsed_seconds", "eta_seconds"}}
    display["peak_gpu_gib"] = f"{entry['peak_gpu_gib']:.2f}"
    display["learning_rate"] = f"{entry['learning_rate']:.2e}"
    return json.dumps(display)


def read_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.get("model", {}).pop("serial_load", None)  # Legacy loading-only option.
    lora = config.get("lora", {})
    if lora.get("enabled") and lora.get("checkpoint"):
        adapter = json.loads((Path(lora["checkpoint"]) / "lora_config.json").read_text(encoding="utf-8"))
        lora.update({key: adapter[key] for key in ("rank", "alpha", "dropout")})
    return config


def setup(seed=42):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    seed_all(seed)
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def prepare_run_directory(config, resume=None):
    """Use output_dir as the parent for new runs; resume keeps the saved run directory."""
    config.get("model", {}).pop("serial_load", None)
    if resume:
        saved = json.loads((Path(resume) / "run_config.json").read_text(encoding="utf-8"))
        saved.get("model", {}).pop("serial_load", None)
        config["output_dir"] = saved["output_dir"]
        if config != saved:
            raise ValueError("Resume requires the same run config; use a fresh run for changed experiments")
    else:
        directory = [None]
        if rank() == 0:
            mode = config["prefetch"].get("mode", "same_token") if config.get("stage", "router") == "router" else "lora"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(config["output_dir"]) / f"{mode}_{stamp}"
            output.mkdir(parents=True, exist_ok=False)
            directory[0] = str(output)
        if dist.is_initialized():
            dist.broadcast_object_list(directory, src=0)
        config["output_dir"] = directory[0]
    return Path(config["output_dir"])


def make_collator(processor, config):
    return SFTCollator(processor, config.get("max_length", 2048),
                       config.get("router_tokens", "assistant"), config.get("max_image_side", 672))


class EvalShard(Sampler):
    """No repeated padding examples; evaluation runs without DDP forward collectives."""
    def __init__(self, dataset):
        self.indices = range(rank(), len(dataset), world_size())

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@torch.no_grad()
def evaluate_task(task, processor, config, device, output_dir, step):
    task.set_training(False)
    summaries = {}
    for name, source in config["data"].get("validation", {}).items():
        dataset = load_records(source["path"], source.get("limit"))
        loader = DataLoader(dataset, batch_size=1, sampler=EvalShard(dataset),
                            collate_fn=make_collator(processor, config["data"]), num_workers=0)
        if task.prerouter_state:
            task.meter.reset()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in loader:
            batch = to_device(batch, device)
            loss = task(batch, record_metrics=True)
            count = ((batch["labels"][:, 1:] != -100).sum() if task.stage == "lora"
                     else (task.prerouter_state.valid_mask &
                           batch["router_mask"][0, task.prerouter_state.target_start:]).sum())
            totals += torch.stack((loss.double() * count, count.double()))
        if dist.is_initialized():
            dist.all_reduce(totals)
        summary = dict(loss=(totals[0] / totals[1]).item() if totals[1] else None,
                       valid_tokens=int(totals[1].item()), examples=len(dataset))
        if task.prerouter_state:
            report = task.meter.report(distributed=dist.is_initialized())
            report["evaluation"] = dict(source=name, step=step, **summary)
            if rank() == 0:
                path = Path(output_dir) / f"{name}-step{step}.json"
                save_report(report, path)
                from prefetch.evaluation.plot import plot_report
                plot_report(report, path.with_suffix(""))
            metrics = report["phases"]["teacher_forcing"]
            summary["curve"] = metrics["global"]
            summary["required_k"] = {k: v for k, v in metrics["required_k"]["global"].items()
                                     if k != "distribution"}
        summaries[name] = summary
    if rank() == 0:
        logger.info("%s", json.dumps({"evaluation": summaries, "step": step}))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"summary-step{step}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    task.set_training(True)
    return summaries
