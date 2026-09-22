"""Distributed setup and validation shared by the CLIs."""
from __future__ import annotations

from datetime import timedelta
import json
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


def read_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
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
                     else (task.prerouter_state.valid_mask & batch["router_mask"][0]).sum())
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
            summary["curve"] = report["phases"]["teacher_forcing"]["global"]
        summaries[name] = summary
    if rank() == 0:
        print(json.dumps({"evaluation": summaries, "step": step}), flush=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"summary-step{step}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    task.set_training(True)
    return summaries
