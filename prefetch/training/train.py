"""torchrun --nproc_per_node=8 -m prefetch.training.train --config ..."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from prefetch.datasets.dataset import to_device, training_records
from prefetch.prerouter.checkpoint import save_predictor
from prefetch.backbone.loading import load_lora, load_model, save_lora
from prefetch.prerouter.patch import patch
from prefetch.backbone.structure import choice_scores
from prefetch.evaluation.routing import RoutingMetrics
from prefetch.training.runtime import (evaluate_task, make_collator, prepare_run_directory,
                                       rank, read_config, seed_all, setup, world_size)
from prefetch.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def router_loss(prediction, teacher, block, temperature, loss_kind):
    """KL(teacher || prediction), averaged over the selected tokens."""
    if loss_kind == "score_kl":
        prediction, teacher = choice_scores(prediction, block), choice_scores(teacher, block)
    elif loss_kind != "logit_kl":
        raise ValueError(f"Unknown loss: {loss_kind}")
    log_q = F.log_softmax(prediction.float() / temperature, dim=-1)
    log_p = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
    return F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)


def compute_prerouter_loss(state, router_mask):
    """Read the collected routing tensors after forward and compute the training loss."""
    if state.config.temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = state.valid_mask & router_mask[0].bool()
    losses = []
    for _, target in state.pairs:
        prediction = state.predictions[target][mask]
        teacher = state.router_logits[target][mask]
        block = state.blocks[target][1]
        loss = (router_loss(prediction, teacher, block, state.config.temperature, state.config.loss)
                if prediction.shape[0] else prediction.sum() * 0)
        losses.append(loss)
    return torch.stack(losses).mean()


class TrainingTask(nn.Module):
    """Run the frozen base, then calculate loss from PrerouterState inside DDP forward."""
    def __init__(self, model, state, stage, forward_kwargs=None):
        super().__init__()
        # DDP registers trainables only; frozen NF4 weights stay rank-local.
        object.__setattr__(self, "backbone", model)
        self.prerouter_state = state
        parameters = state.parameters() if state else (p for p in model.parameters() if p.requires_grad)
        self.trainables = nn.ParameterList(list(parameters))
        self.meter = RoutingMetrics(state) if state else None
        self.stage = stage
        self.forward_kwargs = forward_kwargs or {}

    def set_training(self, training):
        self.train(training)
        self.backbone.train(training and self.stage == "lora")
        if self.prerouter_state:
            for head in self.prerouter_state.prerouters.values():
                head.train(training)

    def forward(self, batch, record_metrics=False):
        if batch["input_ids"].shape[0] != 1:
            raise ValueError("Training requires batch size 1 on each rank")
        inputs = {k: v for k, v in batch.items() if k != "router_mask"}
        if self.stage == "lora":
            return self.backbone(**inputs, use_cache=False).loss
        inputs.pop("labels", None)
        state = self.prerouter_state
        state.reset(train_prerouter=torch.is_grad_enabled())
        with torch.no_grad():
            self.backbone(**inputs, use_cache=False, **self.forward_kwargs)
        loss = compute_prerouter_loss(state, batch["router_mask"])
        if record_metrics:
            self.meter.update(state, batch["router_mask"])
        return loss


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(task, config, optimizer, scheduler, step, epoch, next_batch, device):
    states = [None] * world_size()
    state = rng_state(device)
    if dist.is_initialized():
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    directory = Path(config["output_dir"]) / f"checkpoint-{step}"
    if rank() == 0:
        directory.mkdir(parents=True, exist_ok=True)
        if task.prerouter_state:
            save_predictor(task.prerouter_state, directory)
        else:
            save_lora(task.backbone, directory, config["lora"])
        (directory / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        step=step, epoch=epoch, next_batch=next_batch, rng=states,
                        world_size=world_size()), directory / "trainer_state.pt")
        logger.info("Saved %s", directory)
    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Trusted local checkpoint directory, including optimizer/RNG state")
    parser.add_argument("--limit", type=int, help="Maximum training examples after mixing, shared across ranks; overrides data.train_limit")
    args = parser.parse_args()
    configure_logging()
    config = read_config(args.config)
    if args.limit is not None:
        config["data"]["train_limit"] = args.limit
    limit = config["data"].get("train_limit")
    if limit is not None and limit < 1:
        parser.error("Training limit must be positive")
    device = setup(config.get("seed", 42))
    if limit is not None and limit < world_size():
        parser.error("Training limit must be at least the number of ranks; each rank needs one sample")
    stage = config.get("stage", "router")
    if stage not in {"router", "lora"}:
        raise ValueError("stage must be router or lora")
    output = prepare_run_directory(config, args.resume)
    if rank() == 0:
        logger.info("Output directory: %s", output)
    model, processor, loading_report = load_model(config, device)
    state = None
    if stage == "router":
        model.requires_grad_(False)
        state = patch(model, config["prefetch"], checkpoint=args.resume)
    else:
        if args.resume:
            load_lora(model, args.resume)
        if config["training"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    task = TrainingTask(model, state, stage, config["model"].get("router_forward_kwargs"))
    task.set_training(True)
    ddp = DDP(task, device_ids=[device.index], broadcast_buffers=False) if world_size() > 1 else task
    seed_all(config.get("seed", 42) + rank())
    dataset = training_records(config["data"])
    sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=True,
                                 seed=config.get("seed", 42), drop_last=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=make_collator(processor, config["data"]),
                        num_workers=config["data"].get("num_workers", 0),
                        generator=torch.Generator().manual_seed(config.get("seed", 42)))
    if not len(loader):
        raise ValueError("Training data must contain at least one sample per rank")
    training = config["training"]
    accumulation = training.get("gradient_accumulation", 8)
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    max_steps = training.get("max_steps") or steps_per_epoch * training.get("epochs", 1)
    epochs = math.ceil(max_steps / steps_per_epoch)
    optimizer = torch.optim.AdamW(task.trainables, lr=training["learning_rate"],
                                  weight_decay=training.get("weight_decay", 0.0))
    warmup = training.get("warmup_steps", 50)
    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1, (step - warmup) / max(1, max_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step = start_epoch = start_batch = 0
    if args.resume:
        trainer_state = torch.load(Path(args.resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        if trainer_state["world_size"] != world_size():
            raise ValueError("Exact resume requires the same world size")
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        step, start_epoch, start_batch = trainer_state["step"], trainer_state["epoch"], trainer_state["next_batch"]
        restore_rng(trainer_state["rng"][rank()], device)
    if rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (output / "loading_report.json").write_text(json.dumps(loading_report, indent=2), encoding="utf-8")
        logger.info("stage=%s; trainable=%s; samples=%d; world=%d; max_steps=%d",
                    stage, f"{sum(p.numel() for p in task.trainables):,}",
                    len(dataset), world_size(), max_steps)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    started = time.monotonic()
    last_saved = step if args.resume else -1
    last_position = (start_epoch, start_batch)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if step >= max_steps:
                break
            if epoch == start_epoch and index < start_batch:
                continue
            batch = to_device(batch, device)
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            boundary = (index + 1) % accumulation == 0 or index + 1 == len(loader)
            context = ddp.no_sync() if world_size() > 1 and not boundary else nullcontext()
            with context:
                loss = ddp(batch)
                scaled_loss = loss / group_size
                scaled_loss.backward()
            totals += torch.stack((loss.detach().double(), torch.ones((), device=device),
                                    batch["router_mask"].sum().double()))
            if not boundary:
                continue
            torch.nn.utils.clip_grad_norm_(task.trainables, training.get("max_grad_norm", 1.0), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_position = (epoch, index + 1)
            if step % training.get("log_every", 10) == 0:
                if dist.is_initialized():
                    dist.all_reduce(totals)
                entry = dict(step=step, epoch=epoch, loss=(totals[0] / totals[1]).item(),
                             examples=int(totals[1].item()), text_tokens=int(totals[2].item()),
                             learning_rate=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - started,
                             peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30)
                if rank() == 0:
                    logger.info("%s", json.dumps(entry))
                    with (output / "train.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(entry) + "\n")
                totals.zero_()
            if step % training.get("eval_every", 100) == 0:
                evaluate_task(task, processor, config, device, output / "metrics", step)
            if step % training.get("save_every", 100) == 0:
                save_checkpoint(task, config, optimizer, scheduler, step, *last_position, device)
                last_saved = step
        if step >= max_steps:
            break
    if step != last_saved:
        save_checkpoint(task, config, optimizer, scheduler, step, *last_position, device)
    evaluate_task(task, processor, config, device, output / "metrics", step)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
