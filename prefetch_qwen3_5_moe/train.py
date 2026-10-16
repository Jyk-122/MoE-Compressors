"""python -m prefetch_qwen3_5_moe.train --config prefetch_qwen3_5_moe/configs/train.yaml"""
import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
from time import monotonic

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import (DistributedDataParallelKwargs, GradientAccumulationPlugin,
                              InitProcessGroupKwargs, set_seed)
import torch
from torch.utils.data import DataLoader
import yaml

from prefetch.datasets.dataset import SFTCollator, load_records, to_device, training_records
from .patch import patch, save_prefetch


def read_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    training = config["training"]
    for key in ("max_steps", "gradient_accumulation", "log_every", "eval_every", "save_every"):
        value = training[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        if training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training["warmup_steps"] < 0 or training["weight_decay"] < 0:
        raise ValueError("warmup_steps and weight_decay must be nonnegative")
    temperature = config["prefetch"]["temperature"]
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("prefetch.temperature must be finite and positive")
    if config["data"]["max_length"] <= 0:
        raise ValueError("data.max_length must be positive")
    if not config["data"]["train"]:
        raise ValueError("data.train must contain at least one source")
    return config


def make_accelerator(training, cpu=False):
    """每进程使用一张卡；累积满指定样本数后同步梯度并更新参数。"""
    return Accelerator(
        cpu=cpu, mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=training["gradient_accumulation"], sync_with_dataloader=False),
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        kwargs_handlers=[
            DistributedDataParallelKwargs(broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(hours=1)),
        ],
    )


def make_loaders(processor, data):
    """复用原训练程序的 JSONL 混合、图像处理和 assistant 位置标记。"""
    collator = SFTCollator(
        processor, max_length=data["max_length"],
        router_tokens=data.get("router_tokens", "assistant"),
        max_image_side=data.get("max_image_side"),
    )
    records = training_records(data)
    if len(records) == 0:
        raise ValueError("Training data is empty")
    train_loader = DataLoader(
        records, batch_size=1, shuffle=True, collate_fn=collator,
        num_workers=data.get("num_workers", 0),
        generator=torch.Generator().manual_seed(data.get("seed", 42)),
    )
    validation_loaders = {}
    for name, source in data.get("validation", {}).items():
        records = load_records(source["path"], source.get("limit"))
        validation_loaders[name] = DataLoader(
            records, batch_size=1, shuffle=False, collate_fn=collator,
            num_workers=data.get("num_workers", 0),
        )
    return train_loader, validation_loaders


def infinite_batches(loader):
    """每次 next() 取一个 batch，读完后重新遍历并打乱数据。"""
    while True:
        yield from loader


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def forward_batch(model, batch, temperature=1.0):
    """完整上下文参与 forward，assistant 文本位置参与路由损失。"""
    inputs = dict(batch)
    router_mask = inputs.pop("router_mask")
    inputs.pop("labels", None)
    model(**inputs, use_cache=False, logits_to_keep=1, output_router_logits=False)
    return model.router_cache.loss(router_mask=router_mask, temperature=temperature)


class PrefetchTrainingModel(torch.nn.Module):
    """将路由 loss 作为 forward 的返回值，让 DDP 跟踪预测头的梯度。"""
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, batch):
        return forward_batch(self.model, batch, temperature=self.temperature)


@torch.no_grad()
def evaluate(model, validation_loaders, prefetch_k=(8, 12, 16, 32), max_length=2048,
             temperature=1.0):
    """分别评估各验证集；损失和命中数量都按有效 token-layer 位置加权。"""
    model.eval()
    cache = model.router_cache
    device = model.get_input_embeddings().weight.device
    reports = {}

    for name, loader in validation_loaders.items():
        total_hits = {}
        for k in prefetch_k:
            total_hits[k] = 0
        num_positions = 0
        total_loss = 0.0
        for batch in loader:
            batch = to_device(batch, device)
            loss = forward_batch(model, batch, temperature=temperature)
            stats = cache.prefetch_hit_stats(prefetch_k, router_mask=batch["router_mask"])
            num_positions += stats["num_positions"]
            total_loss += loss.item() * stats["num_positions"]
            for k in total_hits:
                total_hits[k] += stats["total_hits"][k]
            cache.clear(max_tokens=max_length)

        if num_positions == 0:
            raise ValueError(f"No matched valid routes in validation set {name}")
        avg_hits = {}
        for k in total_hits:
            avg_hits[k] = total_hits[k] / num_positions
        reports[name] = {
            "loss": total_loss / num_positions,
            "num_positions": num_positions,
            "avg_hits": avg_hits,
        }
    return reports


def train(model, train_loader, validation_loaders, config, accelerator):
    """Accelerate 管理数据分片和梯度同步；step 表示一次 optimizer 更新。"""
    training = config["training"]
    if len(train_loader) == 0:
        raise ValueError("Training data is empty")
    max_steps = training["max_steps"]

    output = Path(config["output_dir"])
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    warmup = training["warmup_steps"]

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1, (step - warmup) / max(1, max_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    task = PrefetchTrainingModel(model, temperature=config["prefetch"]["temperature"])
    task, optimizer, train_loader = accelerator.prepare(task, optimizer, train_loader)
    train_iterator = infinite_batches(train_loader)
    model = accelerator.unwrap_model(task).model
    cache = model.router_cache
    model.eval()  # 主模型冻结；gate_prefetch 仍通过 autograd 训练。
    optimizer.zero_grad(set_to_none=True)
    step = 0
    totals = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    started = monotonic()

    while step < max_steps:
        batch = next(train_iterator)
        with accelerator.accumulate(task):
            loss = task(batch)
            accelerator.backward(loss)  # Accelerate 自动除以梯度累积步数。
            totals[0] += loss.detach()
            totals[1] += 1
            cache.clear(max_tokens=config["data"]["max_length"])

            if not accelerator.sync_gradients:
                continue
            grad_norm = accelerator.clip_grad_norm_(parameters, training["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gate_prefetch gradient")
            optimizer.step()
            scheduler.step()  # 每次 optimizer 更新后推进一次学习率。
            optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % training["log_every"] == 0 or step == max_steps:
            combined = accelerator.reduce(totals, reduction="sum")
            if accelerator.is_main_process:
                router_kl = (combined[0] / combined[1]).item()
                elapsed_seconds = monotonic() - started
                remaining_seconds = elapsed_seconds / step * (max_steps - step)
                entry = {
                    "step": step, "router_kl": router_kl,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "elapsed_time": format_duration(elapsed_seconds),
                    "eta_time": format_duration(remaining_seconds),
                }
                print(json.dumps(entry), flush=True)
                with (output / "train.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry) + "\n")
            totals.zero_()

        if validation_loaders and (step % training["eval_every"] == 0 or step == max_steps):
            # 主进程评估完整验证集，其他进程在屏障处等待。
            if accelerator.is_main_process:
                reports = evaluate(model, validation_loaders, config["prefetch"]["ks"],
                                   max_length=config["data"]["max_length"],
                                   temperature=config["prefetch"]["temperature"])
                print(json.dumps({"step": step, "validation": reports}), flush=True)
                metrics_dir = output / "metrics"
                metrics_dir.mkdir(exist_ok=True)
                (metrics_dir / f"step-{step}.json").write_text(
                    json.dumps(reports, indent=2), encoding="utf-8")
            accelerator.wait_for_everyone()

        if step % training["save_every"] == 0:
            if accelerator.is_main_process:
                save_prefetch(model, output / f"checkpoint-{step}.safetensors")
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_prefetch(model, output / "gate_prefetch.safetensors")
        print(f"Saved gate_prefetch: {output / 'gate_prefetch.safetensors'}", flush=True)
    accelerator.wait_for_everyone()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    accelerator = make_accelerator(config["training"])
    set_seed(config["seed"])

    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_config["path"])
    dtype = model_config.get("dtype", "auto")
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    model_kwargs = {"dtype": dtype, "device_map": {"": accelerator.device}}
    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_config["path"], **model_kwargs)
    patch(model, config["prefetch"]["mode_prefetch"])
    with accelerator.main_process_first():
        train_loader, validation_loaders = make_loaders(processor, config["data"])
    train(model, train_loader, validation_loaders, config, accelerator)
    accelerator.end_training()


if __name__ == "__main__":
    main()
