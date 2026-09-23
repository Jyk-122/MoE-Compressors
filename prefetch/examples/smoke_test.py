"""Real-checkpoint forward/mask/gradient check on one H20."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from prefetch.datasets.dataset import to_device
from prefetch.backbone.loading import load_model
from prefetch.prerouter.patch import patch
from prefetch.training.runtime import make_collator, read_config, setup
from prefetch.training.train import TrainingTask
from prefetch.backbone.quantization import NF4Experts
from prefetch.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-file", required=True, help="Normalized JSONL; reads its first record")
    args = parser.parse_args()
    configure_logging()
    config = read_config(args.config)
    device = setup(config.get("seed", 42))
    with Path(args.sample_file).open(encoding="utf-8") as file:
        sample = json.loads(next(file))
    model, processor, loading = load_model(config, device)
    batch = to_device(make_collator(processor, config["data"])([sample]), device)
    ids, mask = batch["input_ids"][0], batch["router_mask"][0]
    logger.info("%s", json.dumps(dict(id=sample["id"], tokens=ids.numel(), router_tokens=int(mask.sum()),
                                    supervised_text=processor.tokenizer.decode(ids[mask].tolist())[:1000]), ensure_ascii=False))
    state = None
    if config["stage"] == "router":
        inputs = {k: v for k, v in batch.items() if k not in {"router_mask", "labels"}}
        kwargs = config["model"].get("router_forward_kwargs", {})
        with torch.no_grad():
            original = model(**inputs, use_cache=False, **kwargs).logits.detach()
        state = patch(model, config["prefetch"])
        with torch.no_grad():
            patched = model(**inputs, use_cache=False, **kwargs).logits.detach()
        torch.testing.assert_close(original, patched, rtol=0, atol=0)
        logger.info("Native logits preserved; output shape=%s", tuple(patched.shape))
        del original, patched
    elif config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    task = TrainingTask(model, state, config["stage"], config["model"].get("router_forward_kwargs"))
    task.set_training(True)
    loss = task(batch, record_metrics=True)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            gradients[name] = float(parameter.grad.norm())
        else:
            assert parameter.grad is None, name
    assert any(value > 0 for value in gradients.values()), "All trainable gradients are zero"
    optimizer = torch.optim.AdamW(task.trainables, lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    nf4 = next((module for module in model.modules() if isinstance(module, NF4Experts)), None)
    if nf4 is not None:
        linear = nf4.gate_up[0]
        assert linear.weight.dtype == torch.uint8 and linear.weight.quant_state is not None
        assert linear.weight.quant_state.blocksize == config["model"].get("nf4_blocksize", 64)
        values = torch.randn(2, linear.in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        linear(values).float().square().mean().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
        logger.info("NF4 packed weight and input gradient checked; blocksize=%d", linear.weight.quant_state.blocksize)
    logger.info("%s", json.dumps(dict(loss=float(loss.detach()), trainable_tensors=len(gradients), loading=loading,
                                    peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30), indent=2))
    if state:
        logger.info("%s", json.dumps(task.meter.report()["phases"]["teacher_forcing"]["global"], indent=2))


if __name__ == "__main__":
    main()
