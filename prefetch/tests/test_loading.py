"""Rank-local device placement and concurrent model-loading entrypoints."""

from datetime import timedelta
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.backbone import loading, nf4_checkpoint, quantization


@pytest.mark.parametrize("mode", ["bf16", "cpu_nf4", "packed_nf4"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("legacy_option", [False, True])
def test_rank_local_loading(monkeypatch, mode, rank, legacy_option):
    device = torch.device("cuda", rank)
    base = dict(path="source", quantization="none" if mode == "bf16" else "experts_nf4",
                nf4_blocksize=128, attn_implementation="sdpa")
    if mode == "packed_nf4":
        base["nf4_checkpoint"] = "packed"
    if legacy_option:
        base["serial_load"] = True
    model = nn.Linear(4, 4)
    events = []

    def from_pretrained(path, **kwargs):
        assert mode != "packed_nf4" and path == "source"
        assert kwargs["device_map"] == {"": device if mode == "bf16" else "cpu"}
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert kwargs["attn_implementation"] == "sdpa"
        assert kwargs["key_mapping"] == loading.KEY_MAPPING
        events.append("from_pretrained")
        return model

    def quantize(model_to_quantize, quantize_device, blocksize):
        assert mode == "cpu_nf4" and model_to_quantize is model
        assert quantize_device == "cpu" and blocksize == 128
        assert all(p.device.type == "cpu" and not p.requires_grad for p in model.parameters())
        events.append("quantize_cpu")
        return 96

    def move(target):
        assert mode == "cpu_nf4" and target == device
        assert events[-1] == "quantize_cpu"
        events.append("move_packed_to_gpu")
        return model

    def restore(config, target):
        assert mode == "packed_nf4" and target == device and config is base
        events.append("restore_packed")
        return model.requires_grad_(False), 96

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: "processor")))
    monkeypatch.setattr(model, "to", move)
    monkeypatch.setattr(quantization, "quantize_experts", quantize)
    monkeypatch.setattr(nf4_checkpoint, "load_nf4_checkpoint", restore)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: pytest.fail("Loading is rank-local"))
    result, processor, report = loading.load_model({"model": base}, device)
    assert result is model and processor == "processor" and not model.training
    assert report["quantized_source_parameters"] == (0 if mode == "bf16" else 96)
    expected = {"bf16": ["from_pretrained"], "cpu_nf4": ["from_pretrained", "quantize_cpu", "move_packed_to_gpu"],
                "packed_nf4": ["restore_packed"]}
    assert events == expected[mode]


def _loading_worker(rank, init_method, mode):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    model = nn.Linear(4, 4)

    def rendezvous():
        # Both ranks must enter the selected loading stage together.
        dist.monitored_barrier(timeout=timedelta(seconds=10))

    def from_pretrained(*args, **kwargs):
        if mode == "bf16":
            rendezvous()
        return model

    def quantize(*args):
        rendezvous()
        return 96

    def restore(*args):
        rendezvous()
        return model, 96

    transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
                                   AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: "processor"))
    config = dict(model=dict(path="source", serial_load=True,
                            quantization="none" if mode == "bf16" else "experts_nf4",
                            nf4_checkpoint="packed" if mode == "packed_nf4" else None))
    try:
        with patch.dict(sys.modules, transformers=transformers), \
                patch.object(quantization, "quantize_experts", quantize), \
                patch.object(nf4_checkpoint, "load_nf4_checkpoint", restore), \
                patch.object(torch.cuda, "memory_allocated", return_value=0), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            assert loading.load_model(config, "cpu")[0] is model
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_DDP_TESTS") != "1", reason="Set RUN_DDP_TESTS=1 for the two-process test")
@pytest.mark.parametrize("mode", ["bf16", "cpu_nf4", "packed_nf4"])
def test_two_ranks_enter_loading_concurrently(tmp_path, mode):
    torch.multiprocessing.spawn(_loading_worker, args=((tmp_path / "gloo_init").resolve().as_uri(), mode),
                                nprocs=2, join=True)
