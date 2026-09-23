"""CPU tests check serialization contracts; CUDA tests use real bitsandbytes kernels."""
from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.backbone.loading import load_model
from prefetch.backbone.nf4_checkpoint import (load_nf4_checkpoint, read_nf4_metadata,
                                             restore_nf4_weights, save_nf4_checkpoint)
from prefetch.backbone.quantization import quantize_experts


class TinyMoE(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.gate = nn.Linear(128, 2, bias=False, device=device)
        self.experts = nn.Module()
        self.experts.num_experts = 2
        self.experts.act_fn = nn.functional.silu
        self.experts.gate_up_proj = nn.Parameter(torch.randn(2, 128, 128, device=device) * 0.02)
        self.experts.down_proj = nn.Parameter(torch.randn(2, 128, 64, device=device) * 0.02)

    def route_tokens_to_experts(self, logits):
        weights, indices = logits.softmax(-1).topk(1, dim=-1)
        return indices, weights

    def forward(self, x):
        indices, weights = self.route_tokens_to_experts(self.gate(x))
        return self.experts(x, indices, weights)


class TinyBase(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.moe = TinyMoE(device)
        self.embedding = nn.Embedding(4, 128, device=device)
        self.head = nn.Linear(128, 4, bias=False, device=device)
        self.register_buffer("persistent", torch.ones(1, device=device))
        self.register_buffer("rope", torch.arange(4, dtype=torch.float32), persistent=False)
        self.generation_config = SimpleNamespace(to_dict=lambda: {"eos_token_id": 3})
        self.tie_weights()

    def tie_weights(self):
        self.head.weight = self.embedding.weight

    def forward(self, x):
        return self.head(self.moe(x)) + self.persistent


@pytest.fixture
def fake_bnb(monkeypatch):
    """Preserve weights losslessly as bytes; emulate only the bnb serialization API."""
    class QuantState:
        def __init__(self, shape, blocksize):
            self.shape = torch.Size(shape)
            self.blocksize = blocksize
            self.quant_type, self.nested = "nf4", True

        def as_dict(self, packed=True):
            return {"quant_state.bitsandbytes__nf4": torch.tensor([*self.shape, self.blocksize])}

    class Params4bit(nn.Parameter):
        def __new__(cls, data, requires_grad=False, blocksize=64, module=None, **kwargs):
            obj = super().__new__(cls, data, requires_grad)
            obj.blocksize, obj.module = blocksize, module
            obj.quant_type, obj.quant_state = "nf4", None
            return obj

        def to(self, *args, **kwargs):
            if self.quant_state is None:
                self._quantize()
            return self

        def _quantize(self):
            self.quant_state = QuantState(self.shape, self.blocksize)
            self.data = self.detach().float().view(torch.uint8)
            self.module.quant_state = self.quant_state

        @classmethod
        def from_prequantized(cls, data, quantized_stats, device, module):
            height, width, blocksize = quantized_stats["quant_state.bitsandbytes__nf4"].tolist()
            obj = cls(data, blocksize=blocksize, module=module)
            obj.quant_state = QuantState((height, width), blocksize)
            module.quant_state = obj.quant_state
            return obj

    class Linear4bit(nn.Linear):
        def __init__(self, in_features, out_features, bias=False, device=None, **kwargs):
            super().__init__(in_features, out_features, bias=bias, device=device)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            if getattr(self.weight, "quant_state", None):
                for name, value in self.weight.quant_state.as_dict().items():
                    destination[prefix + "weight." + name] = value

        def forward(self, x):
            weight = self.weight.data.view(torch.float32).reshape(self.weight.quant_state.shape)
            return nn.functional.linear(x, weight.to(x.dtype))

    bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit, Params4bit=Params4bit), __version__="test")
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    return bnb


@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_checkpoint_cpu_roundtrip(tmp_path, fake_bnb, monkeypatch, blocksize):
    model = TinyBase().eval().requires_grad_(False)
    quantize_experts(model, "cpu", blocksize)
    x = torch.randn(3, 128)
    expected = model(x)
    directory = tmp_path / "nf4"
    metadata = save_nf4_checkpoint(model, directory, "source", blocksize, shard_bytes=1024)
    assert len(metadata["native_shards"]) > 1
    assert metadata["quantized_source_parameters"] == 2 * (128 * 128 + 128 * 64)
    assert metadata["generation_config"] == {"eos_token_id": 3}
    assert read_nf4_metadata(directory, "source", blocksize) == metadata
    with pytest.raises(FileExistsError):
        save_nf4_checkpoint(model, directory, "source", blocksize)
    with pytest.raises(ValueError, match="blocksize"):
        read_nf4_metadata(directory, "source", 128 if blocksize == 64 else 64)
    with pytest.raises(ValueError, match="source_model"):
        read_nf4_metadata(directory, "another-source", blocksize)

    def requantize(*args, **kwargs):
        pytest.fail("Loading a packed checkpoint must not requantize")

    monkeypatch.setattr(fake_bnb.nn.Params4bit, "_quantize", requantize)
    restored = restore_nf4_weights(TinyBase("meta"), directory, metadata, "cpu")
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert restored.head.weight is restored.embedding.weight
    assert not any(p.is_meta or p.requires_grad for p in restored.parameters())
    torch.testing.assert_close(restored.rope, model.rope)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    x.requires_grad_(True)
    restored(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_cache_loader_uses_config_skeleton(tmp_path, fake_bnb, monkeypatch):
    model = TinyBase().requires_grad_(False)
    quantize_experts(model, "cpu")
    directory = tmp_path / "nf4"
    save_nf4_checkpoint(model, directory, "source")

    @contextmanager
    def empty_weights(include_buffers):
        assert include_buffers is False
        yield

    def from_config(config, **kwargs):
        assert config == "model-config"
        return TinyBase("meta")

    def from_pretrained(*args, **kwargs):
        pytest.fail("Cached loading must not read BF16 model weights")

    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace(init_empty_weights=empty_weights))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: "model-config"),
        AutoModelForCausalLM=SimpleNamespace(from_config=from_config, from_pretrained=from_pretrained),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: "processor"),
        GenerationConfig=SimpleNamespace(from_dict=lambda value: value)))
    base = dict(path="source", nf4_checkpoint=str(directory), nf4_blocksize=64)
    restored, count = load_nf4_checkpoint(base, "cpu")
    assert restored.generation_config == {"eos_token_id": 3}
    assert count > 0
    import prefetch.backbone.quantization as quantization
    monkeypatch.setattr(quantization, "quantize_experts", from_pretrained)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    base["quantization"] = "experts_nf4"
    restored, processor, report = load_model({"model": base}, "cpu")
    assert processor == "processor"
    assert report["nf4_blocksize"] == 64 and report["nf4_checkpoint"] == str(directory)
    assert report["quantized_source_parameters"] == count
    assert not restored.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real NF4 roundtrip requires CUDA")
@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_checkpoint_cuda_roundtrip(tmp_path, monkeypatch, blocksize):
    bnb = pytest.importorskip("bitsandbytes")
    model = TinyBase().to(dtype=torch.bfloat16).eval().requires_grad_(False)
    quantize_experts(model, "cuda", blocksize)
    model.to("cuda")
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    expected = model(x)
    directory = tmp_path / "nf4"
    metadata = save_nf4_checkpoint(model, directory, "source", blocksize)

    def requantize(*args, **kwargs):
        pytest.fail("Loading must restore packed weights directly")

    monkeypatch.setattr(bnb.nn.Params4bit, "_quantize", requantize)
    restored = restore_nf4_weights(TinyBase("meta"), directory, metadata, "cuda")
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name].cpu(), value.cpu(), rtol=0, atol=0)
    x.requires_grad_(True)
    restored(x).float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
