import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.backbone.quantization import NF4Experts


@pytest.mark.parametrize("blocksize", [0, 32, 63, 96])
def test_nf4_rejects_unsupported_blocksize(blocksize):
    with pytest.raises(ValueError, match="blocksize"):
        NF4Experts(nn.Module(), "cpu", blocksize=blocksize)


def expert_parameters():
    original = nn.Module()
    original.num_experts = 3
    original.gate_up_proj = nn.Parameter(torch.randn(3, 128, 128, dtype=torch.bfloat16) * 0.02)
    original.down_proj = nn.Parameter(torch.randn(3, 128, 64, dtype=torch.bfloat16) * 0.02)
    original.act_fn = nn.functional.silu
    return original


@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_cpu_quantization(blocksize):
    pytest.importorskip("bitsandbytes")
    experts = NF4Experts(expert_parameters(), "cpu", blocksize=blocksize)
    for linear in list(experts.gate_up) + list(experts.down):
        state = linear.weight.quant_state
        assert linear.weight.device.type == state.absmax.device.type == "cpu"
        assert linear.weight.dtype == torch.uint8 and linear.weight.bnb_quantized
        assert linear.weight.numel() * 2 == state.shape.numel()
        assert state.blocksize == blocksize and state.nested
        assert state.state2.absmax.device.type == state.offset.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 inference needs the server CUDA environment")
@pytest.mark.parametrize("quantize_device", ["cpu", "cuda"])
@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_expert_forward_and_input_gradient(monkeypatch, blocksize, quantize_device):
    bnb = pytest.importorskip("bitsandbytes")
    experts = NF4Experts(expert_parameters(), quantize_device, blocksize=blocksize)
    packed = [linear.weight.detach().cpu().clone() for linear in list(experts.gate_up) + list(experts.down)]

    def requantize(*args):
        pytest.fail("Moving packed weights must preserve their quantization")

    monkeypatch.setattr(bnb.nn.Params4bit, "_quantize", requantize)
    experts.to("cuda")
    for linear, before in zip(list(experts.gate_up) + list(experts.down), packed):
        assert linear.weight.dtype == torch.uint8 and linear.weight.quant_state is not None
        assert linear.weight.device.type == linear.weight.quant_state.absmax.device.type == "cuda"
        torch.testing.assert_close(linear.weight.detach().cpu(), before, rtol=0, atol=0)
        assert not linear.weight.requires_grad
        assert linear.weight.quant_state.blocksize == blocksize
    values = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], device="cuda")
    weights = torch.full((5, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    output = experts(values, indices, weights)
    output.float().square().mean().backward()
    assert output.shape == values.shape and torch.isfinite(output).all()
    assert values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
