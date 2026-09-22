import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.backbone.quantization import NF4Experts


@pytest.mark.parametrize("blocksize", [0, 32, 63, 96])
def test_nf4_rejects_unsupported_blocksize(blocksize):
    with pytest.raises(ValueError, match="blocksize"):
        NF4Experts(nn.Module(), "cpu", blocksize=blocksize)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 needs the server CUDA environment")
@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_expert_forward_and_input_gradient(blocksize):
    pytest.importorskip("bitsandbytes")
    original = nn.Module()
    original.num_experts = 3
    original.gate_up_proj = nn.Parameter(torch.randn(3, 128, 128, dtype=torch.bfloat16) * 0.02)
    original.down_proj = nn.Parameter(torch.randn(3, 128, 64, dtype=torch.bfloat16) * 0.02)
    original.act_fn = nn.functional.silu
    experts = NF4Experts(original, "cuda", blocksize=blocksize)
    for linear in list(experts.gate_up) + list(experts.down):
        assert linear.weight.dtype == torch.uint8 and linear.weight.quant_state is not None
        assert not linear.weight.requires_grad
        assert linear.weight.quant_state.blocksize == blocksize
    values = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], device="cuda")
    weights = torch.full((5, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    output = experts(values, indices, weights)
    output.float().square().mean().backward()
    assert output.shape == values.shape and torch.isfinite(output).all()
    assert values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
