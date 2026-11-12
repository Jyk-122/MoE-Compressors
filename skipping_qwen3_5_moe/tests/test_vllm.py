"""Reference, TP and CUDA checks for the vLLM compensation adapter."""
import copy
from datetime import timedelta
from importlib.util import find_spec
import json
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch as mock_patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from ..artifacts import save_artifact
from ..experts import expert_output
from ..model import patch
from ..modules import CompensateCoefficients, MomentCompensation
from ..routing import top_p_mask
from ..serve_vllm import make_command
from ..vllm_patch import install_compensation, validate_vllm_config
from .reference_moe import Qwen3_5MoeSparseMoeBlock


class TupleLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.reduce_results = False

    def forward(self, x):
        return F.linear(x, self.weight), None


class NativeSharedExpert(nn.Module):
    def __init__(self, reference, gate, rank, size):
        super().__init__()
        self.expert_gate = gate
        self.gate_up_proj = TupleLinear(torch.cat((reference.gate_proj.weight.chunk(size, 0)[rank],
                                                   reference.up_proj.weight.chunk(size, 0)[rank])))
        self.down_proj = TupleLinear(reference.down_proj.weight.chunk(size, 1)[rank])

    def forward(self, x):
        gate, up = self.gate_up_proj(x)[0].chunk(2, -1)
        output = self.down_proj(F.silu(gate) * up)[0]
        return output * self.expert_gate(x)[0].sigmoid()


class NativeMoE(nn.Module):
    """vLLM 0.24.0's replicated router and sharded gate/up/down interface."""

    def __init__(self, reference, rank=0, size=1):
        super().__init__()
        self.gate = TupleLinear(reference.gate.weight)
        self.shared_expert_gate = TupleLinear(reference.shared_expert_gate.weight)
        self.shared_expert = NativeSharedExpert(reference.shared_expert, self.shared_expert_gate, rank, size)
        self.experts = nn.Module()
        experts = self.experts.routed_experts = nn.Module()
        gate, up = reference.experts.gate_up_proj.chunk(2, 1)
        experts.w13_weight = nn.Parameter(torch.cat((gate.chunk(size, 1)[rank], up.chunk(size, 1)[rank]), 1))
        experts.w2_weight = nn.Parameter(reference.experts.down_proj.chunk(size, 2)[rank].clone())
        experts.quant_method = SimpleNamespace(unquantized_backend=SimpleNamespace(name="TRITON"))

    def forward(self, hidden):
        raise AssertionError("Expected the worker to install compensation")


def fixtures(rank=0, size=1, *, count=5, width=6, intermediate=8, top_k=4):
    torch.manual_seed(81)
    config = SimpleNamespace(model_type="qwen3_5_moe_text", hidden_act="silu", norm_topk_prob=True,
                             num_experts=count, hidden_size=width, moe_intermediate_size=intermediate,
                             num_experts_per_tok=top_k, shared_expert_intermediate_size=intermediate)
    reference = Qwen3_5MoeSparseMoeBlock(config)
    for parameter in reference.parameters():
        nn.init.normal_(parameter, std=0.4)
    native = NativeMoE(reference, rank, size)
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = native
    model.model.layers = nn.ModuleList([layer])
    state = {0: {"mean": torch.randn(count, width) * 0.2, "log_std": torch.randn(count, width) * 0.3,
                 "logits": torch.randn(count, count), "epsilon": 1e-5, "samples": 32, "top_k": top_k}}
    hidden = torch.randn(9, width)
    return model, reference, config, state, hidden


def reference_output(reference, state, hidden, threshold):
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = copy.deepcopy(reference)
    model.model.layers = nn.ModuleList([layer])
    patch(model, state, threshold)
    with torch.no_grad():
        return layer.mlp(hidden.unsqueeze(0)).squeeze(0)


def reduce_sum(tensor):
    dist.all_reduce(tensor)
    return tensor


def tp_worker(rank, directory, port):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        model, _, config, _, hidden = fixtures(rank, 2)
        reduce = Mock(wraps=reduce_sum)
        install_compensation(model, Path(directory) / "compensation.safetensors", 0.5, config, 2, reduce)
        output = model.model.layers[0].mlp(hidden)
        reduce.assert_called_once()
        assert reduce.call_args.args[0].shape == (len(hidden) * (config.num_experts_per_tok + 1), hidden.shape[1])
        torch.save(output, Path(directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


class VllmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_outputs_match_transformers_compensation_and_experts_are_skipped(self):
        for threshold in (0.1, 0.5, 0.999):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                model, reference, config, state, hidden = fixtures()
                path = Path(directory) / "compensation.safetensors"
                save_artifact(state, path)
                indices = install_compensation(model, path, threshold, config)
                self.assertEqual(indices, [0])
                block = model.model.layers[0].mlp
                self.assertIsInstance(block, NativeMoE)
                expected = reference_output(reference, state, hidden, threshold)
                with mock_patch("skipping_qwen3_5_moe.experts.expert_output", wraps=expert_output) as executed:
                    actual = block(hidden)
                    if threshold == 0.1:
                        self.assertEqual(sum(call.args[1].shape[0] for call in executed.call_args_list), len(hidden))
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                decode = block(hidden[:1])
                torch.testing.assert_close(decode, expected[:1], rtol=1e-6, atol=1e-6)
                self.assertEqual(block(hidden[:0]).shape, (0, 6))
                self.assertTrue(all(not p.requires_grad for p in block.compensation.parameters()))

    def test_all_retained_uses_original_forward_without_loading_device_state(self):
        for threshold, top_k in ((1.0, 4), (0.5, 1)):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                model, _, config, state, hidden = fixtures()
                state[0]["top_k"] = config.num_experts_per_tok = top_k
                block = model.model.layers[0].mlp
                path = Path(directory) / "compensation.safetensors"
                save_artifact(state, path, threshold=threshold)
                with mock_patch.object(block, "forward", return_value=hidden) as native:
                    self.assertEqual(install_compensation(model, path, None, config), [])
                    self.assertFalse(hasattr(block, "compensation"))
                    self.assertIs(block(hidden), hidden)
                    native.assert_called_once()

    def test_calibration_checkpoint_keeps_clamped_scales_and_uniform_coefficients(self):
        model, reference, config, state, hidden = fixtures()
        calibration = state[0]
        calibration["std"] = calibration.pop("log_std").exp()
        calibration["std"][0, 0] = 0.
        del calibration["logits"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.safetensors"
            save_artifact(state, path, threshold=0.65)
            install_compensation(model, path, None, config)
        block = model.model.layers[0].mlp
        torch.testing.assert_close(block.compensation.std, calibration["std"].clamp_min(calibration["epsilon"]))
        torch.testing.assert_close(block(hidden), reference_output(reference, state, hidden, 0.65))

    def test_layout_and_checkpoint_are_validated_before_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, config, state, _ = fixtures()
            path = Path(directory) / "compensation.safetensors"
            save_artifact(state, path)
            block = model.model.layers[0].mlp
            backend = block.experts.routed_experts.quant_method.unquantized_backend
            backend.name = "FLASHINFER_CUTLASS"
            with self.assertRaisesRegex(ValueError, "Triton weight layout"):
                install_compensation(model, path, 0.5, config)
            self.assertFalse(hasattr(block, "compensation"))
            backend.name = "TRITON"
            state[0]["top_k"] = 3
            save_artifact(state, path)
            with self.assertRaisesRegex(ValueError, "geometry"):
                install_compensation(model, path, 0.5, config)
            self.assertFalse(hasattr(block, "compensation"))

    def test_launcher_and_worker_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, text, state, _ = fixtures()
            path = Path(directory) / "compensation.safetensors"
            save_artifact(state, path)
            command, dry_run = make_command(["--model", "/models/qwen", "--checkpoint", str(path),
                                             "--tensor-parallel-size", "2", "--dry-run",
                                             "--api-key", "test-key"])
            self.assertTrue(dry_run)
            self.assertIn("--enforce-eager", command)
            self.assertEqual(command[command.index("--moe-backend") + 1], "triton")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
            self.assertIn("skipping_qwen3_5_moe.vllm_worker.CompensationWorker", command)
            options = json.loads(command[command.index("--additional-config") + 1])["skipping_compensation"]
            self.assertEqual(options, {"checkpoint": str(path.resolve()), "threshold": 0.5})
            config = SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=text, enforce_eager=True, enable_sleep_mode=False),
                parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_eplb=False,
                                                use_sequence_parallel_moe=False, pipeline_parallel_size=1,
                                                data_parallel_size=1),
                quant_config=None, compilation_config=SimpleNamespace(mode=0),
                kernel_config=SimpleNamespace(moe_backend="triton"), speculative_config=None, lora_config=None)
            self.assertIs(validate_vllm_config(config), text)
            config.parallel_config.enable_expert_parallel = True
            with self.assertRaisesRegex(ValueError, "replicated expert IDs"):
                validate_vllm_config(config)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU/Gloo")
    def test_two_tp_shards_match_complete_expert_transport(self):
        _, reference, config, state, hidden = fixtures()
        expected = reference_output(reference, state, hidden, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            save_artifact(state, Path(directory) / "compensation.safetensors")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            context = mp.spawn(tp_worker, args=(directory, port), nprocs=2, join=False)
            deadline = time.monotonic() + 90
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("TP test exceeded 90 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for rank in range(2):
                actual = torch.load(Path(directory) / f"rank-{rank}.pt", map_location="cpu")
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available() and find_spec("vllm"), "Requires CUDA and vLLM 0.24.0")
class VllmCudaTests(unittest.TestCase):
    def test_route_and_transport_for_mixed_prefixes(self):
        from ..vllm_kernels import routes, transport

        torch.manual_seed(52)
        experts, k, width = 7, 3, 70
        logits = torch.full((3, experts), -100., device="cuda", dtype=torch.float16)
        logits[:, [4, 0, 6]] = logits.new_tensor([[3., 0., -1.], [1., 0.8, 0.], [1., 1., 1.]])
        table = CompensateCoefficients(experts).cuda()
        with torch.no_grad():
            table.logits.normal_()
        probs, ids = logits.float().softmax(-1).topk(k)
        weights = (probs / probs.sum(-1, keepdim=True)).to(logits.dtype)
        keep = top_p_mask(weights, 0.7)
        self.assertEqual(keep.sum(-1).tolist(), [1, 2, 3])
        operator = MomentCompensation(torch.randn(experts, width), torch.rand(experts, width) + 0.1).cuda()
        outputs = torch.randn((3, k, width), device="cuda", dtype=logits.dtype)
        shared = torch.randn_like(outputs[:, 0])
        with torch.no_grad():
            actual_weights, actual_ids, dispatch, alpha = routes(logits, table.logits, k, 0.7)
            torch.testing.assert_close(actual_ids, ids)
            torch.testing.assert_close(actual_weights, weights, rtol=0, atol=0)
            torch.testing.assert_close(dispatch, ids.masked_fill(~keep, experts).int())
            coefficients = table(ids, keep)
            torch.testing.assert_close(alpha, coefficients)
            expected = operator.aggregate(outputs, ids, keep, weights, coefficients).to(shared.dtype) + shared
            actual = transport(outputs.masked_fill(~keep[:, :, None], torch.nan), ids, dispatch, weights,
                               alpha, operator.mean, operator.std, shared)
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_cuda_adapter_decode_and_prefill_match_reference(self):
        from .. import vllm_kernels

        # One token exercises vLLM's direct assignment, nine tokens its sorted blocks.
        model, reference, config, state, hidden = fixtures(count=32, width=80, intermediate=32, top_k=3)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(device="cuda", dtype=dtype)
        reference.to(device="cuda", dtype=dtype)
        hidden = hidden.to(device="cuda", dtype=dtype) * 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compensation.safetensors"
            save_artifact(state, path)
            install_compensation(model, path, 0.65, config)
        block = model.model.layers[0].mlp
        with mock_patch("skipping_qwen3_5_moe.vllm_patch.selected_outputs", side_effect=AssertionError):
            for x in (hidden[:1], hidden):
                expected = reference_output(reference, state, x, 0.65)
                with mock_patch.object(vllm_kernels, "expert_outputs", wraps=vllm_kernels.expert_outputs) as grouped:
                    actual = block(x)
                    grouped.assert_called_once()
                    dispatch, _, outputs = grouped.call_args.args[-3:]
                    skipped = outputs[dispatch == config.num_experts]
                    self.assertGreater(skipped.numel(), 0)
                    torch.testing.assert_close(skipped, torch.zeros_like(skipped), atol=0, rtol=0)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.006)


if __name__ == "__main__":
    unittest.main()
