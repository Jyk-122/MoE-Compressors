"""CPU checks for vLLM weight layout, compensation, and TP output reduction."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from ..artifacts import save_artifact
from ..experts import expert_output
from ..model import patch
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


def fixtures(rank=0, size=1):
    torch.manual_seed(81)
    config = SimpleNamespace(model_type="qwen3_5_moe_text", hidden_act="silu", norm_topk_prob=True,
                             num_experts=5, hidden_size=6, moe_intermediate_size=8,
                             num_experts_per_tok=4, shared_expert_intermediate_size=8)
    reference = Qwen3_5MoeSparseMoeBlock(config)
    for parameter in reference.parameters():
        nn.init.normal_(parameter, std=0.4)
    native = NativeMoE(reference, rank, size)
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = native
    model.model.layers = nn.ModuleList([layer])
    state = {0: {"mean": torch.randn(5, 6) * 0.2, "log_std": torch.randn(5, 6) * 0.3,
                 "logits": torch.randn(5, 5), "epsilon": 1e-5, "samples": 32, "top_k": 4}}
    hidden = torch.randn(9, 6)
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
        install_compensation(model, Path(directory) / "compensation.safetensors", 0.5, config, 2, reduce_sum)
        output = model.model.layers[0].mlp(hidden)
        torch.save(output, Path(directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


class VllmTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_outputs_match_transformers_compensation_and_experts_are_skipped(self):
        for threshold in (0.1, 0.5, 1.0):
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


if __name__ == "__main__":
    unittest.main()
