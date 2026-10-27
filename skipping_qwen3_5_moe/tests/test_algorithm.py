import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch as mock_patch

import torch

from ..artifacts import read_artifact, save_artifact
from ..calibrate import calibrate_outputs, calibrate_model
from ..experts import expert_output
from ..model import patch, save_compensation, load_compensation
from ..modules import MomentCompensation, CompensateCoefficients
from ..routing import top_p_mask
from .helpers import TinyModel, calibration_for, batch


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_top_p_boundary_and_minimum(self):
        q = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.8, 0.1, 0.06, 0.04]])
        self.assertEqual(top_p_mask(q, 0.5).sum(-1).tolist(), [2, 1])
        self.assertEqual(top_p_mask(q, 0.4).sum(-1).tolist(), [1, 1])
        self.assertTrue(top_p_mask(q, 1).all())
        self.assertEqual(top_p_mask(q, 1e-8).sum(-1).tolist(), [1, 1])
        for invalid in (0, -1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                top_p_mask(q, invalid)

    def test_calibration_initializes_trainable_population_moments(self):
        outputs = torch.randn(23, 4, 9) * torch.arange(1, 10)
        outputs[:, 0, 0] = 2  # A constant channel exercises the scale floor.
        state = calibrate_outputs(outputs)
        mu = outputs.double().mean(0)
        std = outputs.double().var(0, unbiased=False).sqrt().clamp_min(1e-5)
        torch.testing.assert_close(state["mean"], mu.float())
        torch.testing.assert_close(state["std"], std.float())
        operator = MomentCompensation(state["mean"], state["std"])
        self.assertEqual(set(dict(operator.named_parameters())), {"mean", "log_std"})
        torch.testing.assert_close(operator.mean, mu.float())
        torch.testing.assert_close(operator.std, std.float())
        with torch.no_grad():
            operator.log_std.fill_(-80)
        self.assertTrue((operator.std >= operator.epsilon).all())

    def test_parallel_compensation_matches_pairwise_reference_and_gradients(self):
        operator = MomentCompensation(torch.randn(5, 6), torch.rand(5, 6) + 0.2)
        policy = CompensateCoefficients(5)
        ids = torch.tensor([[0, 1, 3, 4], [2, 4, 1, 0]])
        keep = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        uniform = policy(ids, keep)
        torch.testing.assert_close(uniform[0, :2], torch.full((2, 4), 0.5))
        torch.testing.assert_close(uniform[1, 0], torch.ones(4))
        with torch.no_grad():
            policy.logits.normal_()
        outputs = (torch.randn(2, 4, 6) * keep.unsqueeze(-1)).requires_grad_()
        weights = torch.rand(2, 4).softmax(-1)
        alpha = policy(ids, keep)
        torch.testing.assert_close(alpha.sum(1), torch.ones(2, 4))
        self.assertTrue((alpha[~keep] == 0).all())
        actual = operator.aggregate(outputs, ids, keep, weights, alpha)
        expected = []
        for t in range(2):
            value = torch.zeros(6)
            for j in range(4):
                if keep[t, j]:
                    prediction = outputs[t, j]
                else:
                    sources = keep[t].nonzero(as_tuple=True)[0]
                    reference_alpha = policy.logits[ids[t, sources], ids[t, j]].softmax(0)
                    prediction = sum(a * operator(outputs[t, i], ids[t, i], ids[t, j])
                                     for a, i in zip(reference_alpha, sources))
                value = value + weights[t, j] * prediction
            expected.append(value)
        expected = torch.stack(expected)
        torch.testing.assert_close(actual, expected)
        variables = (policy.logits, outputs, operator.mean, operator.log_std)
        grad_a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
        grad_b = torch.autograd.grad(expected.square().sum(), variables)
        for a, b in zip(grad_a, grad_b):
            torch.testing.assert_close(a, b)

    def test_native_equivalence_sparse_dispatch_and_local_reference_cost(self):
        model = TinyModel().eval()
        native = copy.deepcopy(model)
        runtime = patch(model, calibration_for(model))
        data = batch()
        with torch.no_grad():
            actual = model(data["input_ids"], attention_mask=data["attention_mask"], threshold=1).logits
            expected = native(data["input_ids"], attention_mask=data["attention_mask"]).logits
        torch.testing.assert_close(actual, expected)
        self.assertEqual(runtime.threshold, 0.5)
        block = runtime.blocks[0]
        x = torch.randn(1, 7, 6)
        with mock_patch("skipping_qwen3_5_moe.experts.expert_output", wraps=expert_output) as execute:
            with runtime.context(mode="student", reset=True), torch.no_grad():
                block(x)
            self.assertEqual(sum(call.args[1].shape[0] for call in execute.call_args_list), 7 * 2)
        with mock_patch("skipping_qwen3_5_moe.experts.expert_output", wraps=expert_output) as execute:
            with runtime.context(mode="reconstruction", reset=True):
                block(x)
            self.assertEqual(sum(call.args[1].shape[0] for call in execute.call_args_list), 7 * 4)
        self.assertEqual(runtime.statistics[0].tolist(), [14, 28, 7, 7])

    def test_artifact_round_trip_and_threshold_scope(self):
        model = TinyModel()
        native = copy.deepcopy(model)
        calibration = calibration_for(model)
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / "calibration.safetensors"
            save_artifact(calibration, initial)
            state, metadata = read_artifact(initial)
            self.assertEqual(metadata["kind"], "calibration")
            runtime = load_compensation(model, initial)
            for block in runtime.blocks:
                torch.testing.assert_close(block.operator.mean, state[block.layer_index]["mean"])
                torch.testing.assert_close(block.operator.std, state[block.layer_index]["std"])
                torch.testing.assert_close(block.coefficients.logits, torch.zeros(5, 5))
            with torch.no_grad():
                for block in runtime.blocks:
                    block.coefficients.logits.normal_()
                    block.operator.mean.add_(torch.randn_like(block.operator.mean) * 0.1)
                    block.operator.log_std.add_(torch.randn_like(block.operator.log_std) * 0.1)
                expected = model(batch()["input_ids"], threshold=0.65).logits
            trained = Path(directory) / "compensation.safetensors"
            save_compensation(model, trained)
            loaded = load_compensation(native, trained)
            for source, restored in zip(runtime.blocks, loaded.blocks):
                for name, value in source.named_parameters():
                    torch.testing.assert_close(value, dict(restored.named_parameters())[name], rtol=0, atol=0)
            with torch.no_grad():
                actual = native(batch()["input_ids"], threshold=0.65).logits
            torch.testing.assert_close(actual, expected)
            self.assertEqual(loaded.threshold, 0.5)
            with self.assertRaises(RuntimeError):
                with runtime.context(mode="teacher", threshold=0.9):
                    raise RuntimeError("test")
            self.assertEqual(runtime.mode, "student")
            self.assertEqual(runtime.threshold, 0.5)
            with self.assertRaises(ValueError):
                with runtime.context(mode="teacher", threshold=float("nan")):
                    pass
            self.assertEqual(runtime.mode, "student")

    def test_single_pass_calibration_matches_all_valid_expert_outputs(self):
        model = TinyModel().train()
        data, other = batch(), batch()
        other["input_ids"] = (other["input_ids"] + 1) % 19
        other["attention_mask"][0, -2:] = False
        batches = [data, other]
        captured = {index: [] for index in range(3)}
        handles = []
        for index, layer in enumerate(model.model.layers):
            def capture(module, args, index=index):
                captured[index].append(args[0].detach().clone())
            handles.append(layer.mlp.register_forward_pre_hook(capture))
        try:
            with mock_patch.object(model, "forward", wraps=model.forward) as forward:
                with mock_patch("skipping_qwen3_5_moe.calibrate.expert_output", wraps=expert_output) as execute:
                    result = calibrate_model(model, iter(batches))
                self.assertEqual(forward.call_count, 2)
                self.assertEqual(execute.call_count, 2 * 3 * 5)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(model.training)
        self.assertEqual(set(result), {0, 1, 2})
        for index, state in result.items():
            self.assertEqual(state["samples"], 14)
            self.assertEqual(tuple(state["mean"].shape), (5, 6))
            self.assertEqual(tuple(state["std"].shape), (5, 6))
            hidden = torch.cat([h.reshape(-1, 6)[sample["attention_mask"].reshape(-1)]
                                for h, sample in zip(captured[index], batches)])
            with torch.no_grad():
                outputs = torch.stack([expert_output(model.model.layers[index].mlp.experts, hidden, expert_id)
                                       for expert_id in range(5)], dim=1).double()
            torch.testing.assert_close(state["mean"], outputs.mean(0).float())
            torch.testing.assert_close(state["std"], outputs.std(0, unbiased=False).clamp_min(1e-5).float())
            self.assertEqual(len(model.model.layers[index].mlp._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
