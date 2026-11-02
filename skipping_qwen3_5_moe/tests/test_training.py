import copy
import unittest

import torch
from torch.nn import functional as F

from ..losses import CompensationTrainingModel, hidden_kl, prediction_mask
from ..model import configure_activation_checkpointing, patch
from .helpers import TinyModel, calibration_for, batch


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_lm_kl_and_gradients_match_distribution_reference(self):
        head = torch.nn.Linear(6, 19, bias=False).requires_grad_(False)
        teacher = torch.randn(2, 5, 6, requires_grad=True)
        student = torch.randn(2, 5, 6, requires_grad=True)
        mask = prediction_mask(batch())
        self.assertEqual(mask.tolist(), [[False, False, True, True], [False, False, True, True]])
        actual = hidden_kl(student, teacher, head, mask, temperature=1.7)
        log_s = (head(student[:, :-1][mask]) / 1.7).log_softmax(-1)
        log_t = (head(teacher.detach()[:, :-1][mask]) / 1.7).log_softmax(-1)
        expected = (log_t.exp() * (log_t - log_s)).sum(-1).mean() * 1.7 ** 2
        torch.testing.assert_close(actual, expected)
        ga, teacher_grad = torch.autograd.grad(actual, (student, teacher), retain_graph=True, allow_unused=True)
        self.assertIsNone(teacher_grad)
        gb = torch.autograd.grad(expected, student)[0]
        torch.testing.assert_close(ga, gb)

    def test_each_loss_trains_coefficients_and_moments_through_frozen_backbone(self):
        for feature_weight, kl_weight in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(feature=feature_weight, kl=kl_weight):
                model = TinyModel()
                runtime = patch(model, calibration_for(model))
                frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                          if not value.requires_grad}
                task = CompensationTrainingModel(model, feature_weight=feature_weight,
                                                 kl_weight=kl_weight)
                parameters = [p for p in model.parameters() if p.requires_grad]
                self.assertEqual(len(parameters), 3 * len(runtime.blocks))
                before = [p.detach().clone() for p in parameters]
                result = task(batch())
                result["loss"].backward()
                for block in runtime.blocks:
                    self.assertEqual({name for name, p in block.named_parameters() if p.requires_grad},
                                     {"coefficients.logits", "operator.mean", "operator.log_std"})
                    for parameter in (block.coefficients.logits, block.operator.mean, block.operator.log_std):
                        gradient = parameter.grad
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0)
                optimizer = torch.optim.SGD(parameters, lr=0.1)
                optimizer.step()
                self.assertTrue(all(not torch.equal(a, b) for a, b in zip(before, parameters)))
                for name, value in model.named_parameters():
                    if name in frozen:
                        self.assertIsNone(value.grad)
                        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
                self.assertEqual(result["counts"].tolist(), [48, 96, 24, 24])
                self.assertEqual(runtime.mode, "student")

    def test_local_targets_use_student_layer_inputs(self):
        model = TinyModel()
        runtime = patch(model, calibration_for(model))
        captured = {}
        handles = []
        for block in runtime.blocks:
            def capture(module, args, index=block.layer_index):
                captured[index] = args[0].detach().clone()
            handles.append(block.register_forward_pre_hook(capture))
        result = CompensationTrainingModel(model, kl_weight=0)(batch())
        for handle in handles:
            handle.remove()
        valid = batch()["attention_mask"].reshape(-1)
        terms = []
        from ..experts import selected_outputs
        for block in runtime.blocks:
            h = captured[block.layer_index]
            with torch.no_grad():
                flat = h.reshape(-1, 6)
                _, q, ids = block.original.gate(flat)
                y = selected_outputs(block.original.experts, flat, ids, torch.ones_like(ids, dtype=torch.bool))
                target = (q.unsqueeze(-1) * y).sum(1)
                shared = block.original.shared_expert(flat) * block.original.shared_expert_gate(flat).sigmoid()
                with runtime.context(mode="student", reset=True):
                    prediction = block(h).reshape(-1, 6) - shared
                terms.append(F.mse_loss(prediction[valid], target[valid]))
        torch.testing.assert_close(result["feature_loss"], torch.stack(terms).mean())

    def test_full_retention_and_single_source_compensation_gradients(self):
        for threshold in (1.0, 0.01):
            model = TinyModel()
            runtime = patch(model, calibration_for(model))
            result = CompensationTrainingModel(model, threshold=threshold)(batch())
            result["loss"].backward()
            self.assertTrue(torch.isfinite(result["loss"]))
            for block in runtime.blocks:
                torch.testing.assert_close(block.coefficients.logits.grad,
                                           torch.zeros_like(block.coefficients.logits))
                for parameter in block.operator.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    if threshold == 1:
                        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
                    else:
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
            if threshold == 1:
                self.assertLess(result["feature_loss"].item(), 1e-10)
                self.assertLess(result["kl_loss"].abs().item(), 1e-6)

    def test_activation_checkpointing_preserves_losses_gradients_and_counts(self):
        template = TinyModel()
        calibration = calibration_for(template)
        results = []
        for enabled in (False, True):
            model = copy.deepcopy(template)
            runtime = patch(model, calibration)
            configure_activation_checkpointing(model, enabled)
            parameter_storage = {p.storage().data_ptr() for p in model.parameters()}
            saved = {}

            def pack(tensor):
                storage = tensor.storage()
                if storage.data_ptr() not in parameter_storage:
                    saved[storage.data_ptr()] = storage.size() * tensor.element_size()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                result = CompensationTrainingModel(model, threshold=0.65)(batch())
            counts = {index: value.clone() for index, value in runtime.statistics.items()}
            self.assertEqual(runtime.feature_terms, {})
            runtime.threshold = 0.9
            result["loss"].backward()
            self.assertEqual(runtime.feature_terms, {})
            for index, value in counts.items():
                torch.testing.assert_close(runtime.statistics[index], value)
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
            results.append((result, gradients, sum(saved.values())))
        for name in ("loss", "feature_loss", "kl_loss", "counts"):
            torch.testing.assert_close(results[0][0][name], results[1][0][name])
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name])
        self.assertLess(results[1][2], results[0][2])

    def test_attention_checkpointing_preserves_kwargs_and_tuple_outputs(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, *, hidden_states, position_embeddings):
                self.calls += 1
                cosine, sine = position_embeddings
                return (hidden_states * cosine + sine).sin().square(), None

        model = TinyModel()
        runtime = patch(model, calibration_for(model))
        attention = Attention()
        model.model.layers[0].self_attn = attention
        hidden = torch.randn(2, 5, 6, requires_grad=True)
        positions = (torch.randn_like(hidden), torch.randn_like(hidden))
        expected = attention(hidden_states=hidden, position_embeddings=positions)[0]
        expected_grad = torch.autograd.grad(expected.sum(), hidden)[0]
        attention.calls = 0
        configure_activation_checkpointing(model)
        with runtime.context(mode="reconstruction"):
            actual, weights = attention(hidden_states=hidden, position_embeddings=positions)
        actual_grad = torch.autograd.grad(actual.sum(), hidden)[0]
        self.assertIsNone(weights)
        self.assertEqual(attention.calls, 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
