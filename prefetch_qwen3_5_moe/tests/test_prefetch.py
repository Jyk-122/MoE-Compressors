"""CPU 小模型验证：直接执行 modeling.py 中的四个 MoE 类。"""
import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

import torch
from torch import nn
import torch.nn.functional as F

from ..patch import load_prefetch, moe_blocks, patch, save_prefetch
from ..router_cache import RouterCache


def reference_moe():
    path = Path(__file__).resolve().parent.parent / "modeling.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"Qwen3_5MoeMLP", "Qwen3_5MoeExperts", "Qwen3_5MoeTopKRouter", "Qwen3_5MoeSparseMoeBlock"}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = dict(torch=torch, nn=nn, F=F, Qwen3_5MoeConfig=object,
                     ACT2FN={"silu": F.silu}, use_experts_implementation=lambda cls: cls,
                     use_kernel_forward_from_hub=lambda name: lambda cls: cls)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Qwen3_5MoeSparseMoeBlock"]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = SimpleNamespace(hidden_size=6, num_experts=4, num_experts_per_tok=2,
                              moe_intermediate_size=8, shared_expert_intermediate_size=8, hidden_act="silu")
        block_class = reference_moe()
        self.embed = nn.Embedding(16, 6)
        self.layers = nn.ModuleList()
        for _ in range(4):
            layer = nn.Module()
            layer.mlp = block_class(cfg)
            self.layers.append(layer)
        self.lm_head = nn.Linear(6, 16, bias=False)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.2)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        # 累加前缀模拟因果上下文，验证整段 forward 和带历史 decode 的位置对齐。
        x = self.embed(input_ids).cumsum(dim=1)
        if past_key_values is not None:
            x = x + past_key_values
        past = x[:, -1:].detach()
        for layer in self.layers:
            x = x + layer.mlp(x)
        return SimpleNamespace(logits=self.lm_head(x), past_key_values=past)


MODES = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])

    def test_target_positions_and_native_output(self):
        for dl, dt in MODES:
            with self.subTest(mode=(dl, dt)):
                model = TinyModel().eval()
                expected = model(self.ids).logits.detach()
                cache = patch(model, (dl, dt))
                cache.clear(max_tokens=8)
                cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
                actual = model(self.ids).logits
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(cache.cache_gt), 4)
                self.assertEqual(len(cache.cache_pre), 4)
                for l, block in enumerate(model.layers):
                    teacher = cache.cache_gt[l]
                    self.assertEqual((teacher.token_start, teacher.token_end), (0, 5))
                    prediction = cache.cache_pre[l]
                    if l < dl:
                        self.assertIsNone(prediction)
                        self.assertIsNone(block.mlp.prefetch_indices)
                    else:
                        self.assertEqual((prediction.token_start, prediction.token_end), (dt, 5 + dt))
                        self.assertEqual(prediction.router_logits.shape, (2, 8, 4))
                        torch.testing.assert_close(block.mlp.prefetch_indices, prediction.router_indices[:, :5])
                        torch.testing.assert_close(block.mlp.prefetch_valid_mask, prediction.valid_mask[:, :5])
                        self.assertFalse(prediction.valid_mask[:, :dt].any())
                        self.assertTrue((prediction.router_indices[:, :dt] == -1).all())
                        for t in range(dt):
                            self.assertIsNone(cache.get_prefetch(t, l))
                        torch.testing.assert_close(
                            cache.get_prefetch(dt, l).router_logits, prediction.router_logits[:, dt])
                    for t in range(5):
                        route = cache.get_gt(t, l)
                        self.assertFalse(route.router_logits.requires_grad)
                        torch.testing.assert_close(route.router_indices, route.router_logits.topk(2, dim=-1).indices)

    def test_loss_alignment_masks_and_gradients(self):
        mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        for dl, dt in MODES:
            with self.subTest(mode=(dl, dt)):
                model = TinyModel()
                original = {name: p.detach().clone() for name, p in model.named_parameters()}
                cache = patch(model, (dl, dt))
                cache.clear(max_tokens=8)
                source_hidden, native_logits = {}, {}
                handles = []
                for l, block in enumerate(moe_blocks(model)):
                    def capture(module, inputs, outputs, l=l):
                        source_hidden[l] = inputs[0].reshape(2, 5, -1).detach()
                        native_logits[l] = outputs[0].reshape(2, 5, -1).detach()
                    handles.append(block.gate.register_forward_hook(capture))
                cache.token_start = 0
                cache.attention_mask = mask
                model(self.ids)
                for handle in handles:
                    handle.remove()
                loss = cache.loss()
                loss.backward()
                gradients = [block.gate_prefetch.weight.grad.clone() for block in moe_blocks(model)
                             if block.gate_prefetch is not None]

                # 独立参考：直接按偏移切片，不通过 RouterCache 获取训练对。
                terms = []
                for l, block in enumerate(moe_blocks(model)[:4 - dl]):
                    block.gate_prefetch.weight.grad = None
                    pred = F.linear(source_hidden[l][:, :5 - dt], block.gate_prefetch.weight)
                    gt = native_logits[l + dl][:, dt:]
                    valid = mask[:, :5 - dt] & mask[:, dt:]
                    log_p, log_q = gt[valid].log_softmax(-1), pred[valid].log_softmax(-1)
                    terms.append((log_p.exp() * (log_p - log_q)).sum(-1))
                expected = torch.cat(terms).mean()
                torch.testing.assert_close(loss, expected)
                expected.backward()
                for block, gradient in zip(moe_blocks(model), gradients):
                    torch.testing.assert_close(block.gate_prefetch.weight.grad, gradient)
                    self.assertGreater(gradient.abs().sum().item(), 0)
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
                optimizer.step()
                for name, p in model.named_parameters():
                    if name in original:
                        self.assertIsNone(p.grad)
                        torch.testing.assert_close(p, original[name], rtol=0, atol=0)

    def test_prefetch_uses_independent_target_router(self):
        model = TinyModel()
        cache = patch(model, (1, 0))
        predictor = model.layers[0].mlp.gate_prefetch
        target_gate = model.layers[1].mlp.gate
        self.assertIs(type(predictor), type(target_gate))
        self.assertEqual(type(predictor).__name__, "Qwen3_5MoeTopKRouter")
        self.assertNotEqual(predictor.weight.data_ptr(), target_gate.weight.data_ptr())
        self.assertTrue(predictor.weight.requires_grad)
        self.assertFalse(target_gate.weight.requires_grad)
        self.assertEqual(predictor.weight.dtype, torch.float32)
        torch.testing.assert_close(predictor.weight, target_gate.weight)

        hidden = torch.randn(2, 5, 6)
        expected_routes = target_gate(hidden)
        actual_routes = predictor(hidden)
        for actual, expected in zip(actual_routes, expected_routes):
            torch.testing.assert_close(actual, expected)

        captured = []
        def capture(module, inputs, outputs):
            captured.append(outputs)
        handle = predictor.register_forward_hook(capture)
        cache.clear(max_tokens=5)
        cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
        model(self.ids)
        handle.remove()
        logits, scores, indices = captured[0]
        prediction = cache.cache_pre[1]
        torch.testing.assert_close(prediction.router_logits, logits.reshape(2, 5, 4))
        torch.testing.assert_close(prediction.router_scores, scores.reshape(2, 5, 2))
        torch.testing.assert_close(prediction.router_indices, indices.reshape(2, 5, 2))

        original_weight = target_gate.weight.detach().clone()
        with torch.no_grad():
            predictor.weight.add_(0.5)
        torch.testing.assert_close(target_gate.weight, original_weight, rtol=0, atol=0)

    def test_prefill_decode_matches_full_sequence(self):
        for mode in MODES:
            with self.subTest(mode=mode), torch.no_grad():
                full = TinyModel()
                stream = copy.deepcopy(full)
                full_cache = patch(full, mode)
                cache = patch(stream, mode)
                full_cache.clear(max_tokens=8)
                cache.clear(max_tokens=8)
                full_cache.attention_mask = torch.ones_like(self.ids[:1], dtype=torch.bool)
                expected = full(self.ids[:1]).logits
                past, outputs = None, []
                for start, end in [(0, 3), (3, 4), (4, 5)]:
                    previous = cache.get_prefetch(start, mode[0])
                    cache.token_start = start
                    cache.attention_mask = torch.ones_like(self.ids[:1, start:end], dtype=torch.bool)
                    output = stream(self.ids[:1, start:end], past_key_values=past)
                    past = output.past_key_values
                    outputs.append(output.logits)
                    if start > 0 and mode[1] > 0:
                        self.assertIsNotNone(previous)
                        current = cache.get_prefetch(start, mode[0])
                        self.assertEqual(current.router_logits.data_ptr(), previous.router_logits.data_ptr())
                        torch.testing.assert_close(current.router_logits, previous.router_logits)
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
                for layer_id in range(4):
                    for name in ("cache_gt", "cache_pre"):
                        route = getattr(cache, name)[layer_id]
                        reference = getattr(full_cache, name)[layer_id]
                        if reference is None:
                            self.assertIsNone(route)
                            continue
                        self.assertEqual((route.token_start, route.token_end),
                                         (reference.token_start, reference.token_end))
                        torch.testing.assert_close(route.router_logits, reference.router_logits)
                        torch.testing.assert_close(route.router_indices, reference.router_indices)

    def test_checkpoint_roundtrip(self):
        model = TinyModel()
        restored = copy.deepcopy(model)
        cache = patch(model, (2, 1))
        cache.clear(max_tokens=8)
        with torch.no_grad():
            moe_blocks(model)[0].gate_prefetch.weight.add_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefetch.safetensors"
            save_prefetch(model, path)
            restored_cache = load_prefetch(restored, path)
            restored_cache.clear(max_tokens=8)
        self.assertEqual(restored.mode_prefetch, (2, 1))
        cache.token_start = 0
        restored_cache.token_start = 0
        cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
        restored_cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
        model(self.ids)
        restored(self.ids)
        for layer_id, prediction in enumerate(cache.cache_pre):
            if prediction is not None:
                torch.testing.assert_close(prediction.router_logits, restored_cache.cache_pre[layer_id].router_logits)

    def test_training_steps_reduce_loss(self):
        model = TinyModel()
        cache = patch(model, (0, 1))
        cache.clear(max_tokens=8)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
        losses = []
        for _ in range(5):
            cache.clear(max_tokens=8)
            cache.token_start = 0
            cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            model(self.ids)
            loss = cache.loss()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])

    def test_prediction_exists_before_source_experts_run(self):
        model = TinyModel()
        cache = patch(model, (2, 0))
        cache.clear(max_tokens=8)
        checked = []
        def check(module, inputs):
            prediction = cache.get_prefetch(0, 2)
            self.assertIsNotNone(prediction)
            self.assertIsNone(cache.get_gt(0, 2))
            checked.append(True)
        handle = model.layers[0].mlp.experts.register_forward_pre_hook(check)
        cache.token_start = 0
        cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
        model(self.ids)
        handle.remove()
        self.assertEqual(checked, [True])

    def test_text_and_conditional_model_paths(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                text_model = TinyModel()
                outer = nn.Module()
                if conditional:
                    outer.model = nn.Module()
                    outer.model.language_model = text_model
                else:
                    outer.model = text_model
                cache = patch(outer, (1, 0))
                cache.clear(max_tokens=8)
                cache.token_start = 0
                cache.attention_mask = torch.ones_like(self.ids, dtype=torch.bool)
                text_model(self.ids)
                self.assertIs(cache, text_model.layers[0].mlp.router_cache)
                self.assertEqual(len(cache.cache_gt), 4)
                self.assertEqual(cache.cache_gt[0].token_end, 5)

    def test_empty_matches_and_mode_validation(self):
        for mode in [(0, 0), (-1, 0), (4, 0), (0.5, 1), (1,)]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                patch(TinyModel(), mode)
        model = TinyModel()
        cache = patch(model, (0, 1))
        cache.clear(max_tokens=8)
        cache.token_start = 0
        cache.attention_mask = torch.ones_like(self.ids[:, :1], dtype=torch.bool)
        model(self.ids[:, :1])
        with self.assertRaisesRegex(ValueError, "No matched valid routes"):
            cache.loss()
        cache.clear(max_tokens=8)
        cache.token_start = 0
        cache.attention_mask = torch.zeros_like(self.ids, dtype=torch.bool)
        model(self.ids)
        with self.assertRaisesRegex(ValueError, "No matched valid routes"):
            cache.loss()

    def test_ground_truth_is_detached_at_storage(self):
        cache = RouterCache(2)
        cache.clear(max_tokens=3)
        gt = torch.randn(2, 3, 4, requires_grad=True)
        pred = torch.randn(2, 3, 4, requires_grad=True)
        scores, indices = gt.detach().softmax(-1).topk(2, dim=-1)
        valid_mask = torch.ones(2, 3, dtype=torch.bool)
        cache.save_gt(0, 1, gt, scores, indices, valid_mask)
        cache.save_prefetch(0, 1, pred, scores, indices, valid_mask)
        cache.loss().backward()
        self.assertIsNone(gt.grad)
        self.assertIsNotNone(pred.grad)

    def test_appended_training_tensors_preserve_gradients(self):
        cache = RouterCache(1)
        cache.clear(max_tokens=6)
        prediction = torch.randn(2, 5, 4, requires_grad=True)
        teacher = torch.randn(2, 5, 4)
        scores, indices = teacher.softmax(-1).topk(2, dim=-1)
        mask = torch.ones(2, 5, dtype=torch.bool)
        for start, end in [(0, 3), (3, 4), (4, 5)]:
            cache.save_prefetch(start + 1, 0, prediction[:, start:end],
                                scores[:, start:end], indices[:, start:end], mask[:, start:end])
            cache.save_gt(start, 0, teacher[:, start:end],
                          scores[:, start:end], indices[:, start:end], mask[:, start:end])
        loss = cache.loss()
        loss.backward()
        actual_gradient = prediction.grad.clone()
        prediction.grad = None
        log_p = teacher[:, 1:].log_softmax(-1)
        log_q = prediction[:, :-1].log_softmax(-1)
        expected = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
        expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(actual_gradient, prediction.grad)

    def test_one_bulk_write_per_layer_and_forward(self):
        model = TinyModel()
        cache = patch(model, (1, 0))
        for sequence_length in (1, 257):
            cache.clear(max_tokens=sequence_length)
            ids = torch.randint(0, 16, (1, sequence_length))
            cache.attention_mask = torch.ones_like(ids, dtype=torch.bool)
            with mock_patch.object(cache, "save_prefetch", wraps=cache.save_prefetch) as save_prediction:
                with mock_patch.object(cache, "save_gt", wraps=cache.save_gt) as save_teacher:
                    model(ids)
            self.assertEqual(save_prediction.call_count, 3)
            self.assertEqual(save_teacher.call_count, 4)
            self.assertEqual(save_prediction.call_args.kwargs["router_logits"].shape,
                             (1, sequence_length, 4))

    def test_capacity_and_continuous_writes(self):
        cache = RouterCache(1)
        cache.clear(max_tokens=3)
        logits = torch.randn(1, 2, 4)
        scores, indices = logits.softmax(-1).topk(2, dim=-1)
        mask = torch.ones(1, 2, dtype=torch.bool)
        cache.save_prefetch(0, 0, logits, scores, indices, mask)
        with self.assertRaisesRegex(ValueError, "capacity"):
            cache.save_prefetch(2, 0, logits, scores, indices, mask)
        with self.assertRaisesRegex(ValueError, "previous token_end"):
            cache.save_prefetch(0, 0, logits, scores, indices, mask)


if __name__ == "__main__":
    unittest.main()
