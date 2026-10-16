"""用可手算的专家排名验证命中数量，以及测试集的加权汇总。"""
import unittest

import torch
from torch import nn

from ..router_cache import RouterCache
from ..train import evaluate


def save_routes(cache, layer_id, logits, expected, mask, prediction_start=0, teacher_start=0):
    """预测只缓存 top-2 专家；统计更大的 k' 时需要使用完整 logits。"""
    scores, indices = logits.softmax(-1).topk(2, dim=-1)
    cache.save_prefetch(prediction_start, layer_id, logits, scores, indices, mask)
    teacher_logits = torch.zeros_like(logits)
    teacher_scores = torch.ones_like(expected, dtype=logits.dtype) / expected.shape[-1]
    cache.save_gt(teacher_start, layer_id, teacher_logits, teacher_scores, expected, mask)


class MetricTests(unittest.TestCase):
    def test_requested_budgets_count_experts(self):
        cache = RouterCache(1)
        cache.clear(max_tokens=2)
        # 预测排名固定为专家 0, 1, ..., 39。
        logits = torch.arange(40, 0, -1, dtype=torch.float32).reshape(1, 1, 40).repeat(1, 2, 1)
        expected = torch.tensor([[[0, 7, 8, 11, 12, 15, 16, 31],
                                  [0, 1, 2, 3, 4, 5, 6, 7]]])
        save_routes(cache, 0, logits, expected, torch.ones(1, 2, dtype=torch.bool))
        stats = cache.prefetch_hit_stats([8, 12, 16, 32])
        self.assertEqual(stats["num_positions"], 2)
        self.assertEqual(stats["total_hits"], {8: 10, 12: 12, 16: 14, 32: 16})
        self.assertEqual(stats["avg_hits"], {8: 5.0, 12: 6.0, 16: 7.0, 32: 8.0})

    def test_masks_offsets_and_layer_weighting(self):
        cache = RouterCache(3)
        cache.clear(max_tokens=5)
        logits = torch.tensor([[[4., 3., 2., 1.]]]).repeat(2, 4, 1)
        expected = torch.tensor([[[0, 1]]]).repeat(2, 4, 1)
        mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.bool)
        # 第 1 层：预测从 token 1 开始。源、目标均有效的只有三个位置。
        save_routes(cache, 1, logits, expected, mask, prediction_start=1)
        # 第 2 层：仅一个有效位置；预测 top-2 命中 0 个，top-4 命中 2 个。
        second_logits = torch.tensor([[[1., 2., 4., 3.]]])
        save_routes(cache, 2, second_logits, torch.tensor([[[0, 1]]]),
                    torch.ones(1, 1, dtype=torch.bool), teacher_start=2, prediction_start=2)
        stats = cache.prefetch_hit_stats([1, 2, 4])
        self.assertEqual(stats["num_positions"], 4)
        self.assertEqual(stats["total_hits"], {1: 3, 2: 6, 4: 8})
        self.assertEqual(stats["avg_hits"], {1: 0.75, 2: 1.5, 4: 2.0})

    def test_empty_matches_and_budget_validation(self):
        cache = RouterCache(1)
        stats = cache.prefetch_hit_stats([1, 2])
        self.assertEqual(stats["num_positions"], 0)
        self.assertEqual(stats["avg_hits"], {1: None, 2: None})
        for values in ([], [0], [-1], [1.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cache.prefetch_hit_stats(values)

        logits = torch.tensor([[[4., 3., 2., 1.]]])
        save_routes(cache, 0, logits, torch.tensor([[[0, 1]]]),
                    torch.zeros(1, 1, dtype=torch.bool))
        self.assertEqual(cache.prefetch_hit_stats([2])["num_positions"], 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            cache.prefetch_hit_stats([5])

    def test_statistics_keep_training_graph(self):
        cache = RouterCache(1)
        logits = torch.tensor([[[4., 3., 2., 1.]]], requires_grad=True)
        save_routes(cache, 0, logits, torch.tensor([[[0, 1]]]),
                    torch.ones(1, 1, dtype=torch.bool))
        cache.prefetch_hit_stats([1, 2, 4])
        cache.loss().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_temperature_loss_and_analytic_gradient(self):
        teacher = torch.tensor([[[0.4, -0.1, 0.2, 0.0],
                                 [-0.2, 0.3, 0.1, 0.4]]], requires_grad=True)
        mask = torch.ones(1, 2, dtype=torch.bool)
        for temperature in (1.0, 0.1):
            with self.subTest(temperature=temperature):
                logits = torch.tensor([[[0.1, 0.3, -0.2, 0.0],
                                        [0.3, -0.1, 0.0, 0.2]]], requires_grad=True)
                cache = RouterCache(1)
                scores, indices = logits.softmax(-1).topk(2, dim=-1)
                cache.save_prefetch(0, 0, logits, scores, indices, mask)
                scores, indices = teacher.softmax(-1).topk(2, dim=-1)
                cache.save_gt(0, 0, teacher, scores, indices, mask)

                loss = cache.loss(temperature=temperature)
                p = (teacher.detach() / temperature).softmax(-1)
                q = (logits.detach() / temperature).softmax(-1)
                expected_loss = (p * (p.log() - q.log())).sum(-1).mean()
                torch.testing.assert_close(loss, expected_loss)
                loss.backward()
                # KL 对预测 logits 的梯度为 (q - p) / T，再按两个位置平均。
                torch.testing.assert_close(logits.grad, (q - p) / (temperature * 2))
                self.assertIsNone(teacher.grad)

    def test_evaluation_weights_samples_and_uses_no_grad(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(2, 4)
                self.router_cache = RouterCache(1)
                self.calls = []

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask, use_cache,
                        logits_to_keep, output_router_logits):
                self.calls.append((torch.is_grad_enabled(), use_cache))
                self.router_cache.clear(max_tokens=input_ids.shape[1])
                rankings = torch.tensor([[4., 3., 2., 1.], [2., 1., 4., 3.]])
                logits = rankings[input_ids]
                expected = torch.tensor([0, 1]).expand(*input_ids.shape, 2)
                save_routes(self.router_cache, 0, logits, expected, attention_mask)

        model = Model()
        weights = model.embedding.weight.detach().clone()
        # 三个命中位置和一个未命中位置，按位置加权后 top-2 平均命中 1.5 个。
        batches = []
        for tokens in ([0, 0, 0], [1]):
            ids = torch.tensor([tokens])
            batches.append(dict(input_ids=ids, labels=ids,
                                attention_mask=torch.ones_like(ids),
                                router_mask=torch.ones_like(ids, dtype=torch.bool)))
        result = evaluate(model, {"text": batches}, prefetch_k=[1, 2, 4])
        self.assertEqual(result["text"]["avg_hits"], {1: 0.75, 2: 1.5, 4: 2.0})
        self.assertEqual(result["text"]["num_positions"], 4)
        self.assertEqual(model.calls, [(False, False), (False, False)])
        torch.testing.assert_close(model.embedding.weight, weights)
        self.assertIsNone(model.embedding.weight.grad)
        self.assertIsNone(model.router_cache.cache_pre[0])
        with self.assertRaisesRegex(ValueError, "No matched valid routes"):
            evaluate(model, {"text": []}, prefetch_k=[1])

    def test_router_mask_selects_target_positions(self):
        cache = RouterCache(1)
        logits = torch.tensor([[[4., 3., 2., 1.]]]).repeat(1, 3, 1).requires_grad_()
        expected = torch.tensor([[[0, 1]]]).repeat(1, 3, 1)
        save_routes(cache, 0, logits, expected, torch.ones(1, 3, dtype=torch.bool),
                    prediction_start=1)
        # token 0 是 prompt，token 1 是首个 assistant 输入：它们构成有效训练对。
        router_mask = torch.tensor([[False, True, False]])
        stats = cache.prefetch_hit_stats([1, 2], router_mask=router_mask)
        self.assertEqual(stats["num_positions"], 1)
        self.assertEqual(stats["avg_hits"], {1: 1.0, 2: 2.0})
        loss = cache.loss(router_mask=router_mask)
        expected_loss = (-torch.log(torch.tensor(4.)) - logits[0, 0].log_softmax(-1)).mean()
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        self.assertGreater(logits.grad[:, :1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 1:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
