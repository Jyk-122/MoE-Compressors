"""核对两种融合公式、上一 token 对齐和共同统计范围。"""
import unittest

import torch

from ..router_cache import RouterCache


def fusion_cache():
    cache = RouterCache(2)
    cache.clear(max_tokens=4)
    layer_logits = torch.tensor([[[0.85, 0.01, 0.14]]]).log().requires_grad_()
    teacher_logits = torch.tensor([[[0.02, 0.75, 0.23], [0.7, 0.2, 0.1]]]).log()
    scores, indices = layer_logits.softmax(-1).topk(1, dim=-1)
    cache.save_prefetch(1, 1, layer_logits, scores, indices, torch.ones(1, 1, dtype=torch.bool))
    scores, indices = teacher_logits.softmax(-1).topk(1, dim=-1)
    cache.save_gt(0, 1, teacher_logits, scores, indices, torch.ones(1, 2, dtype=torch.bool))
    return cache, layer_logits


class FusionTests(unittest.TestCase):
    def test_formulas_and_endpoints(self):
        cache, _ = fusion_cache()
        for mode in ("logits", "prob"):
            with self.subTest(mode=mode):
                # 上一 token 的 top-1 是专家 1，当前真值是专家 0。
                previous = cache.prefetch_hit_stats([1, 2, 3], mode=mode, lambda_=0)
                self.assertEqual(previous["num_positions"], 1)
                self.assertEqual(previous["avg_hits"], {1: 0.0, 2: 0.0, 3: 1.0})
                prediction = cache.prefetch_hit_stats([1, 2, 3], mode=mode, lambda_=1)
                self.assertEqual(prediction, cache.prefetch_hit_stats([1, 2, 3]))

        # 算术平均为 [0.435, 0.38, 0.185]，选择专家 0。
        probability = cache.prefetch_hit_stats([1, 2, 3], mode="prob", lambda_=0.5)
        self.assertEqual(probability["avg_hits"], {1: 1.0, 2: 1.0, 3: 1.0})
        # 几何平均的排序为 2, 0, 1。
        logits = cache.prefetch_hit_stats([1, 2, 3], mode="logits", lambda_=0.5)
        self.assertEqual(logits["avg_hits"], {1: 0.0, 2: 1.0, 3: 1.0})

    def test_temperature_changes_probability_fusion(self):
        cache, _ = fusion_cache()
        warm = cache.prefetch_hit_stats([1], mode="prob", lambda_=0.47, temperature=1.0)
        cold = cache.prefetch_hit_stats([1], mode="prob", lambda_=0.47, temperature=0.1)
        self.assertEqual(warm["avg_hits"][1], 1)
        self.assertEqual(cold["avg_hits"][1], 0)
        self.assertEqual(
            cache.prefetch_hit_stats([1, 2], mode="logits", lambda_=0.47, temperature=0.1),
            cache.prefetch_hit_stats([1, 2], mode="logits", lambda_=0.47, temperature=1.0),
        )

    def test_log_space_preserves_small_probability_ranking(self):
        cache = RouterCache(1)
        cache.clear(max_tokens=2)
        layer_logits = torch.tensor([[[0., -1100., -1000.]]])
        teacher_logits = torch.tensor([[[0., -1100., -1000.], [-10., -20., 0.]]])
        scores, indices = layer_logits.softmax(-1).topk(1, dim=-1)
        cache.save_prefetch(1, 0, layer_logits, scores, indices, torch.ones(1, 1, dtype=torch.bool))
        scores, indices = teacher_logits.softmax(-1).topk(1, dim=-1)
        cache.save_gt(0, 0, teacher_logits, scores, indices, torch.ones(1, 2, dtype=torch.bool))
        result = cache.prefetch_hit_stats([1, 2], mode="prob", lambda_=0.5)
        self.assertEqual(result["avg_hits"], {1: 0.0, 2: 1.0})

    def test_common_positions_with_padding_and_cache_offsets(self):
        cache = RouterCache(2)
        cache.clear(max_tokens=7)
        teacher_logits = torch.tensor([[[4., 3., 2., 1.]]]).repeat(2, 4, 1)
        teacher_mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.bool)
        scores, indices = teacher_logits.softmax(-1).topk(2, dim=-1)
        for layer_id in (0, 1):
            cache.save_gt(2, layer_id, teacher_logits, scores, indices, teacher_mask)
        layer_logits = torch.tensor([[[1., 2., 3., 4.]]]).repeat(2, 6, 1)
        prediction_mask = torch.ones(2, 6, dtype=torch.bool)
        prediction_mask[1, 3] = False  # 第二个样本在目标 token 4 缺少有效预测。
        scores, indices = layer_logits.softmax(-1).topk(2, dim=-1)
        cache.save_prefetch(1, 1, layer_logits, scores, indices, prediction_mask)
        router_mask = torch.zeros(2, 7, dtype=torch.bool)
        router_mask[:, 3:] = True  # 历史来源可位于 prompt；筛选的是目标位置。
        for mode in ("logits", "prob"):
            for lambda_ in (0., 0.4, 1.):
                with self.subTest(mode=mode, lambda_=lambda_):
                    stats = cache.prefetch_hit_stats([1, 2, 4], router_mask=router_mask,
                                                     mode=mode, lambda_=lambda_)
                    self.assertEqual(stats["num_positions"], 2)
                    self.assertEqual(stats["total_hits"][4], 4)
                    self.assertEqual(stats["avg_hits"][4], 2)
        # 第一个真实位置没有 t-1 路由，所有 lambda 都采用相同的边界。
        router_mask[:, :] = False
        router_mask[:, 2] = True
        for lambda_ in (0., 0.5, 1.):
            stats = cache.prefetch_hit_stats([1], router_mask=router_mask, lambda_=lambda_)
            self.assertEqual(stats["num_positions"], 0)
            self.assertIsNone(stats["avg_hits"][1])

    def test_statistics_preserve_routes_and_gradients(self):
        cache, logits = fusion_cache()
        prediction = cache.cache_pre[1].router_logits.detach().clone()
        teacher = cache.cache_gt[1].router_logits.clone()
        before = cache.loss().detach().clone()
        for mode in ("logits", "prob"):
            cache.prefetch_hit_stats([1, 2], mode=mode, lambda_=0.5)
        torch.testing.assert_close(cache.cache_pre[1].router_logits, prediction)
        torch.testing.assert_close(cache.cache_gt[1].router_logits, teacher)
        torch.testing.assert_close(cache.loss(), before)
        cache.loss().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_invalid_options(self):
        cache = RouterCache(1)
        with self.assertRaisesRegex(ValueError, "mode"):
            cache.prefetch_hit_stats([1], mode="unknown")
        for lambda_ in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(lambda_=lambda_), self.assertRaisesRegex(ValueError, "lambda"):
                cache.prefetch_hit_stats([1], lambda_=lambda_)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(ValueError, "temperature"):
                cache.prefetch_hit_stats([1], mode="prob", lambda_=0.5, temperature=temperature)


if __name__ == "__main__":
    unittest.main()
