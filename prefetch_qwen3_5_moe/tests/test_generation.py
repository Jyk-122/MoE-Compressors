"""使用原生 Transformers generate 和参考 Qwen MoE 验证缓存衔接。"""
import copy
from functools import wraps
from inspect import signature
import unittest
from unittest.mock import patch as mock_patch

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, StoppingCriteria
from transformers.cache_utils import DynamicCache

from ..patch import moe_blocks, patch
from .test_prefetch import MODES, reference_moe


def tiny_generation_model():
    # 随机初始化一个很小的因果模型，将其 MLP 换成参考文件中的 Qwen MoE。
    config = LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=8, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
        num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    block_class = reference_moe()
    for layer in model.model.layers:
        layer.mlp = block_class(config)
        for parameter in layer.mlp.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    return model


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.ids = torch.tensor([[1, 2, 3, 4, 5]])

    def test_native_greedy_sampling_and_route_alignment(self):
        for mode in MODES:
            for do_sample in (False, True):
                with self.subTest(mode=mode, do_sample=do_sample):
                    model = tiny_generation_model()
                    baseline = copy.deepcopy(model)
                    full = copy.deepcopy(model)
                    cache = patch(model, mode)
                    options = dict(max_new_tokens=3, do_sample=do_sample,
                                   return_dict_in_generate=True, output_scores=True)
                    torch.manual_seed(23)
                    expected = baseline.generate(self.ids, **options)
                    torch.manual_seed(23)
                    actual = model.generate(self.ids, **options)
                    torch.testing.assert_close(actual.sequences, expected.sequences)
                    for score, reference in zip(actual.scores, expected.scores):
                        torch.testing.assert_close(score, reference, rtol=0, atol=0)

                    # 独立整段 forward，检查每层、每个位置的缓存路由。
                    full_cache = patch(full, mode)
                    ids = actual.sequences[:, :-1]
                    with torch.no_grad():
                        full(ids, use_cache=False)
                    for name in ("cache_gt", "cache_pre"):
                        for route, reference in zip(getattr(cache, name), getattr(full_cache, name)):
                            if reference is None:
                                self.assertIsNone(route)
                                continue
                            self.assertEqual((route.token_start, route.token_end),
                                             (reference.token_start, reference.token_end))
                            self.assertFalse(route.router_logits.requires_grad)
                            end = route.token_end
                            torch.testing.assert_close(route.router_logits[:, :end],
                                                       reference.router_logits[:, :end])
                            torch.testing.assert_close(route.router_indices[:, :end],
                                                       reference.router_indices[:, :end])

    def test_signatures_arguments_and_return_identity(self):
        model = tiny_generation_model()
        native_generate = model.generate
        generate_signature = signature(model.generate)
        forward_signature = signature(model.forward)
        calls = []

        @wraps(native_generate)
        def record_generate(*args, **kwargs):
            result = native_generate(*args, **kwargs)
            calls.append((args, kwargs, result))
            return result

        record_generate.__signature__ = generate_signature
        model.generate = record_generate
        patch(model, (1, 0))
        config = GenerationConfig(max_new_tokens=2, pad_token_id=0)
        actual = model.generate(self.ids, config, return_dict_in_generate=True, output_scores=True)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0], self.ids)
        self.assertIs(calls[0][0][1], config)
        self.assertTrue(calls[0][1]["output_scores"])
        self.assertIs(actual, calls[0][2])
        self.assertEqual(signature(model.generate), generate_signature)
        self.assertEqual(signature(model.forward), forward_signature)
        self.assertIs(model.generate.__wrapped__, record_generate)

    def test_request_reset_and_training_after_generation(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))
        with mock_patch.object(cache, "clear", wraps=cache.clear) as clear:
            model.generate(self.ids, max_new_tokens=3)
        self.assertEqual(clear.call_count, 1)
        self.assertEqual(cache.cache_gt[0].token_end, 7)
        self.assertIsNotNone(cache.get_prefetch(7, 0))
        model.generate(self.ids[:, :1], max_new_tokens=1)
        for layer_id in range(4):
            self.assertEqual(cache.cache_gt[layer_id].token_end, 1)
            self.assertEqual(cache.cache_pre[layer_id].token_end, 2)
            self.assertIsNone(cache.get_gt(1, layer_id))

        model(self.ids, use_cache=False)
        cache.loss().backward()
        for block in moe_blocks(model):
            self.assertGreater(block.gate_prefetch.weight.grad.abs().sum().item(), 0)

    def test_left_padding_and_sampling_multiple_sequences(self):
        model = tiny_generation_model()
        baseline = copy.deepcopy(model)
        cache = patch(model, (0, 1))
        ids = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]])
        mask = ids.ne(0)
        options = dict(attention_mask=mask, max_new_tokens=3, do_sample=True,
                       num_return_sequences=2, top_k=5, temperature=0.8)
        torch.manual_seed(23)
        expected = baseline.generate(ids, **options)
        torch.manual_seed(23)
        actual = model.generate(ids, **options)
        torch.testing.assert_close(actual, expected)
        expected_mask = mask.repeat_interleave(2, dim=0)
        torch.testing.assert_close(cache.cache_gt[0].valid_mask[:, :5], expected_mask)
        torch.testing.assert_close(cache.cache_pre[0].valid_mask[:, 1:6], expected_mask)
        self.assertTrue(cache.cache_gt[0].valid_mask[:, 5:7].all())

        # 省略 mask 时，使用原生 generate 根据 pad_token_id 推断的 mask。
        model.generate(ids, max_new_tokens=1)
        torch.testing.assert_close(cache.cache_gt[0].valid_mask[:, :5], mask)

    def test_generation_config_and_length_options(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))
        default_capacity = cache.max_tokens
        config = GenerationConfig(max_length=9, max_new_tokens=2, pad_token_id=0)
        original_config = config.to_dict()
        output = model.generate(self.ids, config, max_new_tokens=4)
        self.assertEqual(output.shape[1], 9)
        self.assertEqual(cache.max_tokens, default_capacity)
        self.assertEqual(config.to_dict(), original_config)

        config.max_new_tokens = None
        output = model.generate(self.ids, generation_config=config, max_length=7)
        self.assertEqual(output.shape[1], 7)
        self.assertEqual(cache.max_tokens, default_capacity)

        model.generation_config.max_new_tokens = 2
        output = model.generate(self.ids)
        self.assertEqual(output.shape[1], 7)
        self.assertEqual(cache.max_tokens, default_capacity)

    def test_native_stopping_and_streamer(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))

        class StopAtSeven(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return input_ids.shape[1] >= 7

        class Streamer:
            def __init__(self):
                self.tokens = []
                self.finished = False

            def put(self, value):
                self.tokens.append(value.reshape(1, -1))

            def end(self):
                self.finished = True

        streamer = Streamer()
        output = model.generate(
            self.ids, max_new_tokens=8, stopping_criteria=[StopAtSeven()],
            prefix_allowed_tokens_fn=lambda batch_id, input_ids: [3], streamer=streamer,
        )
        self.assertEqual(output.shape[1], 7)
        self.assertTrue((output[:, 5:] == 3).all())
        torch.testing.assert_close(torch.cat(streamer.tokens, dim=1), output)
        self.assertTrue(streamer.finished)
        self.assertEqual(cache.cache_gt[0].token_end, 6)

    def test_native_eos_stops_generation(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))
        output = model.generate(
            self.ids, max_new_tokens=8, eos_token_id=3,
            prefix_allowed_tokens_fn=lambda batch_id, input_ids: [3],
        )
        self.assertEqual(output.shape[1], 6)
        self.assertEqual(output[0, -1].item(), 3)
        self.assertEqual(cache.cache_gt[0].token_end, 5)
        self.assertIsNone(cache.get_gt(5, 0))
        self.assertIsNotNone(cache.get_prefetch(5, 0))

    def test_new_request_after_generation_exception(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))

        class RaiseAfterForward(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                raise RuntimeError("generation callback failed")

        with self.assertRaisesRegex(RuntimeError, "generation callback failed"):
            model.generate(self.ids, max_new_tokens=2, stopping_criteria=[RaiseAfterForward()])
        self.assertEqual(cache.cache_gt[0].token_end, 5)
        model.generate(self.ids[:, :1], max_new_tokens=1)
        self.assertEqual(cache.cache_gt[0].token_end, 1)
        self.assertIsNone(cache.get_gt(1, 0))

    def test_use_cache_false_recomputes_routes(self):
        model = tiny_generation_model()
        baseline = copy.deepcopy(model)
        cache = patch(model, (0, 1))
        expected = baseline.generate(self.ids, max_new_tokens=3, use_cache=False)
        actual = model.generate(self.ids, max_new_tokens=3, use_cache=False)
        torch.testing.assert_close(actual, expected)
        self.assertEqual((cache.cache_gt[0].token_start, cache.cache_gt[0].token_end), (0, 7))
        self.assertEqual(cache.cache_pre[0].token_end, 8)
        self.assertEqual(cache.max_tokens, 8)

    def test_training_batches_prepare_routes_and_keep_gradients(self):
        model = tiny_generation_model()
        cache = patch(model, (0, 1))
        default_capacity = cache.max_tokens
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01)

        # 连续训练 batch 直接传入模型，wrapper 按本批长度和 mask 准备缓存。
        for ids in (self.ids, self.ids[:, :3]):
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[:, 0] = False
            optimizer.zero_grad(set_to_none=True)
            model(ids, attention_mask=mask, use_cache=False)
            self.assertEqual(cache.max_tokens, ids.shape[1] + 1)
            self.assertEqual(cache.cache_gt[0].token_end, ids.shape[1])
            torch.testing.assert_close(cache.cache_gt[0].valid_mask[:, :ids.shape[1]], mask)
            loss = cache.loss()
            loss.backward()
            for block in moe_blocks(model):
                self.assertGreater(block.gate_prefetch.weight.grad.abs().sum().item(), 0)
            optimizer.step()

        # 训练后的生成请求使用生成容量。
        model.generate(self.ids, max_new_tokens=2)
        self.assertEqual(cache.max_tokens, default_capacity)
        self.assertEqual(cache.cache_gt[0].token_end, 6)

    def test_generation_from_existing_kv_prefix(self):
        model = tiny_generation_model()
        expected = model.generate(self.ids, max_new_tokens=2)
        with torch.no_grad():
            past = model(self.ids[:, :3], past_key_values=DynamicCache(), use_cache=True).past_key_values
        cache = patch(model, (0, 1))
        actual = model.generate(self.ids, past_key_values=past, max_new_tokens=2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual((cache.cache_gt[0].token_start, cache.cache_gt[0].token_end), (3, 6))
        self.assertIsNone(cache.get_gt(2, 0))

    def test_inputs_embeds(self):
        model = tiny_generation_model()
        baseline = copy.deepcopy(model)
        cache = patch(model, (0, 1))
        embeds = model.get_input_embeddings()(self.ids).detach()
        expected = baseline.generate(inputs_embeds=embeds, max_new_tokens=3)
        actual = model.generate(inputs_embeds=embeds, max_new_tokens=3)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(cache.cache_gt[0].token_end, 7)

if __name__ == "__main__":
    unittest.main()
