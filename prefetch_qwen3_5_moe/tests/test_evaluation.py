"""验证 checkpoint 加载、原生生成和 decode 路由的统计边界。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch as mock_patch

import torch
import transformers
import yaml

from prefetch.datasets.dataset import OverlengthSample
from ..patch import moe_blocks, patch, save_prefetch
from ..test import evaluate_generation, generation_inputs, main
from ..train import forward_batch, make_loaders
from .test_training import Processor, make_config, training_model


def generation_model():
    model = training_model()
    prepare_inputs = model.prepare_inputs_for_generation

    # 测试模型接收图像字段；token 生成由 Llama 的原生 generate 执行。
    def prepare(input_ids, pixel_values=None, image_grid_thw=None, **kwargs):
        inputs = prepare_inputs(input_ids, **kwargs)
        inputs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return inputs

    model.prepare_inputs_for_generation = prepare
    return model


class EvaluationTests(unittest.TestCase):
    def test_checkpoint_generation_matches_routes_on_generated_sequence(self):
        for split, mode in (("test", (1, 0)), ("validation", (0, 1))):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = make_config(directory)
                train_loader, _ = make_loaders(Processor(), config["data"])
                model = training_model()
                base_weights = copy.deepcopy(model.state_dict())
                patch(model, mode)
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
                forward_batch(model, next(iter(train_loader)),
                              temperature=config["prefetch"]["temperature"]).backward()
                optimizer.step()
                checkpoint = directory / "gate_prefetch.safetensors"
                save_prefetch(model, checkpoint)

                config.pop("training")
                config["data"].pop("train")
                config["prefetch"].pop("mode_prefetch")
                if split == "test":
                    config["data"]["test"] = config["data"].pop("validation")
                    config["data"]["validation"] = {"unused": {"path": "unused-validation.jsonl"}}
                config_path = directory / "test.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

                restored = generation_model()
                restored.load_state_dict(base_weights)
                generated = []
                native_generate = restored.generate

                def record_generate(**kwargs):
                    output = native_generate(**kwargs)
                    generated.append((kwargs, output))
                    return output

                restored.generate = record_generate
                model_loader = Mock(return_value=restored)
                processor_loader = Mock(return_value=Processor())
                replacements = {
                    "AutoProcessor": SimpleNamespace(from_pretrained=processor_loader),
                    "Qwen3_5MoeForConditionalGeneration": SimpleNamespace(from_pretrained=model_loader),
                }
                argv = ["test", "--config", str(config_path), "--checkpoint", str(checkpoint),
                        "--device", "cpu", "--max-new-tokens", "4"]
                output = Path(config["output_dir"]) / "metrics" / "test.json"
                if split == "test":
                    output = directory / "reports" / "checkpoint-test.json"
                    argv.extend(["--output", str(output)])
                with mock_patch.dict(vars(transformers), replacements), mock_patch.object(sys, "argv", argv):
                    main()

                actual = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(actual["evaluation"], "first_answer_generation")
                self.assertEqual(actual["scope"], "decode")
                self.assertEqual(actual["mode_prefetch"], list(mode))
                self.assertEqual(actual["max_new_tokens"], 4)
                self.assertEqual(restored.mode_prefetch, mode)
                self.assertEqual(len(generated), 2)  # 每个数据集 limit=1。
                for name, (inputs, ids) in zip(sorted(config["data"][split]), generated):
                    prompt = [1, 6, 3, 2]  # user a, assistant 前缀。
                    if name == "vl":
                        prompt = [1, 4, 4, 4, 6, 3, 2]
                        self.assertIn("pixel_values", inputs)
                    self.assertEqual(inputs["input_ids"].tolist(), [prompt])
                    self.assertTrue(inputs["use_cache"])
                    self.assertNotIn("router_cache_max_tokens", inputs)  # wrapper 消费此参数。
                    prompt_length = len(prompt)
                    # 将实际生成的前缀整段重放，独立检查 decode 路由。
                    with torch.no_grad():
                        model(ids[:, :-1], use_cache=False)
                        mask = torch.zeros_like(ids[:, :-1], dtype=torch.bool)
                        mask[:, prompt_length:] = True
                        expected = model.router_cache.prefetch_hit_stats([1, 2, 4], router_mask=mask)
                        expected_loss = model.router_cache.loss(
                            router_mask=mask, temperature=config["prefetch"]["temperature"])
                    report = actual["datasets"][name]
                    self.assertEqual(report["num_examples"], 1)
                    self.assertEqual(report["generated_tokens"], 4)
                    self.assertEqual(report["decode_tokens"], 3)
                    self.assertEqual(report["num_positions"], 3 * (4 - mode[0]))
                    self.assertEqual(report["avg_hits"], json.loads(json.dumps(expected["avg_hits"])))
                    self.assertAlmostEqual(report["loss"], expected_loss.item(), places=5)
                self.assertEqual(len(restored.forward_calls), 8)
                self.assertTrue(any(call["has_image"] for call in restored.forward_calls))
                self.assertTrue(all(not call["grad_enabled"] for call in restored.forward_calls))
                self.assertTrue(all(call["use_cache"] for call in restored.forward_calls))
                for trained_block, restored_block in zip(moe_blocks(model), moe_blocks(restored)):
                    if trained_block.gate_prefetch is not None:
                        torch.testing.assert_close(restored_block.gate_prefetch.weight,
                                                   trained_block.gate_prefetch.weight, rtol=0, atol=0)
                processor_loader.assert_called_once_with("test-model")
                model_loader.assert_called_once_with("test-model", dtype=torch.float32,
                                                     device_map={"": torch.device("cpu")})

    def test_first_answer_prompt_and_prompt_only_records(self):
        example = {"id": "text", "images": [], "messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "bc"},
            {"role": "user", "content": "d"}, {"role": "assistant", "content": "ef"},
        ]}
        expected = generation_inputs(Processor(), example)
        self.assertEqual(expected["input_ids"].tolist(), [[1, 6, 3, 2]])
        example["messages"] = example["messages"][:1]
        torch.testing.assert_close(generation_inputs(Processor(), example)["input_ids"], expected["input_ids"])

    def test_native_eos_and_weighting_by_decode_positions(self):
        model = generation_model()
        cache = patch(model, (0, 1))
        records = [
            {"id": text, "images": [], "messages": [{"role": "user", "content": text}]}
            for text in ("a", "ab")
        ]
        generate = model.generate
        sample_stats = []

        def generate_until_eos(**kwargs):
            prompt_length = kwargs["input_ids"].shape[1]
            stop_length = 5
            if prompt_length == 5:
                stop_length = 8

            def allowed_tokens(batch_id, input_ids):
                if len(input_ids) >= stop_length:
                    return [3]
                return [9]

            return generate(**kwargs, eos_token_id=3, prefix_allowed_tokens_fn=allowed_tokens)

        hit_stats = cache.prefetch_hit_stats

        def record_stats(*args, **kwargs):
            stats = hit_stats(*args, **kwargs)
            sample_stats.append(stats)
            return stats

        with mock_patch.object(model, "generate", side_effect=generate_until_eos), \
                mock_patch.object(cache, "prefetch_hit_stats", side_effect=record_stats):
            report = evaluate_generation(model, Processor(), records, [1, 2, 4], max_new_tokens=8)
        self.assertEqual(report["generated_tokens"], 6)
        self.assertEqual(report["decode_tokens"], 4)
        self.assertEqual([stats["num_positions"] for stats in sample_stats], [4, 12])
        self.assertEqual(report["num_positions"], 16)
        for k in (1, 2, 4):
            self.assertEqual(report["avg_hits"][k], sum(s["total_hits"][k] for s in sample_stats) / 16)

        # 首次采样即 EOS 时，只有 prefill 路由，decode 指标为 null。
        with mock_patch.object(model, "generate", side_effect=lambda **kwargs: generate(
                **kwargs, eos_token_id=3, prefix_allowed_tokens_fn=lambda batch_id, ids: [3])):
            report = evaluate_generation(model, Processor(), records[:1], [1, 2, 4], max_new_tokens=8)
        self.assertEqual(report["generated_tokens"], 1)
        self.assertEqual(report["decode_tokens"], 0)
        self.assertEqual(report["num_positions"], 0)
        self.assertIsNone(report["loss"])
        self.assertEqual(report["avg_hits"], {1: None, 2: None, 4: None})

    def test_long_prompt_capacity_and_length_limit(self):
        model = generation_model()
        patch(model, (0, 2))
        records = [{"id": "long", "images": [], "messages": [
            {"role": "user", "content": "a" * 2046},
        ]}]
        with self.assertRaises(OverlengthSample):
            evaluate_generation(model, Processor(), records, [1, 2, 4], max_length=2048)
        report = evaluate_generation(model, Processor(), records, [1, 2, 4],
                                     max_length=2049, max_new_tokens=3)
        self.assertEqual(model.router_cache.max_tokens, 2054)
        self.assertEqual(report["decode_tokens"], 2)
        self.assertEqual(report["num_positions"], 8)
        self.assertEqual(report["avg_hits"][4], 2)


if __name__ == "__main__":
    unittest.main()
