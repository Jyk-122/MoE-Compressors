"""验证 YAML、原训练数据格式、梯度累积和分验证集评估。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch as mock_patch

from PIL import Image
import torch
import transformers
import yaml

from prefetch.datasets.dataset import OverlengthSample, training_records
from ..patch import patch
from ..train import main, make_accelerator, make_loaders, read_config, train
from .test_generation import tiny_generation_model


class Processor:
    tokenizer = SimpleNamespace(all_special_ids=[1, 2, 3, 4])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ""
        for message in messages:
            if message["role"] == "user":
                text += "U"
            else:
                text += "A"
            for item in message["content"]:
                if item["type"] == "image":
                    text += "I"
                else:
                    text += item["text"]
            text += "E"
        if add_generation_prompt:
            text += "A"
        return text

    def __call__(self, text, images=None, **kwargs):
        ids = []
        for char in text[0]:
            if char == "I":
                ids.extend([4, 4, 4])
            elif char in "UAE":
                ids.append({"U": 1, "A": 2, "E": 3}[char])
            else:
                ids.append(ord(char) - ord("a") + 6)
        input_ids = torch.tensor([ids])
        batch = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        if images:
            batch["pixel_values"] = torch.ones(len(images), 3, 4, 4)
            batch["image_grid_thw"] = torch.tensor([[1, 1, 3]] * len(images))
        return batch


def training_model():
    model = tiny_generation_model()
    original_forward = model.forward
    model.forward_calls = []

    def forward(input_ids=None, attention_mask=None, past_key_values=None, use_cache=None,
                pixel_values=None, image_grid_thw=None, logits_to_keep=1,
                output_router_logits=False, **kwargs):
        if "labels" in kwargs or "router_mask" in kwargs:
            raise AssertionError("Router supervision fields belong to the training loop")
        model.forward_calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "use_cache": use_cache,
            "has_image": pixel_values is not None and image_grid_thw is not None,
            "tokens": input_ids.detach().cpu().tolist(),
        })
        return original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                past_key_values=past_key_values, use_cache=use_cache, **kwargs)

    model.forward = forward
    return model


def make_config(directory):
    config_path = Path(__file__).resolve().parent.parent / "configs" / "train.yaml"
    config = read_config(config_path)
    image_path = directory / "image.png"
    Image.new("RGB", (4, 4)).save(image_path)
    sources = {}
    for source in ("text", "vl"):
        rows = []
        for index in range(2):
            images = []
            if source == "vl":
                images = [str(image_path)]
            rows.append({
                "id": f"{source}/{index}", "source": source, "images": images,
                "messages": [{"role": "user", "content": "a"},
                             {"role": "assistant", "content": "bc"}],
            })
        path = directory / f"{source}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        sources[source] = str(path)
    config["output_dir"] = str(directory / "output")
    config["model"].update(path="test-model", dtype="float32")
    config["prefetch"].update(mode_prefetch=[0, 1], ks=[1, 2, 4])
    config["data"].update(
        max_length=32, train_limit=3,
        train=[dict(path=sources["text"], weight=0.5), dict(path=sources["vl"], weight=0.5)],
        validation={name: dict(path=path, limit=1) for name, path in sources.items()},
    )
    config["training"].update(max_steps=3, gradient_accumulation=2, learning_rate=0.01,
                              warmup_steps=1, log_every=1, eval_every=2, save_every=2)
    return config


class TrainingTests(unittest.TestCase):
    def test_original_data_mixing_and_multimodal_collation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            loader, validation = make_loaders(Processor(), config["data"])
            expected = training_records(config["data"])
            self.assertEqual(list(loader.dataset), list(expected))
            self.assertEqual(len(loader.dataset), 3)
            self.assertEqual(set(validation), {"text", "vl"})
            text = next(iter(validation["text"]))
            image = next(iter(validation["vl"]))
            self.assertNotIn("pixel_values", text)
            self.assertIn("pixel_values", image)
            self.assertIn("image_grid_thw", image)
            self.assertEqual(image["router_mask"].sum().item(), 2)
            self.assertFalse(image["router_mask"][image["input_ids"] == 4].any())
            self.assertEqual(image["input_ids"][image["router_mask"]].tolist(), [7, 8])

            config["data"]["max_length"] = 2
            _, validation = make_loaders(Processor(), config["data"])
            with self.assertRaises(OverlengthSample):
                next(iter(validation["vl"]))

    def test_training_accumulation_evaluation_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            model = training_model()
            patch(model, config["prefetch"]["mode_prefetch"])
            original = {name: value.detach().clone() for name, value in model.named_parameters()}
            loader, validation = make_loaders(Processor(), config["data"])
            accelerator = make_accelerator(config["training"], cpu=True)
            with mock_patch.object(model.router_cache, "loss", wraps=model.router_cache.loss) as loss:
                step = train(model, loader, validation, config, accelerator)
            self.assertEqual(loss.call_count, len(model.forward_calls))
            for call in loss.call_args_list:
                self.assertEqual(call.kwargs["temperature"], config["prefetch"]["temperature"])
            self.assertEqual(step, 3)
            # 每次更新累积两个样本，跨过数据末尾后继续累积。
            calls = model.forward_calls
            self.assertEqual(sum(call["grad_enabled"] for call in calls), 6)
            self.assertTrue(any(call["has_image"] for call in calls))
            self.assertTrue(all(call["use_cache"] is False for call in calls))
            changed_heads = 0
            for name, parameter in model.named_parameters():
                if "gate_prefetch" in name:
                    if not torch.equal(parameter, original[name]):
                        changed_heads += 1
                else:
                    torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
            self.assertGreater(changed_heads, 0)

            output = Path(config["output_dir"])
            self.assertTrue((output / "run_config.yaml").is_file())
            self.assertTrue((output / "checkpoint-2.safetensors").is_file())
            self.assertTrue((output / "gate_prefetch.safetensors").is_file())
            for step in (2, 3):
                report = json.loads((output / "metrics" / f"step-{step}.json").read_text())
                self.assertEqual(set(report), {"vl", "text"})
                for result in report.values():
                    self.assertEqual(result["num_positions"], 8)  # 2 assistant token × 4 层
                    self.assertEqual(result["avg_hits"]["4"], 2.0)
            logs = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            self.assertEqual([entry["step"] for entry in logs], [1, 2, 3])
            self.assertNotEqual(logs[0]["learning_rate"], logs[-1]["learning_rate"])

    def test_yaml_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            config["training"]["max_steps"] = 1
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            model = training_model()
            model_loader = Mock(return_value=model)
            processor_loader = Mock(return_value=Processor())
            replacements = {
                "AutoProcessor": SimpleNamespace(from_pretrained=processor_loader),
                "Qwen3_5MoeForConditionalGeneration": SimpleNamespace(from_pretrained=model_loader),
            }
            with mock_patch.dict(vars(transformers), replacements):
                with mock_patch.object(sys, "argv", ["train", "--config", str(path)]):
                    with mock_patch.dict("os.environ", {"ACCELERATE_USE_CPU": "true"}):
                        main()
            processor_loader.assert_called_once_with("test-model")
            model_loader.assert_called_once_with("test-model", dtype=torch.float32,
                                                 device_map={"": torch.device("cpu")})
            self.assertTrue((Path(config["output_dir"]) / "gate_prefetch.safetensors").is_file())

    def test_max_steps_and_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            config["training"].update(max_steps=4, gradient_accumulation=4)
            model = training_model()
            patch(model, config["prefetch"]["mode_prefetch"])
            loader, _ = make_loaders(Processor(), config["data"])
            accelerator = make_accelerator(config["training"], cpu=True)
            self.assertEqual(train(model, loader, {}, config, accelerator), 4)
            self.assertEqual(sum(call["grad_enabled"] for call in model.forward_calls), 16)
            path = Path(directory) / "config.yaml"
            for key in ("max_steps", "gradient_accumulation", "eval_every", "learning_rate"):
                invalid = copy.deepcopy(config)
                invalid["training"][key] = 0
                path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read_config(path)
            for value in (None, 1.5, True):
                invalid = copy.deepcopy(config)
                invalid["training"]["max_steps"] = value
                path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                with self.subTest(max_steps=value), self.assertRaises(ValueError):
                    read_config(path)
            for value in (0, -1, float("nan"), float("inf")):
                invalid = copy.deepcopy(config)
                invalid["prefetch"]["temperature"] = value
                path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                with self.subTest(temperature=value), self.assertRaisesRegex(ValueError, "prefetch.temperature"):
                    read_config(path)


if __name__ == "__main__":
    unittest.main()
