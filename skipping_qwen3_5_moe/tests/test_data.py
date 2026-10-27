import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch

from ..data import (MissingUserTurn, OverlengthSample, SFTCollator, load_records,
                    make_loader, to_device, training_records, validation_loaders)


class ToyProcessor:
    tokenizer = SimpleNamespace(all_special_ids=[1, 2, 3, 4])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ""
        for message in messages:
            text += "U" if message["role"] == "user" else "A"
            for content in message["content"]:
                text += "I" if content["type"] == "image" else content["text"]
            text += "E"
        return text + ("A" if add_generation_prompt else "")

    def __call__(self, text, images=None, **kwargs):
        self.image_sizes = [image.size for image in images or []]
        self.image_modes = [image.mode for image in images or []]
        sequence = []
        for char in text[0]:
            sequence.extend([4, 4, 4] if char == "I" else [{"U": 1, "A": 2, "E": 3}.get(char, ord(char))])
        ids = torch.tensor([sequence])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


def example(identifier="one", images=None):
    return {"id": identifier, "images": images or [], "messages": [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "why"}, {"role": "assistant", "content": "ok"}]}


class DataTests(unittest.TestCase):
    def test_text_and_image_assistant_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGBA", (8, 4)).save(path)
            for paths in ([], [str(path)]):
                with self.subTest(images=paths):
                    processor = ToyProcessor()
                    record = example(images=paths)
                    batch = SFTCollator(processor, max_image_side=4)([record])
                    text = "".join(chr(token) for token in batch["input_ids"][batch["router_mask"]].tolist())
                    self.assertEqual(text, "yesok")
                    self.assertEqual((batch["labels"] != -100).sum().item(), 7)
                    self.assertEqual((batch["input_ids"] == 4).sum().item(), 3 if paths else 0)
                    self.assertTrue((batch["labels"][batch["input_ids"] == 4] == -100).all())
                    self.assertEqual(processor.image_sizes, [(4, 2)] if paths else [])
                    self.assertEqual(processor.image_modes, ["RGB"] if paths else [])
                    self.assertEqual(record["messages"][0]["content"], "hello")

    def test_sample_validation_and_device_transfer(self):
        with self.assertRaises(OverlengthSample):
            SFTCollator(ToyProcessor(), max_length=2)([example()])
        with self.assertRaises(MissingUserTurn):
            SFTCollator(ToyProcessor())([{"id": "invalid", "images": ["unopened.png"],
                                         "messages": [{"role": "assistant", "content": "yes"}]}])
        with self.assertRaises(ValueError):
            SFTCollator(ToyProcessor())([example(), example()])
        batch = {"input_ids": torch.tensor([[1, 2]]), "id": "one"}
        moved = to_device(batch, torch.device("cpu"))
        torch.testing.assert_close(moved["input_ids"], batch["input_ids"])
        self.assertEqual(moved["id"], "one")

    def test_jsonl_mixing_and_loader_splits(self):
        from datasets import Sequence, Value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (4, 4)).save(image)
            text_path, vision_path = root / "text.jsonl", root / "vision.jsonl"
            text_path.write_text("".join(json.dumps(example(f"text-{i}")) + "\n" for i in range(3)),
                                 encoding="utf-8")
            vision_path.write_text("".join(json.dumps(example(f"vision-{i}", [str(image)])) + "\n"
                                           for i in range(2)), encoding="utf-8")
            sources = [{"path": str(text_path), "weight": 3}, {"path": str(vision_path), "weight": 1}]
            loaded = load_records(str(text_path), limit=2)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.features["images"], Sequence(Value("string")))
            mixed = training_records({"train": sources, "seed": 7})
            self.assertEqual({row["id"].split("-")[0] for row in mixed}, {"text", "vision"})
            config = {"train": sources, "seed": 7, "train_limit": 4, "max_length": 128,
                      "calibration": [{"path": str(vision_path)}], "calibration_limit": 1,
                      "validation": {"text": {"path": str(text_path), "limit": 2}}}
            training = make_loader(ToyProcessor(), config)
            calibration = make_loader(ToyProcessor(), config, split="calibration")
            validation = validation_loaders(ToyProcessor(), config)
            self.assertEqual(len(training.dataset), 4)
            self.assertEqual(len(calibration.dataset), 1)
            self.assertEqual(calibration.dataset[0]["id"], "vision-0")
            self.assertEqual(len(validation["text"].dataset), 2)
            for loader in (training, calibration, validation["text"]):
                batch = next(iter(loader))
                self.assertEqual(batch["input_ids"].shape[0], 1)
                self.assertTrue((batch["labels"][:, 1:] != -100).any())


if __name__ == "__main__":
    unittest.main()
