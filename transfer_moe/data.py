"""JSONL conversations and image processing for calibration and generation."""

from copy import deepcopy
from itertools import islice
import json

from PIL import Image
import torch
from torch.utils.data import DataLoader


def open_images(paths, max_image_side=None):
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if max_image_side is not None:
                image.thumbnail((max_image_side, max_image_side))
            images.append(image)
    return images


def multimodal_messages(messages, image_paths):
    messages = deepcopy(messages)
    for message in messages:
        message["content"] = [{"type": "text", "text": message["content"]}]
    if image_paths:
        user = next(message for message in messages if message["role"] == "user")
        user["content"] = [{"type": "image", "image": path} for path in image_paths] + user["content"]
    return messages


def processor_call(processor, messages, images, generation=False):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation,
        enable_thinking=False, preserve_thinking=True)
    kwargs = dict(text=[text], padding=False, return_tensors="pt")
    if images:
        kwargs["images"] = images
    return processor(**kwargs)


def load_records(paths, limit=None):
    def records():
        for path in paths:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)
    return list(islice(records(), limit))


class CalibrationCollator:
    """Truncate a conversation to max_length, keeping image blocks complete."""

    def __init__(self, processor, max_length=2048, max_image_side=672):
        self.processor = processor
        self.max_length = max_length
        self.max_image_side = max_image_side

    def __call__(self, examples):
        example, = examples
        paths = example.get("images", [])
        messages = multimodal_messages(example["messages"], paths)
        images = open_images(paths, self.max_image_side)
        batch = processor_call(self.processor, messages, images)
        length = self.max_length
        if "image_grid_thw" in batch:
            ids = batch["input_ids"][0]
            starts = (ids == self.processor.vision_start_token_id).nonzero().flatten()
            ends = (ids == self.processor.vision_end_token_id).nonzero().flatten()
            count = int((ends < length).sum())
            if count < starts.numel():
                length = min(length, int(starts[count]))
            if count:
                grid = batch["image_grid_thw"][:count]
                batch["pixel_values"] = batch["pixel_values"][:int(grid.prod(-1).sum())]
                batch["image_grid_thw"] = grid
            else:
                del batch["pixel_values"], batch["image_grid_thw"]
        for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"):
            if key in batch:
                batch[key] = batch[key][:, :length]
        return batch


def make_loader(processor, paths, limit=32, max_length=2048, max_image_side=672):
    records = load_records(paths, limit)
    collator = CalibrationCollator(processor, max_length, max_image_side)
    return DataLoader(records, batch_size=1, shuffle=False, collate_fn=collator)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
