"""Single-example multimodal collation with explicit assistant-token alignment."""
from __future__ import annotations

from copy import deepcopy

import torch
from PIL import Image


class OverlengthSample(ValueError):
    pass


class MissingUserTurn(ValueError):
    pass


def open_images(paths, max_image_side=None):
    images = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if max_image_side:
                image.thumbnail((max_image_side, max_image_side))
            images.append(image.copy())
    return images


def multimodal_messages(messages, image_paths):
    messages = deepcopy(messages)
    inserted = False
    for message in messages:
        content = [{"type": "text", "text": message["content"]}]
        if image_paths and message["role"] == "user" and not inserted:
            content = [{"type": "image", "image": path} for path in image_paths] + content
            inserted = True
        message["content"] = content
    if image_paths and not inserted:
        raise MissingUserTurn("An image example needs a user turn")
    return messages


def processor_call(processor, messages, images, generation=False):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
    if not generation:
        text = text.removesuffix("<|message_start|>助手：</think>")
    kwargs = dict(text=[text], padding=False, return_tensors="pt")
    if images:
        kwargs["images"] = images
    return processor(**kwargs)


class SFTCollator:
    def __init__(self, processor, max_length=2048, router_tokens="assistant", max_image_side=672):
        if router_tokens != "assistant":
            raise ValueError("router_tokens must be assistant: router supervision uses response inputs")
        self.processor = processor
        self.max_length = max_length
        self.max_image_side = max_image_side
        self.special_ids = set(processor.tokenizer.all_special_ids)

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("Use micro_batch_size=1; use gradient accumulation for a larger training batch")
        example = examples[0]
        paths = example.get("images") or []
        messages = multimodal_messages(example["messages"], paths)
        images = open_images(paths, self.max_image_side)
        batch = dict(processor_call(self.processor, messages, images))
        ids = batch["input_ids"]
        if ids.shape[1] > self.max_length:
            raise OverlengthSample(f"{example['id']}: {ids.shape[1]} tokens > {self.max_length}. "
                                   "Run prepare_data filter with the same processor/image/length settings.")
        assistant = torch.zeros_like(ids, dtype=torch.bool)
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = processor_call(self.processor, messages[:i], images, generation=True)["input_ids"]
            after = processor_call(self.processor, messages[:i + 1], images)["input_ids"]
            start, end = before.shape[1], after.shape[1]
            if start >= end or not torch.equal(ids[:, :start], before) or not torch.equal(ids[:, :end], after):
                raise ValueError(f"{example['id']}: chat-template prefixes do not align with the full sequence; "
                                 "adapt assistant-span extraction to this checkpoint's processor")
            assistant[:, start:end] = True
        valid = batch.setdefault("attention_mask", torch.ones_like(ids)).bool()
        text_tokens = valid.clone()
        for token_id in self.special_ids:
            text_tokens &= ids != token_id
        router_mask = text_tokens & assistant
        labels = ids.clone()
        labels[~(assistant & valid)] = -100
        if not (assistant & text_tokens).any() or not (labels[:, 1:] != -100).any():
            raise ValueError(f"{example['id']}: no assistant text tokens usable for supervision")
        batch.update(labels=labels, router_mask=router_mask)
        return batch


def load_records(paths, limit=None):
    from datasets import Sequence, Value, load_dataset
    if isinstance(paths, str):
        paths = [paths]
    dataset = load_dataset("json", data_files=paths, split="train")
    # Text-only files contain [] throughout; Arrow infers list<null> without this cast.
    image_feature = Sequence(Value("string"))
    if dataset.features["images"] != image_feature:
        dataset = dataset.cast_column("images", image_feature)
    return dataset.select(range(min(limit, len(dataset)))) if limit else dataset


def training_records(config):
    """Mix training sources, then cap the shared dataset before DDP sharding."""
    limit = config.get("train_limit")
    if limit is not None and limit < 1:
        raise ValueError("data.train_limit must be positive")
    from datasets import interleave_datasets
    sources = config["train"]
    parts = [load_records(source["path"], source.get("limit")) for source in sources]
    if len(parts) == 1:
        dataset = parts[0]
    else:
        probabilities = [source.get("weight", 1.0) for source in sources]
        total = sum(probabilities)
        dataset = interleave_datasets(parts, probabilities=[x / total for x in probabilities],
                                      seed=config.get("seed", 42), stopping_strategy="all_exhausted")
    return dataset.select(range(min(limit, len(dataset)))) if limit is not None else dataset


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
