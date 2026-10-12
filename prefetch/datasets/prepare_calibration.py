"""Export the OptimalScale text/image mixture as processor-validated JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random

from PIL import Image
from tqdm import tqdm

from prefetch.datasets.dataset import (
    MissingUserTurn, OverlengthSample, SFTCollator, multimodal_messages, processor_call,
)
from prefetch.datasets.prepare import cog_records


SOURCES = ("slimpajama", "codealpaca", "cogvlm", "chartqa", "alpaca_zh")
CONTINUATION_PROMPT = "Continue the following text:\n"
logger = logging.getLogger(__name__)


def record(source, index, messages, images=()):
    key = f"{source}/{index}"
    return dict(id=key, group_id=key, source=source, messages=messages, images=list(images))


def conversation(question, answer):
    return [dict(role="user", content=question), dict(role="assistant", content=answer)]


def text_length(processor, messages):
    return processor_call(processor, multimodal_messages(messages, []), [])["input_ids"].shape[1]


def text_chunks(rows, processor, max_length, prefix_tokens=64):
    """Pack raw text tokens, then expose the continuation as the assistant span."""
    tokenizer = processor.tokenizer
    overhead = text_length(processor, conversation(CONTINUATION_PROMPT, "x"))
    chunk_size = max_length - overhead - 16
    if chunk_size < 2:
        raise ValueError("max_length leaves no room for a text continuation")
    pending, index = [], 0
    text_column = None

    def make_chunk(tokens, index):
        split = min(prefix_tokens, max(1, len(tokens) // 4))
        prefix = tokenizer.decode(tokens[:split], skip_special_tokens=True)
        answer = tokenizer.decode(tokens[split:], skip_special_tokens=True)
        return record("slimpajama", index, conversation(CONTINUATION_PROMPT + prefix, answer))

    for item in rows:
        if text_column is None:
            # Match the original OptimalScale loader's column selection.
            text_column = "text" if "text" in item else next(iter(item))
            logger.info("SlimPajama text column: %s; available columns: %s", text_column, list(item))
        value = item[text_column]
        if not value or not str(value).strip():
            continue
        text = str(value)
        pending.extend(tokenizer.encode(text + "\n\n", add_special_tokens=False))
        while len(pending) >= chunk_size:
            yield make_chunk(pending[:chunk_size], index)
            pending = pending[chunk_size:]
            index += 1
    if len(pending) > 1:
        yield make_chunk(pending, index)


def instruction_records(rows, source):
    suffix = "_zh" if source == "alpaca_zh" else ""
    for index, item in enumerate(rows):
        question = item[f"instruction{suffix}"].strip()
        context = (item.get(f"input{suffix}") or "").strip()
        answer = (item.get(f"output{suffix}") or "").strip()
        if context:
            question += "\n" + context
        if question and answer:
            yield record(source, index, conversation(question, answer))


def pack_instructions(records, processor, max_length):
    """Pack complete user/assistant turns within the configured context length."""
    messages, members, index = [], [], 0
    source = None
    for item in records:
        source = item["source"]
        candidate = messages + item["messages"]
        if text_length(processor, candidate) > max_length:
            if messages:
                yield dict(record(source, f"chunk-{index}", messages), members=members)
                index += 1
                messages, members = [], []
            # The common validator counts/rejects a single overlength conversation.
            if text_length(processor, item["messages"]) > max_length:
                yield item
                continue
        messages.extend(item["messages"])
        members.append(item["id"])
    if messages:
        yield dict(record(source, f"chunk-{index}", messages), members=members)


def cog_examples(root):
    # The original mixture uses English CogVLM annotations.
    for item in cog_records(root, "Describe this image in detail."):
        if "/labels_zh/" not in item["id"]:
            yield item


def chart_examples(rows, split):
    for index, item in enumerate(rows):
        question, answer = item["question"], item["answer"]
        if question and answer is not None and str(answer).strip():
            yield record("chartqa", f"{split}/{index}", conversation(question, str(answer)), [item["image"]])


def source_streams(args, processor, data):
    """Open each source lazily so a zero quota does not load that dataset."""
    from datasets import load_dataset
    root = Path(args.datasets_root)
    max_length = data.get("max_length", 2048)

    def slim():
        rows = load_dataset(str(root / "slimpajama-200k"), split="train", streaming=True)
        yield from text_chunks(rows, processor, max_length, args.text_prefix_tokens)

    def instructions(name, directory):
        rows = load_dataset(str(root / directory), split="train", streaming=True)
        yield from pack_instructions(instruction_records(rows, name), processor, max_length)

    def charts():
        rows = load_dataset(str(root / "ChartQA"), split=args.chart_split).shuffle(seed=args.seed)
        yield from chart_examples(rows, args.chart_split)

    cog_root = args.cog_root or root / "CogVLM-SFT-311K" / "CogVLM-SFT-311K" / "llava_instruction_single_conversation_formate"
    return dict(slimpajama=slim(),
                codealpaca=instructions("codealpaca", "CodeAlpaca-20k"),
                cogvlm=cog_examples(cog_root),
                chartqa=charts(),
                alpaca_zh=instructions("alpaca_zh", "alpaca-data-gpt4-chinese"))


def save_images(item, directory):
    """Persist both file-backed and decoded dataset images next to the JSONL."""
    saved = dict(item)
    paths = []
    if item["images"]:
        directory.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(item["id"].encode()).hexdigest()[:20]
    for slot, image in enumerate(item["images"]):
        destination = directory / f"{stem}-{slot}.png"
        if isinstance(image, Image.Image):
            image.convert("RGB").save(destination)
        else:
            with Image.open(image) as loaded:
                loaded.convert("RGB").save(destination)
        paths.append(str(destination.resolve()))
    saved["images"] = paths
    return saved


def interleave_sources(groups, seed):
    """Distribute each source proportionally throughout the file, including prefixes."""
    rng = random.Random(seed)
    ordered = []
    for source_index, source in enumerate(SOURCES):
        rows = list(groups[source])
        rng.shuffle(rows)
        for index, item in enumerate(rows):
            ordered.append(((index + 0.5) / len(rows), source_index, item))
    ordered.sort(key=lambda entry: entry[:2])
    return [item for _, _, item in ordered]


def export_mixture(streams, counts, processor, data, output, seed=42, metadata=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    collate = SFTCollator(processor, data.get("max_length", 2048),
                          router_tokens=data.get("router_tokens", "assistant"),
                          max_image_side=data.get("max_image_side", 672))
    groups, stats = {}, {}
    for source, count in zip(SOURCES, counts):
        accepted = []
        rejected = dict(overlength=0, missing_user_turn=0)
        if count:
            with tqdm(total=count, desc=f"Calibration data: {source}", unit="sample") as progress:
                for item in streams[source]:
                    item = save_images(item, output.parent / "images")
                    try:
                        batch = collate([item])
                    except OverlengthSample:
                        rejected["overlength"] += 1
                        continue
                    except MissingUserTurn:
                        rejected["missing_user_turn"] += 1
                        continue
                    item["num_tokens"] = batch["input_ids"].shape[1]
                    item["assistant_text_tokens"] = int(batch["router_mask"].sum())
                    accepted.append(item)
                    progress.update()
                    if len(accepted) == count:
                        break
            if len(accepted) < count:
                raise ValueError(f"{source}: only {len(accepted)}/{count} valid samples; rejected={rejected}")
        groups[source] = accepted
        stats[source] = dict(samples=len(accepted),
                             assistant_text_tokens=sum(item["assistant_text_tokens"] for item in accepted),
                             **rejected)
    rows = interleave_sources(groups, seed)
    with output.open("w", encoding="utf-8") as target:
        for item in rows:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = dict(metadata or {}, output=str(output.resolve()), seed=seed,
                  samples=len(rows), image_samples=sum(bool(item["images"]) for item in rows),
                  sources=stats, data=data)
    output.with_suffix(".meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Predictor checkpoint containing run_config.json; loads only its config")
    parser.add_argument("--config", help="Run YAML, using the same processor and data settings as calibration")
    parser.add_argument("--datasets-root", default="/data1/jiangyikun/datasets")
    parser.add_argument("--cog-root", help="CogVLM directory containing images/ and labels_en/")
    parser.add_argument("--chart-split", default="test", help="ChartQA split; original OptimalScale uses test")
    parser.add_argument("--counts", type=int, nargs=5, default=[32, 32, 64, 32, 32],
                        metavar=("SLIM", "CODE", "COG", "CHART", "ZH"),
                        help="Valid chunks/pairs per source, in the original mixture order")
    parser.add_argument("--text-prefix-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="prefetch/data/calib/mixed.jsonl")
    args = parser.parse_args()
    if not args.config and not args.checkpoint:
        parser.error("Provide --checkpoint or --config")
    if min(args.counts) < 0 or sum(args.counts) == 0 or args.text_prefix_tokens < 1:
        parser.error("counts must be nonnegative with a positive total; text-prefix-tokens must be positive")

    from transformers import AutoProcessor
    from prefetch.evaluation.evaluate import experiment_config
    from prefetch.utils.logging import configure_logging
    configure_logging()
    config = experiment_config(args.config, args.checkpoint)
    processor = AutoProcessor.from_pretrained(config["model"]["path"], trust_remote_code=True)
    report = export_mixture(source_streams(args, processor, config["data"]), args.counts,
                            processor, config["data"], args.output, args.seed,
                            metadata=dict(processor=config["model"]["path"], datasets_root=args.datasets_root,
                                          cog_root=args.cog_root, chart_split=args.chart_split,
                                          text_prefix_tokens=args.text_prefix_tokens))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
