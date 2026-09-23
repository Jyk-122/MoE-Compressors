"""Normalize CogVLM/Tulu and filter complete examples before distributed training."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from prefetch.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def validation_group(key, fraction, seed):
    value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
    return value < fraction


def cog_records(root, caption_prompt, limit=None, caption_prompt_zh="请详细描述这张图片。"):
    """Yield English and Chinese examples, counting each caption separately toward limit."""
    if limit is not None and limit <= 0:
        return
    count = 0
    root = Path(root)
    labels = [label for directory in ("labels_en", "labels_zh")
              for label in root.glob(f"**/{directory}/*.json")]
    if not labels:
        labels = list(root.glob("**/labels/*.json"))
    # Keep both languages of each image together before applying the sample limit.
    labels.sort(key=lambda label: (label.parent.parent, label.stem, label.parent.name))
    if not labels:
        raise ValueError("Expected CogVLM folders containing images/ and labels_en/labels_zh (or labels/)")
    for label in labels:
        image_files = sorted((label.parent.parent / "images").glob(label.stem + ".*"))
        if len(image_files) != 1:
            logger.warning("Skipping %s: expected one matching image, found %d (%s)",
                           label, len(image_files), image_files)
            continue
        image = image_files[0].resolve()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        payload = json.loads(label.read_text(encoding="utf-8"))
        if "conversations" in payload:
            variants = [payload["conversations"]]
        else:
            prompt = caption_prompt_zh if label.parent.name == "labels_zh" else caption_prompt
            variants = [[{"role": "user", "content": prompt},
                         {"role": "assistant", "content": caption["content"]}]
                        for caption in payload["captions"]]
        for index, messages in enumerate(variants):
            cleaned = []
            for message in messages:
                if message["role"] not in {"system", "user", "assistant"}:
                    raise ValueError(f"Unexpected role in {label}: {message['role']}")
                cleaned.append({"role": message["role"], "content": message["content"].replace("<image>", "").strip()})
            yield dict(id=f"cog/{label.relative_to(root).as_posix()}/{index}", source="cogvlm",
                       group_id=digest, messages=cleaned, images=[str(image)])
            count += 1
            if limit is not None and count >= limit:
                return


def tulu_records(dataset_name, limit):
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield dict(id=f"tulu/{example['id']}", group_id=str(example["id"]),
                   source=f"tulu/{example['source']}", messages=example["messages"], images=[])


def write_splits(records, directory, fraction, seed, total=None, desc="Prepare"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0}
    with logging_redirect_tqdm(), (directory / "train.jsonl").open("w", encoding="utf-8") as train, \
            (directory / "validation.jsonl").open("w", encoding="utf-8") as validation:
        # records is lazy: this bar covers dataset reading, normalization and writing.
        for record in tqdm(records, total=total, desc=desc, unit="sample", dynamic_ncols=True):
            split = "validation" if validation_group(record["group_id"], fraction, seed) else "train"
            file = validation if split == "validation" else train
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
    return counts


def filter_records(args):
    from transformers import AutoProcessor
    from prefetch.datasets.dataset import MissingUserTurn, OverlengthSample, SFTCollator
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    collate = SFTCollator(processor, args.max_length, max_image_side=args.max_image_side)
    kept = rejected = missing_user_turn = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with logging_redirect_tqdm(), Path(args.input).open(encoding="utf-8") as source, \
            output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(tqdm(source, desc=f"Filter {Path(args.input).name}",
                                                unit="sample", dynamic_ncols=True), start=1):
            record = json.loads(line)
            try:
                batch = collate([record])
            except OverlengthSample:
                rejected += 1
                continue
            except MissingUserTurn as error:
                missing_user_turn += 1
                logger.warning("Skipping %s (%s:%d): %s", record.get("id", "<unknown>"),
                               args.input, line_number, error)
                continue
            record["num_tokens"] = batch["input_ids"].shape[1]
            record["assistant_text_tokens"] = int(batch["router_mask"].sum())
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return dict(kept=kept, overlength=rejected, missing_user_turn=missing_user_turn, processor=args.model_path,
                max_length=args.max_length, max_image_side=args.max_image_side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cog", "tulu"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
        command.add_argument("--validation-fraction", type=float, default=0.01)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--limit", type=int, help="Maximum examples before splitting (default: all)")
        if name == "cog":
            command.add_argument("--root", required=True)
            command.add_argument("--caption-prompt", default="Describe this image in detail.")
            command.add_argument("--caption-prompt-zh", default="请详细描述这张图片。")
        else:
            command.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    command = commands.add_parser("filter")
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--model-path", required=True)
    command.add_argument("--max-length", type=int, default=2048)
    command.add_argument("--max-image-side", type=int, default=672)
    args = parser.parse_args()
    configure_logging()
    if args.command == "filter":
        report = filter_records(args)
        metadata_path = Path(args.output).with_suffix(".meta.json")
    else:
        if not 0 < args.validation_fraction < 1:
            parser.error("validation-fraction must lie strictly between 0 and 1")
        if args.limit is not None and args.limit < 0:
            parser.error("limit must be non-negative")
        if args.command == "cog":
            records = cog_records(args.root, args.caption_prompt, args.limit, args.caption_prompt_zh)
        else:
            records = tulu_records(args.dataset, args.limit)
        report = write_splits(records, args.output, args.validation_fraction, args.seed,
                              total=args.limit, desc=f"Prepare {args.command}")
        report.update(vars(args))
        metadata_path = Path(args.output) / "metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
