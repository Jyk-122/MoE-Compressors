"""Token-weighted assistant NLL for comparing BF16 and NF4 on fixed held-out data."""
from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from prefetch.backbone.loading import load_model
from prefetch.datasets.dataset import to_device
from prefetch.training.runtime import make_collator, read_config
from prefetch.utils.logging import configure_logging


@torch.inference_mode()
def evaluate_base(model, processor, data, sample_file, device, limit=None):
    collate = make_collator(processor, data)
    total_nll, tokens, examples = 0.0, 0, 0
    model.eval()
    with Path(sample_file).open(encoding="utf-8") as source:
        for line in tqdm(islice(source, limit), total=limit, desc="Base NLL", unit="sample"):
            batch = to_device(collate([json.loads(line)]), device)
            batch.pop("router_mask")
            count = int((batch["labels"][:, 1:] != -100).sum())
            loss = model(**batch, use_cache=False).loss.float()
            if not torch.isfinite(loss):
                raise ValueError("Base evaluation loss is not finite")
            total_nll += float(loss) * count
            tokens += count
            examples += 1
    if tokens == 0:
        raise ValueError("No supervised next-token targets in the evaluation set")
    nll = total_nll / tokens
    return dict(assistant_nll=nll, assistant_perplexity=math.exp(nll), tokens=tokens, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quantization", choices=["none", "experts_nf4"])
    parser.add_argument("--sample-file", required=True, help="Same held-out filtered JSONL for both runs")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_logging()
    if args.limit <= 0:
        parser.error("limit must be positive")
    config = read_config(args.config)
    config["stage"], config["lora"] = "base", {"enabled": False}
    if args.quantization:
        config["model"]["quantization"] = args.quantization
        if args.quantization == "none":
            config["model"]["nf4_checkpoint"] = None
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, processor, loading = load_model(config, device)
    report = evaluate_base(model, processor, config["data"], args.sample_file, device, args.limit)
    report.update(sample_file=args.sample_file, data=config["data"], loading=loading)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
