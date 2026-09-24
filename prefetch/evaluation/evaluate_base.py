"""Token-weighted assistant NLL for quantization and expert-routing comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from prefetch.backbone.loading import load_model
from prefetch.datasets.dataset import to_device
from prefetch.evaluation.evaluate import experiment_config
from prefetch.prerouter.patch import patch
from prefetch.training.runtime import make_collator, read_config
from prefetch.utils.logging import configure_logging


@torch.inference_mode()
def evaluate_base(model, processor, data, sample_file, device, limit=None):
    collate = make_collator(processor, data)
    total_nll, tokens, examples = 0.0, 0, 0
    model.eval()
    state = getattr(model, "prerouter_state", None)
    with Path(sample_file).open(encoding="utf-8") as source:
        for line in tqdm(islice(source, limit), total=limit, desc="Assistant NLL", unit="sample"):
            batch = to_device(collate([json.loads(line)]), device)
            router_mask = batch.pop("router_mask")
            count = int((batch["labels"][:, 1:] != -100).sum())
            if state is not None:
                state.reset(router_mask=router_mask)
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
    parser.add_argument("--config", help="YAML config; can be inferred from a predictor checkpoint")
    parser.add_argument("--checkpoint", help="Trained predictor checkpoint for routing comparisons")
    parser.add_argument("--prerouter-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Execute predicted experts with native weights; default follows config/checkpoint")
    parser.add_argument("--quantization", choices=["none", "experts_nf4"])
    parser.add_argument("--sample-file", required=True, help="Same held-out filtered JSONL for both runs")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_logging()
    if args.limit <= 0:
        parser.error("limit must be positive")
    if not args.config and not args.checkpoint:
        parser.error("Provide --config or --checkpoint")
    if args.prerouter_enabled and not args.checkpoint:
        parser.error("--prerouter-enabled requires a trained --checkpoint")
    if args.checkpoint:
        config = experiment_config(args.config, args.checkpoint)
        if args.quantization and args.quantization != config["model"].get("quantization", "none"):
            parser.error("Routing comparison requires the checkpoint's base quantization")
    else:
        config = read_config(args.config)
        config["stage"], config["lora"] = "base", {"enabled": False}
    if args.quantization:
        config["model"]["quantization"] = args.quantization
        if args.quantization == "none":
            config["model"]["nf4_checkpoint"] = None
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, processor, loading = load_model(config, device)
    state = (patch(model, config.get("prefetch"), checkpoint=args.checkpoint,
                   prerouter_enabled=args.prerouter_enabled) if args.checkpoint else None)
    report = evaluate_base(model, processor, config["data"], args.sample_file, device, args.limit)
    report.update(sample_file=args.sample_file, data=config["data"], loading=loading)
    if state is not None:
        report.update(checkpoint=args.checkpoint, prefetch=asdict(state.config),
                      routing_scope="teacher_forced_response_inputs", prefill_routing="native")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
