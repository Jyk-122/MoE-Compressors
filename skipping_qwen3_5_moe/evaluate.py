"""Evaluate reconstruction, LM KL, and actual routing counts on held-out data."""
import argparse
import json
from pathlib import Path

from .config import read_config, load_model_and_processor
from .data import validation_loaders
from .model import load_compensation
from .train import make_task, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.threshold is not None:
        config["skipping"]["threshold"] = args.threshold
    model, processor = load_model_and_processor(config, args.device)
    load_compensation(model, args.checkpoint, threshold=config["skipping"]["threshold"])
    loaders = validation_loaders(processor, config["data"])
    if not loaders:
        raise ValueError("Configure data.validation for evaluation")
    report = {"threshold": config["skipping"]["threshold"],
              "validation": evaluate(make_task(model, config), loaders)}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
