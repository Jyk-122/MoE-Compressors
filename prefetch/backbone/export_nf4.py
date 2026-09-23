"""Quantize the base on CPU, move to one GPU, and export a packed NF4 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prefetch.backbone.loading import load_model
from prefetch.backbone.nf4_checkpoint import save_nf4_checkpoint
from prefetch.training.runtime import read_config
from prefetch.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New checkpoint directory")
    parser.add_argument("--blocksize", type=int, choices=[64, 128, 256, 512, 1024, 2048, 4096])
    args = parser.parse_args()
    configure_logging()
    if Path(args.output).exists():
        parser.error("output already exists; choose a new checkpoint directory")
    config = read_config(args.config)
    config["stage"] = "base"
    config["lora"] = {"enabled": False}
    base = config["model"]
    base.update(quantization="experts_nf4", nf4_checkpoint=None)
    if args.blocksize is not None:
        base["nf4_blocksize"] = args.blocksize
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, _, _ = load_model(config, device)
    metadata = save_nf4_checkpoint(model, args.output, base["path"], base.get("nf4_blocksize", 64))
    print(json.dumps(dict(checkpoint=args.output, blocksize=metadata["blocksize"],
                          quantized_source_parameters=metadata["quantized_source_parameters"]), indent=2))


if __name__ == "__main__":
    main()
