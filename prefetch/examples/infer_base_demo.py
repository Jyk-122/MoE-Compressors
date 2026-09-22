"""Single-prompt BF16/NF4 base inference, without a prerouter or attention adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from prefetch.backbone.loading import load_model
from prefetch.datasets.dataset import multimodal_messages, open_images, processor_call, to_device
from prefetch.training.runtime import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quantization", choices=["none", "experts_nf4"], help="Override the YAML setting")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", help="Optional JSON report")
    args = parser.parse_args()
    config = read_config(args.config)
    config["stage"], config["lora"] = "base", {"enabled": False}
    if args.quantization:
        config["model"]["quantization"] = args.quantization
        if args.quantization == "none":
            config["model"]["nf4_checkpoint"] = None
    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, processor, loading = load_model(config, device)
    paths = [args.image] if args.image else []
    messages = multimodal_messages([dict(role="user", content=args.prompt)], paths)
    images = open_images(paths, config["data"].get("max_image_side", 672))
    inputs = to_device(processor_call(processor, messages, images, generation=True), device)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                                max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    report = dict(prompt=args.prompt, image=args.image, answer=answer, loading=loading,
                  generated_tokens=output.shape[1] - inputs["input_ids"].shape[1],
                  generation_seconds=elapsed, peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    print(answer)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
