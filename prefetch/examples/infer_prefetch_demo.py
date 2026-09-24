"""Single-request demo with native or prerouter-selected expert execution."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import torch

from prefetch.datasets.dataset import multimodal_messages, open_images, processor_call, to_device
from prefetch.evaluation.evaluate import experiment_config
from prefetch.backbone.loading import load_model
from prefetch.evaluation.metrics import save_report
from prefetch.prerouter.patch import patch
from prefetch.evaluation.plot import plot_report
from prefetch.evaluation.routing import RoutingMetrics, capture_generation
from prefetch.utils.logging import configure_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config; can be inferred from a trained checkpoint")
    parser.add_argument("--checkpoint", help="Omit for an initialized-head wiring smoke test")
    parser.add_argument("--prerouter-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Execute predicted experts with native weights; default follows config/checkpoint")
    parser.add_argument("--prompt", default="请描述一下这张图。")
    parser.add_argument("--image")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trace-limit", type=int, default=8)
    parser.add_argument("--trace-k", type=int, default=16)
    parser.add_argument("--output", default="prefetch/outputs/demo/metrics.json")
    args = parser.parse_args()
    configure_logging()
    config = experiment_config(args.config, args.checkpoint)
    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, processor, _ = load_model(config, device)
    state = patch(model, config["prefetch"], checkpoint=args.checkpoint,
                  prerouter_enabled=args.prerouter_enabled)
    logger.info("prerouter_enabled=%s", state.config.prerouter_enabled)
    state.config.trace_limit = args.trace_limit
    paths = [args.image] if args.image else []
    messages = multimodal_messages([dict(role="user", content=args.prompt)], paths)
    inputs = to_device(processor_call(processor, messages, open_images(paths, config["data"].get("max_image_side", 672)),
                                     generation=True), device)
    model.eval()
    meter = RoutingMetrics(state)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode(), capture_generation(model, state, meter):
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                num_beams=1, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    report = meter.report()
    report["demo"] = dict(prompt=args.prompt, image=args.image, answer=answer,
                          predictor="trained" if args.checkpoint else "initialized",
                          generation_with_instrumentation_seconds=elapsed,
                          generated_tokens=output.shape[1] - inputs["input_ids"].shape[1])
    report["trace"] = [dict(row, predictions=row["predictions"][:args.trace_k]) for row in meter.trace]
    save_report(report, args.output)
    plot_report(report, Path(args.output).with_suffix(""))
    print(answer)
    logger.info("decode %s", json.dumps(report["phases"]["decode"]["global"], ensure_ascii=False))
    logger.info("Metrics: %s; instrumented generation: %.3fs", args.output, elapsed)


if __name__ == "__main__":
    main()
