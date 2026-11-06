"""Launch a vLLM 0.24.0 OpenAI API server with trained MoE compensation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from .routing import validate_threshold


def make_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     epilog="Additional arguments are passed to vllm serve.")
    parser.add_argument("--model", required=True, help="Original Qwen3.6-35B-A3B model directory")
    parser.add_argument("--checkpoint", required=True, help="Trained compensation.safetensors")
    parser.add_argument("--threshold", type=validate_threshold, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--served-model-name", default="qwen3.6-pruned")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting vLLM")
    args, extra = parser.parse_known_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Compensation checkpoint does not exist: {checkpoint}")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    for option in extra:
        if option.split("=", 1)[0] in {"--worker-cls", "--additional-config", "--moe-backend"}:
            parser.error(f"{option} is configured by the compensation launcher")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--dtype", "bfloat16", "--served-model-name", args.served_model_name,
               "--host", args.host, "--port", str(args.port),
               "--tensor-parallel-size", str(args.tensor_parallel_size),
               "--max-model-len", str(args.max_model_len),
               "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--max-num-seqs", str(args.max_num_seqs),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               *extra, "--enforce-eager", "--moe-backend", "triton",
               "--worker-cls", "skipping_qwen3_5_moe.vllm_worker.CompensationWorker",
               "--additional-config", json.dumps({"skipping_compensation": {
                   "checkpoint": str(checkpoint), "threshold": args.threshold}})]
    return command, args.dry_run


def main():
    command, dry_run = make_command()
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [root, environment.get("PYTHONPATH")]))
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
