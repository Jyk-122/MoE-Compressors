#!/usr/bin/env python3
"""
MoE-Compressors 统一运行入口

用法:
  python run.py <method> <action1> [action2] ... [options]
  python run.py frequency_pruning eval --model ...
  python run.py frequency_pruning patch eval --model ... --adapter_dir ...
  python run.py frequency_pruning calib patch eval --model ... --adapter_dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from methods.frequency_pruning.model_qwen3_moe import FrequencyPruningQwen3Moe

# 方法注册表：method_name -> (compressor_cls, default_model_type)
METHOD_REGISTRY = {
    "frequency_pruning": (FrequencyPruningQwen3Moe, "qwen3_moe"),
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE-Compressors: 支持 calib / patch / eval 任意组合"
    )
    parser.add_argument(
        "method",
        choices=list(METHOD_REGISTRY),
        help="压缩方法名称",
    )
    parser.add_argument(
        "actions",
        nargs="+",
        choices=["calib", "patch", "eval"],
        help="要执行的操作，按顺序执行",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="adapter 目录。eval 可不传，calib/patch 时必填")
    parser.add_argument("--model_type", type=str, default=None,
                        help="模型架构，默认由各方法决定")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--calibration_dataset", type=str, default="wikitext:wikitext-2-raw-v1")
    parser.add_argument("--max_calib_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_model_path", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=["wikitext"])
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=float, default=None)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if "calib" in args.actions or "patch" in args.actions:
        if args.adapter_dir is None:
            raise ValueError("执行 calib 或 patch 时需提供 --adapter_dir")

    cls, default_model_type = METHOD_REGISTRY[args.method]
    model_type = args.model_type or default_model_type

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    compressor = cls(
        model_name_or_path=args.model,
        adapter_dir=args.adapter_dir,
        prune_ratio=args.prune_ratio,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    for action in args.actions:
        if action == "calib":
            compressor.calib(
                calibration_dataset=args.calibration_dataset,
                max_calib_samples=args.max_calib_samples,
                batch_size=args.batch_size,
            )
            print(f"校准完成，adapter 已保存至: {compressor._get_adapter_path()}")

        elif action == "patch":
            compressor.patch()
            print("剪枝补丁已应用至 self.model")
            if args.output_model_path:
                compressor.model.save_pretrained(args.output_model_path)
                compressor.tokenizer.save_pretrained(args.output_model_path)
                print(f"剪枝模型已保存至: {args.output_model_path}")

        elif action == "eval":
            results = compressor.eval(
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.eval_batch_size,
                limit=args.limit,
            )
            print(results)


if __name__ == "__main__":
    main()
