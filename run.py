#!/usr/bin/env python3
"""
MoE-Compressors 统一运行入口

【两种模式】
- calib：单卡校准，在校准集上计算统计量，保存 adapter.safetensors
- eval：多卡评测。若传入 adapter_dir 则先 patch（剪枝）再评测；不传则评测原模型

用法:
  python run.py frequency_pruning calib --model ... --adapter_dir ...
  python run.py frequency_pruning eval --model ...                    # 评测原模型
  python run.py frequency_pruning eval --model ... --adapter_dir ...  # patch 后评测剪枝模型
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
        description="MoE-Compressors: calib（校准）| eval（评测）"
    )
    parser.add_argument(
        "method",
        choices=list(METHOD_REGISTRY),
        help="压缩方法名称",
    )
    parser.add_argument(
        "mode",
        choices=["calib", "eval"],
        help="calib=单卡校准保存adapter；eval=多卡评测（adapter_dir非空时先patch再评测）",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="adapter目录。calib 时必填（保存路径）；eval 时可选，非空则 patch 后评测剪枝模型，空则评测原模型")
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--calibration_dataset", type=str, default="wikitext:wikitext-2-raw-v1")
    parser.add_argument("--max_calib_samples", type=int, default=512)
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_model_path", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=["wikitext"])
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=float, default=None)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.mode == "calib" and args.adapter_dir is None:
        raise ValueError("calib 模式需提供 --adapter_dir（用于保存 adapter）")

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

    if args.mode == "calib":
        compressor.calib(
            calibration_dataset=args.calibration_dataset,
            max_calib_samples=args.max_calib_samples,
            max_context_len=args.max_context_len,
            batch_size=args.batch_size,
        )
        print(f"校准完成，adapter 已保存至: {compressor._get_adapter_path()}")

    elif args.mode == "eval":
        # adapter_dir 非空：先 patch 再评测剪枝模型；否则直接评测原模型
        if args.adapter_dir is not None:
            compressor.patch()
            print("已应用剪枝补丁，开始评测剪枝模型")
        else:
            print("评测原模型（未传 adapter_dir）")

        if args.output_model_path:
            compressor.model.save_pretrained(args.output_model_path)
            compressor.tokenizer.save_pretrained(args.output_model_path)
            print(f"模型已保存至: {args.output_model_path}")

        results = compressor.eval(
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.eval_batch_size,
            limit=args.limit,
        )
        print(results)


if __name__ == "__main__":
    main()
