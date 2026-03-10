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
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from methods.frequency_pruning.model_qwen3_moe import FrequencyPruningQwen3Moe

# 方法注册表：method_name -> (compressor_cls, default_model_type)
METHOD_REGISTRY = {
    "frequency_pruning": (FrequencyPruningQwen3Moe, "qwen3_moe"),
}


def _save_calib_config(args: argparse.Namespace, adapter_dir: Path) -> None:
    """将 calib 涉及的全部参数保存到 adapter_dir/config.json。"""
    config_path = adapter_dir / "config.json"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method,
        "mode": "calib",
        "model": args.model,
        "adapter_dir": str(adapter_dir),
        "model_type": args.model_type,
        "device": args.device,
        "dtype": args.dtype,
        "prune_ratio": args.prune_ratio,
        "calibration_dataset": args.calibration_dataset,
        "max_calib_samples": args.max_calib_samples,
        "max_context_len": args.max_context_len,
        "batch_size": args.batch_size,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logging.info("Calib config saved to: %s", config_path)


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
    parser.add_argument("--tasks", type=str, nargs="+", default=["wikitext"])
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output_base", type=str, default=None,
                        help="输出根目录，如 outputs/model_name。用于推导 eval 默认结果路径")
    parser.add_argument("--eval_output_path", type=str, default=None,
                        help="eval 结果保存路径。默认：剪枝 model 用 adapter_dir/results_{时间}.json，原 model 用 output_base/results_{时间}.json")
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.mode == "calib" and args.adapter_dir is None:
        raise ValueError("calib 模式需提供 --adapter_dir（用于保存 adapter）")

    logging.info("========== MoE-Compressors: %s mode ==========", args.mode.upper())
    cls, default_model_type = METHOD_REGISTRY[args.method]
    model_type = args.model_type or default_model_type

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    logging.info("Instantiating %s, model: %s", args.method, args.model)
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
        logging.info("Calibration done, adapter saved to: %s", compressor._get_adapter_path())

        # 保存 calib 涉及的全部参数到 method 目录下
        _save_calib_config(args, Path(args.adapter_dir))

    elif args.mode == "eval":
        results = compressor.eval(
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.eval_batch_size,
            limit=args.limit,
        )

        # 确定结果保存路径：显式指定 或 按规范默认
        # 剪枝模型: output_base/method/results_{时间}.json；原模型: output_base/results_{时间}.json
        eval_output = args.eval_output_path
        if eval_output is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.output_base:
                output_base = Path(args.output_base)
            else:
                model_name = args.model.replace("\\", "/").split("/")[-1]
                output_base = Path("./outputs") / model_name
            if args.adapter_dir:
                eval_output = str(Path(args.adapter_dir) / f"results_{ts}.json")
            else:
                eval_output = str(output_base / f"results_{ts}.json")

        try:
            from accelerate.state import PartialState
            is_main = PartialState().is_main_process
        except Exception:
            is_main = True

        if is_main:
            Path(eval_output).parent.mkdir(parents=True, exist_ok=True)
            obj = results.get("results", results) if isinstance(results, dict) else getattr(results, "results", results)
            obj = obj if obj is not None else {}
            with open(eval_output, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
            
            logging.info("Evaluation done, results saved to: %s", eval_output)
            logging.info("Evaluation results: %s", obj)
        


if __name__ == "__main__":
    main()
