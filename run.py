#!/usr/bin/env python3
"""
MoE-Compressors 统一入口：剪枝与 skipping 共用 calib | eval。

方法相关超参通过 --calib_kwargs / --patch_kwargs（JSON）。
run.py 仅做 JSON 解析与透传：
- calib_kwargs -> calib(**kwargs)
- patch_kwargs -> patch(**kwargs)

示例:
  python run.py frequency_pruning calib --model ... --adapter_dir ... \\
    --calib_kwargs '{"prune_ratio":0.5}'
  accelerate launch run.py frequency_pruning eval --model ... --adapter_dir ... \\
    --patch_kwargs '{"prune_ratio":0.3}'
  accelerate launch run.py topk_skip eval --model ... --patch_kwargs '{"k":2}'
  accelerate launch run.py topp_skip eval --model ... --patch_kwargs '{"threshold":0.8}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from methods_pruning.camera_pruning.model_qwen3_moe import CAMERAPruningQwen3Moe
from methods_pruning.ean_pruning.model_qwen3_moe import EANPruningQwen3Moe
from methods_pruning.frequency_pruning.model_qwen3_moe import FrequencyPruningQwen3Moe
from methods_pruning.moei2_pruning.model_qwen3_moe import MoEI2PruningQwen3Moe
from methods_pruning.reap_pruning.model_qwen3_moe import REAPPruningQwen3Moe
from methods_skipping.topk_skip.model_qwen3_moe import TopKSkipQwen3Moe
from methods_skipping.topp_skip.model_qwen3_moe import TopPSkipQwen3Moe

from transformers import AutoConfig
from utils.method_kwargs import parse_json_object

METHOD_REGISTRY: dict[str, dict[str, type]] = {
    "frequency_pruning": {"qwen3_moe": FrequencyPruningQwen3Moe},
    "ean_pruning": {"qwen3_moe": EANPruningQwen3Moe},
    "reap_pruning": {"qwen3_moe": REAPPruningQwen3Moe},
    "camera_pruning": {"qwen3_moe": CAMERAPruningQwen3Moe},
    "moei2_pruning": {"qwen3_moe": MoEI2PruningQwen3Moe},
    "topk_skip": {"qwen3_moe": TopKSkipQwen3Moe},
    "topp_skip": {"qwen3_moe": TopPSkipQwen3Moe},
}


def infer_model_type(model_name_or_path: str, trust_remote_code: bool = True) -> str:
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return config.model_type


def save_calib_config(
    *,
    adapter_dir: Path,
    model_type: str,
    method: str,
    model: str,
    device: str,
    dtype: str,
    calibration_dataset: str,
    max_calib_samples: int,
    max_context_len: int,
    batch_size: int,
    calib_kwargs: dict[str, Any],
) -> None:
    """统一落盘：含 calib_kwargs（run.py 解析后的原始字典）。"""
    config_path = adapter_dir / "config.json"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": method,
        "mode": "calib",
        "model": model,
        "adapter_dir": str(adapter_dir),
        "model_type": model_type,
        "device": device,
        "dtype": dtype,
        "calibration_dataset": calibration_dataset,
        "max_calib_samples": max_calib_samples,
        "max_context_len": max_context_len,
        "batch_size": batch_size,
        "calib_kwargs": calib_kwargs,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logging.info("Calib config saved to: %s", config_path)


def default_eval_output_path(args: Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_base:
        output_base = Path(args.output_base)
    else:
        model_name = args.model.replace("\\", "/").split("/")[-1]
        output_base = Path("./outputs") / model_name
    if getattr(args, "adapter_dir", None):
        return str(Path(args.adapter_dir) / f"results_{ts}.json")
    return str(output_base / f"results_{ts}.json")


def write_eval_results_file(
    results: Any,
    eval_output: str,
    eval_output_content: str,
) -> None:
    try:
        from accelerate.state import PartialState

        is_main = PartialState().is_main_process
    except Exception:
        is_main = True

    if not is_main:
        return

    Path(eval_output).parent.mkdir(parents=True, exist_ok=True)
    obj = results.get("results", results) if isinstance(results, dict) else getattr(results, "results", results)
    obj = obj if obj is not None else {}
    if isinstance(results, dict) and "runtime_routing" in results and isinstance(obj, dict):
        obj = dict(obj)
        obj["runtime_routing"] = results["runtime_routing"]

    if eval_output_content == "metrics":
        to_dump = obj
    else:
        if isinstance(results, dict):
            to_dump = results
        else:
            to_dump = {}
            for attr in ("results", "config", "samples", "git_hash", "date"):
                v = getattr(results, attr, None)
                if v is not None:
                    to_dump[attr] = v
            if not to_dump:
                to_dump = {"results": obj}

    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(to_dump, f, ensure_ascii=False, indent=2, default=str)
    logging.info("Evaluation done, results saved to: %s (%s)", eval_output, eval_output_content)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE-Compressors: calib | eval（剪枝 + skipping）"
    )
    parser.add_argument(
        "method",
        choices=sorted(METHOD_REGISTRY.keys()),
        help="方法名（剪枝 *_pruning 或 skipping: topk_skip/topp_skip）",
    )
    parser.add_argument(
        "mode",
        choices=["calib", "eval"],
        help="calib=单卡校准；eval=评测（建议 accelerate launch）",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="adapter 目录：calib 必填；eval 可选",
    )
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--calib_kwargs",
        type=str,
        default="{}",
        help="校准阶段 JSON，原样传给 calib(**calib_kwargs)",
    )
    parser.add_argument(
        "--patch_kwargs",
        type=str,
        default="{}",
        help="评测阶段 JSON，原样传给 patch(**patch_kwargs)",
    )
    parser.add_argument("--calibration_dataset", type=str, default="wikitext:wikitext-2-raw-v1")
    parser.add_argument("--max_calib_samples", type=int, default=512)
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tasks", type=str, nargs="+", default=["wikitext"])
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--eval_output_path", type=str, default=None)
    parser.add_argument("--gen_kwargs", type=str, default=None)
    parser.add_argument(
        "--eval_output_content",
        type=str,
        default="metrics",
        choices=["metrics", "full"],
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.mode == "calib" and args.adapter_dir is None:
        raise ValueError("calib 需提供 --adapter_dir")

    logging.info("========== MoE-Compressors: %s | %s ==========", args.method, args.mode.upper())

    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = infer_model_type(args.model)
        logging.info("Inferred model_type='%s' from model config", model_type)

    reg = METHOD_REGISTRY[args.method]
    if model_type not in reg:
        raise ValueError(
            f"方法 '{args.method}' 未适配 model_type='{model_type}'，支持: {list(reg.keys())}"
        )
    cls = reg[model_type]

    import torch

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    if args.mode == "calib":
        calib_kwargs = parse_json_object(args.calib_kwargs, default={})
        logging.info("Instantiating %s, model: %s", args.method, args.model)
        compressor = cls(
            model_name_or_path=args.model,
            adapter_dir=args.adapter_dir,
            device=args.device,
            torch_dtype=torch_dtype,
        )
        compressor.calib(
            calibration_dataset=args.calibration_dataset,
            max_calib_samples=args.max_calib_samples,
            max_context_len=args.max_context_len,
            batch_size=args.batch_size,
            **calib_kwargs,
        )
        save_calib_config(
            adapter_dir=Path(args.adapter_dir),
            model_type=model_type,
            method=args.method,
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            calibration_dataset=args.calibration_dataset,
            max_calib_samples=args.max_calib_samples,
            max_context_len=args.max_context_len,
            batch_size=args.batch_size,
            calib_kwargs=calib_kwargs,
        )
        logging.info("Calibration done under: %s", args.adapter_dir)

    elif args.mode == "eval":
        patch_kwargs = parse_json_object(args.patch_kwargs, default={})
        logging.info("Instantiating %s, model: %s", args.method, args.model)
        compressor = cls(
            model_name_or_path=args.model,
            adapter_dir=args.adapter_dir,
            device=args.device,
            torch_dtype=torch_dtype,
        )
        results = compressor.eval(
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.eval_batch_size,
            limit=args.limit,
            gen_kwargs=args.gen_kwargs,
            patch_kwargs=patch_kwargs if patch_kwargs else None,
        )

        out = args.eval_output_path or default_eval_output_path(args)
        write_eval_results_file(results, out, args.eval_output_content)


if __name__ == "__main__":
    main()
