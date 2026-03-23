"""读取 adapter 目录下 calib 写入的 config.json（与 run.py 落盘格式一致）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("MoECompressor")


def load_adapter_dir_calib_config(adapter_dir: str | Path | None) -> dict[str, Any]:
    if adapter_dir is None:
        return {}
    p = Path(adapter_dir) / "config.json"
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 adapter config.json 失败 %s: %s", p, e)
        return {}


def saved_prune_ratio_from_adapter_dir(adapter_dir: str | Path | None) -> float | None:
    cfg = load_adapter_dir_calib_config(adapter_dir)
    ck = cfg.get("calib_kwargs")
    if isinstance(ck, dict):
        pr = ck.get("prune_ratio")
        if isinstance(pr, (int, float)):
            return float(pr)
    pr = cfg.get("prune_ratio")
    if isinstance(pr, (int, float)):
        return float(pr)
    return None
