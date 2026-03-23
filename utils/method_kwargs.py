"""run.py 共用的 JSON 解析工具。"""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(raw: str | None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        if default is not None:
            return dict(default)
        raise ValueError("JSON 字符串不能为空")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("必须是 JSON 对象 (dict)，而非数组或标量")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}") from e
