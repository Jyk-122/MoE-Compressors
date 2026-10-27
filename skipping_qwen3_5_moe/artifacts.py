"""Portable expert moments and trained compensation states in safetensors format."""
from __future__ import annotations

import json
from pathlib import Path


FORMAT = "qwen_moe_compensation_v2"
FIELDS = {"calibration": ("mean", "std"), "trained": ("mean", "log_std", "logits")}


def save_artifact(states, path, *, threshold=0.5, base_model=""):
    from safetensors.torch import save_file

    if not states:
        raise ValueError("At least one layer is required")
    kinds = {"trained" if "logits" in state else "calibration" for state in states.values()}
    if len(kinds) != 1:
        raise ValueError("All layers must use the same compensation state kind")
    kind = kinds.pop()
    tensors, records = {}, {}
    for index, state in states.items():
        prefix = f"layers.{index}"
        for name in FIELDS[kind]:
            tensors[f"{prefix}.{name}"] = state[name].detach().cpu().contiguous().clone()
        records[str(index)] = {name: state[name] for name in ("samples", "epsilon", "top_k")}
    metadata = {"format": FORMAT, "kind": kind,
                "layers": records, "threshold": threshold, "base_model": str(base_model),
                "pair_axes": ["replacement", "replaced"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"compensation": json.dumps(metadata, ensure_ascii=False)})


def read_artifact(path):
    from safetensors import safe_open

    states = {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = json.loads((file.metadata() or {}).get("compensation", "{}"))
        if metadata.get("format") != FORMAT or metadata.get("kind") not in {"calibration", "trained"}:
            raise ValueError(f"Expected a {FORMAT} calibration or trained checkpoint")
        if metadata.get("pair_axes") != ["replacement", "replaced"]:
            raise ValueError("Checkpoint pair axes must be replacement, replaced")
        expected_keys = set()
        for index, record in metadata["layers"].items():
            prefix = f"layers.{index}"
            state = dict(record)
            for name in FIELDS[metadata["kind"]]:
                key = f"{prefix}.{name}"
                expected_keys.add(key)
                state[name] = file.get_tensor(key).clone()
            states[int(index)] = state
        if set(file.keys()) != expected_keys:
            raise ValueError("Checkpoint tensor names differ from its layer metadata")
    return states, metadata
