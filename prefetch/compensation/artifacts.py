"""Small safetensors artifacts, keyed by target MoE ordinal."""
from __future__ import annotations

import json
from pathlib import Path

from .exfold import ExFold
from .owa import OWA


def save_artifact(path, tensors, metadata):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()},
              str(path), metadata={"compensation": json.dumps(metadata, ensure_ascii=False)})


def load_artifact(path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["compensation"])
    return load_file(str(path)), metadata


def build_compensators(state):
    options = dict(state.config.compensation)
    method = options.pop("method", None)
    artifact = options.pop("path", None)
    tensors, metadata = load_artifact(artifact) if artifact else ({}, {})
    if artifact:
        if method is not None and method != metadata["method"]:
            raise ValueError("Compensation method differs from artifact")
        method = metadata["method"]
        stored = {row["target_moe"]: row for row in metadata["layers"]}
        for row in state.layers:
            saved = stored.get(row["target_moe"], {})
            if any(saved.get(key) != row[key] for key in ("target_name", "experts", "top_k")):
                raise ValueError("Compensation artifact target layers differ from model")
    if method == "owa":
        calibrated = metadata.get("parameters", {})
        if artifact:
            calibrated = dict(calibrated, alpha1=float(tensors["alpha"][0]),
                              alpha2=float(tensors["alpha"][1]))
        return {target: OWA(**dict(calibrated, **options)) for target in state.targets}
    if method == "exfold":
        if not artifact:
            raise ValueError("ExFold requires compensation.path to a calibrated safetensors file")
        if options:
            raise ValueError(f"Unknown ExFold options: {sorted(options)}")
        result = {}
        for target in state.targets:
            block = state.blocks[target][1]
            module = ExFold(tensors[f"{target}.coefficients"], tensors[f"{target}.loss"])
            if module.coefficients.shape[0] != block.gate.weight.shape[0]:
                raise ValueError("ExFold table size differs from target expert count")
            result[target] = module.to(block.gate.weight.device)
        return result
    raise ValueError("Compensated execution requires method 'owa' or 'exfold'")
