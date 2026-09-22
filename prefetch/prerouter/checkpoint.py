"""Predictor weights use target-MoE keys; live modules belong to producer blocks."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path


def predictor_state_dict(state):
    return {f"{target}.{name}": value.detach().cpu().contiguous()
            for source, target in state.pairs
            for name, value in state.prerouters[source].net.state_dict().items()}


def save_predictor(state, directory):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(predictor_state_dict(state), str(directory / "predictor.safetensors"))
    metadata = dict(config=asdict(state.config), layers=state.layers,
                    module_ownership="source_moe", weight_key="target_moe")
    (directory / "prefetch_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_metadata(directory):
    return json.loads((Path(directory) / "prefetch_config.json").read_text(encoding="utf-8"))


def load_predictor(state, directory, metadata):
    from safetensors.torch import load_file
    if metadata["layers"] != state.layers:
        raise ValueError("Checkpoint source/target layer mapping differs from the loaded base")
    weights = load_file(str(Path(directory) / "predictor.safetensors"))
    expected = {f"{target}.{name}" for source, target in state.pairs
                for name in state.prerouters[source].net.state_dict()}
    if set(weights) != expected:
        raise ValueError("Checkpoint head keys differ from the configured predictors")
    for source, target in state.pairs:
        prefix = f"{target}."
        state.prerouters[source].net.load_state_dict(
            {name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)})
