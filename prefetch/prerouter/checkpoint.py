"""Predictor weights use target-MoE keys; live modules belong to producer blocks."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path

from prefetch.prerouter.configuration import PrefetchConfig


def _weight_key(target, name):
    # Preserve existing net keys; the conditioning branch uses token_proj.* keys.
    return f"{target}.{name[4:] if name.startswith('net.') else name}"


def predictor_state_dict(state):
    return {_weight_key(target, name): value.detach().cpu().contiguous()
            for source, target in state.pairs
            for name, value in state.prerouters[source].state_dict().items()}


def save_predictor(state, directory):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(predictor_state_dict(state), str(directory / "predictor.safetensors"))
    metadata = dict(config=asdict(state.config), layers=state.layers,
                    module_ownership="source_moe", weight_key="target_moe")
    (directory / "prefetch_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_metadata(directory):
    metadata = json.loads((Path(directory) / "prefetch_config.json").read_text(encoding="utf-8"))
    # Load saved fields in the current schema; absent/null fields use its defaults.
    names = {field.name for field in fields(PrefetchConfig)}
    metadata["config"] = {key: value for key, value in metadata["config"].items()
                          if key in names and value is not None}
    return metadata


def load_predictor(state, directory, metadata):
    from safetensors.torch import load_file
    if metadata["layers"] != state.layers:
        raise ValueError("Checkpoint source/target layer mapping differs from the loaded base")
    weights = load_file(str(Path(directory) / "predictor.safetensors"))
    expected = {_weight_key(target, name) for source, target in state.pairs
                for name in state.prerouters[source].state_dict()}
    if set(weights) != expected:
        raise ValueError("Checkpoint head keys differ from the configured predictors")
    for source, target in state.pairs:
        head = state.prerouters[source]
        head.load_state_dict({name: weights[_weight_key(target, name)] for name in head.state_dict()})
