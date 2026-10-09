"""CLI compensation options reach the real predictor patch and artifact loader."""
import argparse
import importlib
import json
import sys

import pytest

torch = pytest.importorskip("torch")

from prefetch.compensation import OWA, ExFold
from prefetch.compensation.artifacts import save_artifact
from prefetch.compensation.cli import add_compensation_arguments, compensation_from_args
from prefetch.prerouter.checkpoint import save_predictor
from prefetch.prerouter.patch import patch, unpatch
from prefetch.tests.test_patch import ToyModel, config


def parse(arguments):
    parser = argparse.ArgumentParser()
    add_compensation_arguments(parser)
    return parser.parse_args(arguments)


def test_unspecified_compensation_preserves_patch_defaults():
    assert compensation_from_args(parse([]), checkpoint="unused") is None


def test_cli_merges_with_yaml_without_modifying_it(tmp_path):
    saved = {"method": "owa", "alpha1": 0.5, "alpha2": 0.6}
    (tmp_path / "prefetch_config.json").write_text(json.dumps({"config": {"compensation": saved}}))
    yaml = {"compensation": {"method": "owa", "path": "owa.safetensors", "alpha1": 2.0}}
    actual = compensation_from_args(parse(["--alpha2", "0.9", "--hit-min", "0"]), yaml, tmp_path)
    assert actual == {"method": "owa", "path": "owa.safetensors", "alpha1": 2.0,
                      "alpha2": 0.9, "hit_min": 0}
    assert "alpha2" not in yaml["compensation"]


def test_cli_inherits_checkpoint_compensation(tmp_path):
    saved = {"method": "exfold", "path": "saved.safetensors"}
    (tmp_path / "prefetch_config.json").write_text(json.dumps({"config": {"compensation": saved}}))
    actual = compensation_from_args(parse(["--path", "new.safetensors"]), {}, tmp_path)
    assert actual == {"method": "exfold", "path": "new.safetensors"}


@pytest.mark.parametrize("module_name", [
    "prefetch.examples.infer_prefetch_demo",
    "prefetch.evaluation.evaluate",
    "prefetch.evaluation.evaluate_base",
])
@pytest.mark.parametrize("method", ["exfold", "owa"])
def test_cli_loads_compensation_on_predictor_checkpoint(tmp_path, monkeypatch, module_name, method):
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config())
    checkpoint = tmp_path / "predictor"
    save_predictor(state, checkpoint)
    artifact = tmp_path / "exfold.safetensors"
    tensors = {f"{target}.{key}": value for target in state.targets
               for key, value in (("coefficients", torch.eye(8)), ("loss", torch.ones(8, 8)))}
    save_artifact(artifact, tensors, {"method": "exfold", "layers": state.layers})
    unpatch(model)

    # Each invocation selects a new method over the run's previous compensation.
    previous = ({"method": "owa", "alpha1": 2.0} if method == "exfold"
                else {"method": "exfold", "path": "previous-exfold.safetensors"})
    run_config = {"prefetch": {"compensation": previous}, "model": {}, "data": {}}
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "experiment_config", lambda *args: run_config)
    monkeypatch.setattr(module, "load_model", lambda *args: (model, None, {}))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    if hasattr(module, "setup"):
        monkeypatch.setattr(module, "setup", lambda *args: "cpu")

    class Patched(Exception):
        pass

    def install(*args, **kwargs):
        state = patch(*args, **kwargs)
        assert state.config.execution == "compensated"
        if method == "exfold":
            assert state.config.compensation == {"method": "exfold", "path": str(artifact)}
            assert all(isinstance(module, ExFold) for module in state.compensators.values())
        else:
            assert state.config.compensation == {"method": "owa", "alpha1": 2.0, "alpha2": 0.8,
                                                 "hit_min": 1, "hit_max": 1}
            assert all(isinstance(module, OWA) and module.alpha1 == 2.0 and module.alpha2 == 0.8
                       for module in state.compensators.values())
        raise Patched

    monkeypatch.setattr(module, "patch", install)
    argv = [module_name, "--checkpoint", str(checkpoint), "--output", "unused",
            "--execution-mode", "compensated", "--method", method]
    if module_name.endswith("evaluate_base"):
        argv += ["--sample-file", "unused"]
    argv += (["--path", str(artifact)] if method == "exfold"
             else ["--alpha1", "2", "--alpha2", "0.8", "--hit-min", "1", "--hit-max", "1"])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Patched):
        module.main()
