"""CLI imports and --help should work without loading weights or touching CUDA."""
import importlib
import sys

import pytest


@pytest.mark.parametrize("name", [
    "prefetch.datasets.prepare",
    "prefetch.backbone.export_nf4",
    "prefetch.training.train",
    "prefetch.evaluation.evaluate",
    "prefetch.evaluation.evaluate_base",
    "prefetch.evaluation.plot",
    "prefetch.examples.infer_prefetch_demo",
    "prefetch.examples.infer_base_demo",
    "prefetch.examples.smoke_test",
])
def test_cli_help(name, monkeypatch, capsys):
    if name not in {"prefetch.datasets.prepare", "prefetch.evaluation.plot"}:
        pytest.importorskip("torch")
    module = importlib.import_module(name)
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out
