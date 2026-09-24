import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.evaluation.evaluate_base import evaluate_base
from prefetch.prerouter.configuration import PrefetchConfig


@pytest.mark.parametrize("limit,expected_nll,expected_tokens", [(1, 1.0, 1), (2, 2.5, 4)])
@pytest.mark.parametrize("patched", [False, True])
def test_base_nll_is_token_weighted(tmp_path, monkeypatch, limit, expected_nll, expected_tokens, patched):
    import prefetch.evaluation.evaluate_base as evaluation

    def collate(records):
        tokens = records[0]["tokens"]
        labels = torch.ones(1, tokens + 1, dtype=torch.long)
        labels[:, 0] = -100
        return dict(input_ids=torch.ones_like(labels), labels=labels,
                    router_mask=labels != -100)

    class Base(nn.Module):
        def forward(self, input_ids, labels, use_cache):
            assert not use_cache and not torch.is_grad_enabled()
            assert not self.training
            return SimpleNamespace(loss=torch.tensor(float(input_ids.shape[1] - 1)))

    monkeypatch.setattr(evaluation, "make_collator", lambda processor, data: collate)
    path = tmp_path / "validation.jsonl"
    path.write_text('\n'.join(json.dumps({"tokens": n}) for n in [1, 3]), encoding="utf-8")
    model, masks = Base(), []
    if patched:
        model.prerouter_state = SimpleNamespace(reset=lambda router_mask: masks.append(router_mask.clone()))
    report = evaluate_base(model, None, {}, path, "cpu", limit=limit)
    if patched:
        assert len(masks) == limit
        assert all(not mask[0, 0] and mask[0, 1:].all() for mask in masks)
    assert report["examples"] == limit
    assert report["tokens"] == expected_tokens
    assert report["assistant_nll"] == expected_nll
    assert report["assistant_perplexity"] == pytest.approx(math.exp(expected_nll))


def test_base_evaluation_rejects_empty_data(tmp_path, monkeypatch):
    import prefetch.evaluation.evaluate_base as evaluation
    monkeypatch.setattr(evaluation, "make_collator", lambda processor, data: None)
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No supervised"):
        evaluate_base(nn.Identity(), None, {}, path, "cpu")


def test_predictor_evaluation_checks_nf4_blocksize(tmp_path, monkeypatch):
    import prefetch.evaluation.evaluate as evaluation
    saved = {"model": {"path": "base", "quantization": "experts_nf4"}}
    (tmp_path / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
    requested = {"model": dict(saved["model"], nf4_blocksize=64)}
    monkeypatch.setattr(evaluation, "read_config", lambda path: requested)
    assert evaluation.experiment_config("config.yaml", tmp_path)["model"]["nf4_blocksize"] == 64
    requested["model"]["nf4_blocksize"] = 128
    with pytest.raises(ValueError, match="nf4_blocksize"):
        evaluation.experiment_config("config.yaml", tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_routing_nll_cli_preserves_adapter(tmp_path, monkeypatch, enabled):
    import sys
    import prefetch.evaluation.evaluate_base as evaluation
    config = dict(stage="router", model={"quantization": "none"}, data={},
                  lora={"enabled": True, "checkpoint": "adapter"}, prefetch={})
    monkeypatch.setattr(evaluation, "experiment_config", lambda *args: config)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    def load(config, device):
        assert config["stage"] == "router"
        assert config["lora"] == {"enabled": True, "checkpoint": "adapter"}
        return None, None, {}
    monkeypatch.setattr(evaluation, "load_model", load)
    def install(model, config, checkpoint, prerouter_enabled):
        assert checkpoint == "predictor" and prerouter_enabled is enabled
        return SimpleNamespace(config=PrefetchConfig(prerouter_enabled=enabled))
    monkeypatch.setattr(evaluation, "patch", install)
    monkeypatch.setattr(evaluation, "evaluate_base", lambda *args: {"assistant_nll": 1.0})
    output = tmp_path / "nll.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_base", "--checkpoint", "predictor",
                                     "--sample-file", "samples", "--output", str(output),
                                     "--prerouter-enabled" if enabled else "--no-prerouter-enabled"])
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["prefetch"]["prerouter_enabled"] is enabled


def test_routing_nll_requires_trained_checkpoint(monkeypatch):
    import sys
    import prefetch.evaluation.evaluate_base as evaluation
    monkeypatch.setattr(sys, "argv", ["evaluate_base", "--config", "unused", "--prerouter-enabled",
                                     "--sample-file", "unused", "--output", "unused"])
    with pytest.raises(SystemExit) as result:
        evaluation.main()
    assert result.value.code == 2
