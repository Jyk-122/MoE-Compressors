import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.evaluation.evaluate_base import evaluate_base


@pytest.mark.parametrize("limit,expected_nll,expected_tokens", [(1, 1.0, 1), (2, 2.5, 4)])
def test_base_nll_is_token_weighted(tmp_path, monkeypatch, limit, expected_nll, expected_tokens):
    import prefetch.evaluation.evaluate_base as evaluation

    def collate(records):
        tokens = records[0]["tokens"]
        labels = torch.ones(1, tokens + 1, dtype=torch.long)
        labels[:, 0] = -100
        return dict(input_ids=torch.ones_like(labels), labels=labels,
                    router_mask=torch.ones_like(labels, dtype=torch.bool))

    class Base(nn.Module):
        def forward(self, input_ids, labels, use_cache):
            assert not use_cache and not torch.is_grad_enabled()
            assert not self.training
            return SimpleNamespace(loss=torch.tensor(float(input_ids.shape[1] - 1)))

    monkeypatch.setattr(evaluation, "make_collator", lambda processor, data: collate)
    path = tmp_path / "validation.jsonl"
    path.write_text('\n'.join(json.dumps({"tokens": n}) for n in [1, 3]), encoding="utf-8")
    report = evaluate_base(Base(), None, {}, path, "cpu", limit=limit)
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
