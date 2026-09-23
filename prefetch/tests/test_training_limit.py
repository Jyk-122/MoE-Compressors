"""Training subsets are shared across ranks and recorded in the run config."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from prefetch.datasets import dataset
from prefetch.training import train


class Records(list):
    def select(self, indices):
        return Records(self[index] for index in indices)


@pytest.mark.parametrize("limit", [None, 1, 5, 99])
@pytest.mark.parametrize("mixed", [False, True])
def test_limit_applies_after_source_limits_and_mixing(monkeypatch, limit, mixed):
    parts = {"vl": Records(["v0", "v1"]), "text": Records(["t0", "t1", "t2"])}
    load = Mock(side_effect=lambda path, source_limit: parts[path])
    monkeypatch.setattr(dataset, "load_records", load)
    mixed_records = Records(["v0", "t0", "v1", "t1", "v0", "t2"])
    interleave = Mock(return_value=mixed_records)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(interleave_datasets=interleave))
    sources = [dict(path="vl", weight=1, limit=2)]
    if mixed:
        sources.append(dict(path="text", weight=3))
    config = dict(train=sources, seed=7, train_limit=limit)
    selected = dataset.training_records(config)
    expected = mixed_records if mixed else parts["vl"]
    assert selected == expected[:limit]
    load.assert_any_call("vl", 2)
    if mixed:
        load.assert_any_call("text", None)
        interleave.assert_called_once_with(list(parts.values()), probabilities=[0.25, 0.75],
                                           seed=7, stopping_strategy="all_exhausted")
    else:
        interleave.assert_not_called()

    # The shared subset is sharded afterward; it is not replicated N times per rank.
    if len(selected) >= 2:
        from torch.utils.data import DistributedSampler
        shards = [list(DistributedSampler(selected, num_replicas=2, rank=rank,
                                          shuffle=False, drop_last=True)) for rank in range(2)]
        assert len(shards[0]) == len(shards[1]) == len(selected) // 2
        assert not set(shards[0]) & set(shards[1])


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_training_limit(limit):
    with pytest.raises(ValueError, match="train_limit must be positive"):
        dataset.training_records(dict(train=[], train_limit=limit))


@pytest.mark.parametrize("yaml_limit,cli_limit,expected", [(None, None, None), (8, None, 8), (8, 4, 4)])
def test_cli_limit_is_in_config_before_run_creation(monkeypatch, yaml_limit, cli_limit, expected):
    config = dict(data=dict(train_limit=yaml_limit, validation={"text": {"limit": 6}}))
    monkeypatch.setattr(train, "read_config", lambda path: config)
    monkeypatch.setattr(train, "setup", lambda seed: torch.device("cpu"))
    monkeypatch.setattr(train, "world_size", lambda: 2)

    class ConfigChecked(Exception):
        pass

    def check_config(actual, resume):
        assert actual["data"]["train_limit"] == expected
        assert actual["data"]["validation"]["text"]["limit"] == 6
        raise ConfigChecked

    monkeypatch.setattr(train, "prepare_run_directory", check_config)
    argv = ["train", "--config", "unused.yaml"]
    if cli_limit is not None:
        argv += ["--limit", str(cli_limit)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ConfigChecked):
        train.main()


@pytest.mark.parametrize("limit", [0, -1, 1])
def test_cli_rejects_invalid_or_too_small_limit_before_model_loading(monkeypatch, capsys, limit):
    monkeypatch.setattr(train, "read_config", lambda path: dict(data={}))
    monkeypatch.setattr(train, "setup", lambda seed: torch.device("cpu"))
    monkeypatch.setattr(train, "world_size", lambda: 2)
    monkeypatch.setattr(train, "load_model", lambda *args: pytest.fail("Must validate limit before loading weights"))
    monkeypatch.setattr(sys, "argv", ["train", "--config", "unused.yaml", "--limit", str(limit)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert "Training limit" in capsys.readouterr().err
