from copy import deepcopy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from prefetch.training import runtime


def run_config(root, mode="same_token", stage="router"):
    config = dict(output_dir=str(root), stage=stage, training={"learning_rate": 0.001})
    if stage == "router":
        config["prefetch"] = {"mode": mode, "distance": 1}
    return config


@pytest.mark.parametrize("mode,stage,prefix", [
    ("same_token", "router", "same_token"),
    ("previous_token", "router", "previous_token"),
    (None, "lora", "lora"),
])
def test_run_name_uses_mode_and_startup_time(tmp_path, monkeypatch, mode, stage, prefix):
    monkeypatch.setattr(runtime, "datetime", SimpleNamespace(now=lambda: datetime(2026, 9, 23, 14, 5, 30, 123456)))
    config = run_config(tmp_path, mode, stage)
    output = runtime.prepare_run_directory(config)
    assert output == tmp_path / f"{prefix}_20260923_140530"
    assert output.is_dir()
    assert config["output_dir"] == str(output)


def test_repeated_launches_keep_existing_runs(tmp_path, monkeypatch):
    timestamps = iter([datetime(2026, 9, 23, 14, 5, 30), datetime(2026, 9, 23, 14, 5, 31)])
    monkeypatch.setattr(runtime, "datetime", SimpleNamespace(now=lambda: next(timestamps)))
    (tmp_path / "run_config.json").write_text("existing run", encoding="utf-8")
    first = runtime.prepare_run_directory(run_config(tmp_path))
    (first / "train.jsonl").write_text("existing log", encoding="utf-8")
    second = runtime.prepare_run_directory(run_config(tmp_path))
    assert first != second and first.is_dir() and second.is_dir()
    assert (tmp_path / "run_config.json").read_text(encoding="utf-8") == "existing run"
    assert (first / "train.jsonl").read_text(encoding="utf-8") == "existing log"


def test_same_second_collision_preserves_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "datetime", SimpleNamespace(now=lambda: datetime(2026, 9, 23, 14, 5, 30)))
    output = runtime.prepare_run_directory(run_config(tmp_path))
    (output / "train.jsonl").write_text("existing log", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runtime.prepare_run_directory(run_config(tmp_path))
    assert (output / "train.jsonl").read_text(encoding="utf-8") == "existing log"


@pytest.mark.parametrize("run_name", ["same_token_m1", "same_token_20260923_140530",
                                     "same_token_20260923_140530_123456"])
def test_resume_keeps_saved_directory(tmp_path, monkeypatch, run_name):
    original = run_config(tmp_path)
    saved = deepcopy(original)
    saved["output_dir"] = str(tmp_path / run_name)
    checkpoint = Path(saved["output_dir"]) / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(runtime, "datetime", SimpleNamespace(now=lambda: pytest.fail("Resume must keep its timestamp")))
    output = runtime.prepare_run_directory(original, checkpoint)
    assert output == checkpoint.parent
    assert original == saved
    assert list(tmp_path.iterdir()) == [output]


def test_resume_validates_training_config(tmp_path):
    config = run_config(tmp_path)
    saved = deepcopy(config)
    saved["output_dir"] = str(tmp_path / "same_token_20260923_140530")
    checkpoint = Path(saved["output_dir"]) / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
    config["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="same run config"):
        runtime.prepare_run_directory(config, checkpoint)


def test_legacy_loading_option_is_ignored_on_read_and_resume(tmp_path):
    config = run_config(tmp_path)
    config["model"] = dict(path="base", serial_load=True)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    current = runtime.read_config(path)
    assert current["model"] == {"path": "base"}
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    saved_path = checkpoint / "run_config.json"
    saved_path.write_text(json.dumps(config), encoding="utf-8")
    runtime.prepare_run_directory(current, checkpoint)
    assert current["model"] == {"path": "base"}
    assert json.loads(saved_path.read_text(encoding="utf-8"))["model"]["serial_load"] is True


def _run_directory_worker(process_rank, init_method, root):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=init_method, rank=process_rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        with patch.object(runtime, "datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 23, 14, 5, 30, 123456)
            if process_rank != 0:
                clock.now.side_effect = AssertionError("Only rank 0 selects the startup timestamp")
            config = run_config(root)
            output = runtime.prepare_run_directory(config)
        paths = [None, None]
        dist.all_gather_object(paths, str(output))
        assert paths[0] == paths[1] == config["output_dir"]
        assert output.is_dir()
        assert list(Path(root).iterdir()) == [output]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_DDP_TESTS") != "1", reason="Set RUN_DDP_TESTS=1 for the two-process test")
def test_two_ranks_share_run_directory(tmp_path):
    torch.multiprocessing.spawn(_run_directory_worker,
                                args=((tmp_path / "gloo_init").resolve().as_uri(), str(tmp_path / "outputs")),
                                nprocs=2, join=True)
