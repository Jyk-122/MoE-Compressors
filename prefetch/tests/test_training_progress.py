"""Training progress formatting and current-launch optimizer-step ETA estimates."""

import json

import pytest

pytest.importorskip("torch")

from prefetch.training.runtime import estimate_eta, format_duration, format_training_log


@pytest.mark.parametrize("seconds,label", [
    (0, "000:00:00"),
    (59.9, "000:00:59"),
    (60, "000:01:00"),
    (3599, "000:59:59"),
    (3600, "001:00:00"),
    (3661.9, "001:01:01"),
    (90000, "025:00:00"),
    (360000, "100:00:00"),
    (3600000, "1000:00:00"),
])
def test_duration_format(seconds, label):
    assert format_duration(seconds) == label


@pytest.mark.parametrize("elapsed,completed,remaining,seconds,label", [
    (600, 100, 900, 5400, "001:30:00"),
    (59, 1, 1, 59, "000:00:59"),
    (59.1, 1, 1, 59.1, "000:01:00"),
    (60, 1, 1, 60, "000:01:00"),
    (60.1, 1, 1, 60.1, "000:01:01"),
    (3599.1, 1, 1, 3599.1, "001:00:00"),
    (3600, 1, 25, 90000, "025:00:00"),
    (600, 100, 0, 0, "000:00:00"),
    (600, 100, -1, 0, "000:00:00"),
])
def test_eta_format_and_boundaries(elapsed, completed, remaining, seconds, label):
    result = estimate_eta(elapsed, completed, remaining)
    assert result["eta_seconds"] == pytest.approx(seconds)
    assert result["eta_time"] == label


def test_eta_before_first_optimizer_step():
    assert estimate_eta(0, 0, 100) == dict(eta_seconds=None, eta_time=None)


def test_resume_uses_new_steps_only():
    initial_step, step, max_steps = 500, 510, 600
    assert estimate_eta(120, step - initial_step, max_steps - step) == dict(
        eta_seconds=1080, eta_time="000:18:00")


@pytest.mark.parametrize("peak,lr,peak_label,lr_label", [
    (55.2, 0.0003, "55.20", "3.00e-04"),
    (55.256, 0.000123456, "55.26", "1.23e-04"),
    (0.0, 0.0, "0.00", "0.00e+00"),
])
def test_console_format_preserves_jsonl_values(peak, lr, peak_label, lr_label):
    entry = dict(step=10, epoch=0, loss=0.123456, examples=10, text_tokens=100,
                 elapsed_seconds=754.8, elapsed_time=format_duration(754.8),
                 peak_gpu_gib=peak, learning_rate=lr)
    entry.update(estimate_eta(754.8, 10, 100))
    saved = entry.copy()

    display = json.loads(format_training_log(entry))
    assert display["elapsed_time"] == "000:12:34"
    assert display["eta_time"] == "002:05:48"
    assert display["peak_gpu_gib"] == peak_label
    assert display["learning_rate"] == lr_label
    assert "elapsed_seconds" not in display
    assert "eta_seconds" not in display
    for key in ("step", "epoch", "loss", "examples", "text_tokens"):
        assert display[key] == entry[key]
    assert entry == saved
    stored = json.loads(json.dumps(entry))
    assert isinstance(stored["peak_gpu_gib"], float)
    assert isinstance(stored["learning_rate"], float)
    assert stored == saved
