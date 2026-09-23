"""Remaining-time estimates use optimizer steps completed in the current launch."""

import pytest

pytest.importorskip("torch")

from prefetch.training.runtime import estimate_eta


@pytest.mark.parametrize("elapsed,completed,remaining,seconds,label", [
    (600, 100, 900, 5400, "01:30"),
    (59, 1, 1, 59, "00:01"),
    (60, 1, 1, 60, "00:01"),
    (60.1, 1, 1, 60.1, "00:02"),
    (3600, 1, 25, 90000, "25:00"),
    (600, 100, 0, 0, "00:00"),
    (600, 100, -1, 0, "00:00"),
])
def test_eta_format_and_boundaries(elapsed, completed, remaining, seconds, label):
    result = estimate_eta(elapsed, completed, remaining)
    assert result["eta_seconds"] == pytest.approx(seconds)
    assert result["eta"] == label


def test_eta_before_first_optimizer_step():
    assert estimate_eta(0, 0, 100) == dict(eta_seconds=None, eta=None)


def test_resume_uses_new_steps_only():
    initial_step, step, max_steps = 500, 510, 600
    assert estimate_eta(120, step - initial_step, max_steps - step) == dict(
        eta_seconds=1080, eta="00:18")
