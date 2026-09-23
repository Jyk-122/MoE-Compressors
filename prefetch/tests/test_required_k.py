"""RequiredK reporting and offline plots run without model weights or PyTorch."""
import csv
import json
import sys

import pytest

from prefetch.evaluation.metrics import (histogram_curve, rebuild_report, report_histograms,
                                        required_k_statistics, save_report)


def make_report():
    # Layer 1: true ranks [1,3], [2,4]. Layer 2: true ranks [1,2].
    layers = [[[0, 1, 1, 1, 1], [0, 0, 0, 1, 1]],
              [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]]]
    empty = [[[0] * 5, [0] * 5] for _ in layers]
    return report_histograms({"teacher_forcing": layers, "decode": empty},
                             {"layers": [{"target_moe": 1}, {"target_moe": 2}]}, [2, 4])


def test_required_k_summary_and_distribution():
    stats = required_k_statistics([0, 0, 1, 0, 2, 0, 0, 0, 1])  # [2,4,4,8]
    assert stats["token_layer_pairs"] == 4
    assert stats["mean"] == 4.5
    assert stats["median"] == 4
    assert (stats["min"], stats["max"]) == (2, 8)
    assert stats["p90"] == stats["p95"] == stats["p99"] == 8
    assert [row["k"] for row in stats["distribution"]] == list(range(1, 9))
    assert [row["cdf"] for row in stats["distribution"]] == [0, .25, .25, .75, .75, .75, .75, 1]
    assert sum(row["probability"] for row in stats["distribution"]) == 1


def test_median_and_integer_budget_percentiles():
    assert required_k_statistics([0, 0, 1, 0, 0, 1])["median"] == 3.5
    assert required_k_statistics([0, 0, 1, 0, 2])["median"] == 4
    stats = required_k_statistics([0, 0, 90, 5, 4, 1])
    assert (stats["p90"], stats["p95"], stats["p99"]) == (2, 3, 4)
    assert stats["max"] == 5


def test_empty_required_k():
    stats = required_k_statistics([0] * 5)
    assert stats["token_layer_pairs"] == 0
    assert all(stats[key] is None for key in ("mean", "median", "min", "max", "p90", "p95", "p99"))
    assert all(row["count"] == 0 and row["probability"] is None and row["cdf"] is None
               for row in stats["distribution"])


def test_recall_hits_and_required_k_cdf_agree():
    ranks, maximum = [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]
    curve = histogram_curve(ranks, maximum, range(1, 5))
    distribution = required_k_statistics(maximum)["distribution"]
    assert [row["mean_hits"] for row in curve] == [.5, 1, 1.5, 2]
    for row, bucket in zip(curve, distribution):
        assert row["mean_hits"] == row["recall"] * 2
        assert row["full_coverage"] == bucket["cdf"]
    assert histogram_curve([0] * 5, [0] * 5, [4])[0]["mean_hits"] is None


def test_global_statistics_weight_calls_not_layers():
    report = make_report()
    stats = report["phases"]["teacher_forcing"]["required_k"]
    assert stats["global"]["mean"] == 3  # [3,4,2], not the layer-mean average 2.75.
    assert stats["global"]["median"] == 3
    assert stats["global"]["token_layer_pairs"] == 3
    assert stats["per_layer"]["1"]["mean"] == 3.5
    assert stats["per_layer"]["2"]["mean"] == 2


def test_full_expert_range_with_sparse_candidate_curve(tmp_path):
    from prefetch.evaluation.plot import plot_report
    ranks, maximum = [0] * 385, [0] * 385
    for last_rank in (8, 23, 384):
        for rank in [1, 2, 3, 4, 5, 6, 7, last_rank]:
            ranks[rank] += 1
        maximum[last_rank] += 1
    report = report_histograms({"prefill": [[ranks, maximum]], "decode": [[ranks, maximum]]},
                             {"layers": [{"target_moe": 1}]}, [8, 16, 384])
    phase = report["phases"]["decode"]
    stats = phase["required_k"]["global"]
    assert stats["mean"] == pytest.approx((8 + 23 + 384) / 3)
    assert stats["median"] == 23  # Not one of the requested curve points.
    assert (stats["min"], stats["max"], stats["p99"]) == (8, 384, 384)
    assert len(stats["distribution"]) == 384
    assert stats["distribution"][22]["cdf"] == 2 / 3
    assert phase["global"][-1]["mean_hits"] == 8
    assert plot_report(report, tmp_path / "experts384") == report


def test_rebuild_historical_report_and_select_layers():
    original = make_report()
    original["evaluation"] = {"step": 300}
    original["trace"] = [{"target_moe": 1}, {"target_moe": 2}]
    for phase in original["phases"].values():
        del phase["required_k"]
        for curve in [phase["global"], *phase["per_layer"].values()]:
            for row in curve:
                del row["mean_hits"]
    snapshot = json.dumps(original)
    rebuilt = rebuild_report(original, [2])
    assert json.dumps(original) == snapshot
    assert rebuilt["metadata"]["layers"] == [{"target_moe": 2}]
    assert rebuilt["evaluation"] == {"step": 300}
    assert rebuilt["trace"] == [{"target_moe": 2}]
    phase = rebuilt["phases"]["teacher_forcing"]
    assert phase["global"][0]["mean_hits"] == 2
    assert phase["required_k"]["global"]["mean"] == 2
    assert set(phase["required_k"]["per_layer"]) == {"2"}
    assert len(phase["required_k"]["global"]["distribution"]) == 4
    assert rebuild_report(original)["phases"]["teacher_forcing"]["required_k"]["global"]["mean"] == 3


@pytest.mark.parametrize("layers", [[99], [1, 99], [1, 1], []])
def test_invalid_layer_selection(layers):
    with pytest.raises(ValueError):
        rebuild_report(make_report(), layers)


def test_saved_json_and_csv(tmp_path):
    path = tmp_path / "metrics.json"
    save_report(make_report(), path)
    phase = json.loads(path.read_text())["phases"]["teacher_forcing"]
    assert phase["required_k"]["global"]["mean"] == 3
    with path.with_suffix(".csv").open(newline="") as file:
        assert "mean_hits" in next(csv.DictReader(file))
    with (tmp_path / "metrics-required_k.csv").open(newline="") as file:
        summaries = list(csv.DictReader(file))
    assert len(summaries) == 6  # Two phases, global and two target layers.
    assert summaries[0]["mean"] == "3.0"
    with (tmp_path / "metrics-required_k_distribution.csv").open(newline="") as file:
        distribution = list(csv.DictReader(file))
    assert len(distribution) == 24
    assert distribution[3]["cdf"] == "1.0"


@pytest.mark.parametrize("custom_prefix", [False, True])
def test_offline_cli_writes_reports_and_four_plots(tmp_path, monkeypatch, capsys, custom_prefix):
    from prefetch.evaluation.plot import main
    path = tmp_path / "metrics.json"
    report = make_report()
    for phase in report["phases"].values():
        del phase["required_k"]
    original = json.dumps(report)
    path.write_text(original, encoding="utf-8")
    prefix = tmp_path / "figures" / "layer-2" if custom_prefix else tmp_path / "metrics-report"
    argv = ["plot", str(path)]
    if custom_prefix:
        argv += ["--output-prefix", str(prefix), "--layers", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert path.read_text(encoding="utf-8") == original
    summary = json.loads(capsys.readouterr().out)["required_k"]["teacher_forcing"]
    assert summary["mean"] == (2 if custom_prefix else 3)
    assert "distribution" not in summary
    assert json.loads(prefix.with_suffix(".json").read_text())["phases"]["teacher_forcing"]["required_k"]
    for suffix in (".csv", "-required_k.csv", "-required_k_distribution.csv"):
        assert (prefix.parent / (prefix.name + suffix)).is_file()
    for metric in ("recall", "mean_hits", "required_k_distribution", "required_k_cdf"):
        png = prefix.parent / (prefix.name + f"-{metric}.png")
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
