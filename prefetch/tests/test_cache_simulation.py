"""Exact cache accounting, optimal demand-only baseline, and offline reports."""
from functools import lru_cache
from itertools import combinations, product
import json
import random
import sys

import pytest

from prefetch.evaluation.cache.oracle import simulate_layer
from prefetch.evaluation.cache.simulate import simulate_files


def totals(events):
    return {key: sum(row[key] for row in events) for key in events[0] if key != "token"}


def test_perfect_predictions_move_reads_before_demand():
    truth = [[0, 1], [1, 2], [0, 2]]
    events = list(simulate_layer(truth, truth, 2, 4))
    assert [row["baseline_loads"] for row in events] == [2, 1, 1]
    assert [row["prefetch_loads"] for row in events] == [2, 1, 1]
    assert all(row["demand_loads"] == 0 for row in events)
    assert events[0]["baseline_evictions"] == events[0]["prefetch_evictions"] == 0


def test_wrong_prefetch_can_evict_cached_truth():
    events = list(simulate_layer([[0, 1], [0, 1]], [[0, 1], [2, 3]], 2, 4))
    second = events[1]
    assert second["baseline_loads"] == 0
    assert second["prefetch_loads"] == second["demand_loads"] == 2
    assert second["prefetch_evictions"] == second["demand_evictions"] == 2


def test_cache_can_supply_experts_missing_from_prediction():
    events = list(simulate_layer([[0, 1], [0, 1]], [[0, 1], [2, 3]], 4, 4))
    assert events[1]["prefetch_loads"] == 2
    assert events[1]["demand_loads"] == 0


def test_wrong_for_current_token_can_be_useful_later():
    events = list(simulate_layer([[0], [1], [2]], [[2], [1], [2]], 2, 3))
    counts = totals(events)
    assert counts["baseline_loads"] == 3
    assert counts["prefetch_loads"] == 2
    assert counts["demand_loads"] == 1


def test_no_head_uses_native_demand_only():
    events = list(simulate_layer([[0, 1], [2, 3], [0, 2]], None, 3, 4))
    assert all(row["prefetch_loads"] == row["prefetch_evictions"] == 0 for row in events)
    assert all(row["baseline_loads"] == row["demand_loads"] for row in events)
    assert all(row["baseline_evictions"] == row["demand_evictions"] for row in events)


def test_belady_matches_exhaustive_optimum_for_set_requests():
    requests = list(combinations(range(4), 2))
    for truth in product(requests, repeat=3):
        for capacity in (2, 3, 4):
            @lru_cache(None)
            def optimal(token, resident):
                if token == len(truth):
                    return 0
                required, cache = set(truth[token]), set(resident)
                available = cache | required
                keep = min(capacity, len(available)) - len(required)
                return len(required - cache) + min(
                    optimal(token + 1, tuple(sorted(required | set(extra))))
                    for extra in combinations(available - required, keep))
            events = list(simulate_layer(truth, None, capacity, 4))
            assert sum(row["baseline_loads"] for row in events) == optimal(0, ())


def test_prefetch_total_never_beats_optimal_demand_reads():
    rng = random.Random(23)
    for _ in range(200):
        truth = [rng.sample(range(6), 2) for _ in range(12)]
        prediction = [rng.sample(range(6), 2) for _ in truth]
        for capacity in (2, 3, 4, 6):
            counts = totals(list(simulate_layer(truth, prediction, capacity, 6)))
            assert counts["prefetch_loads"] + counts["demand_loads"] >= counts["baseline_loads"]


@pytest.mark.parametrize("truth,prediction,capacity", [
    ([[0, 1]], [[0, 1]], 1), ([[0, 0]], None, 2), ([[0, 4]], None, 2),
    ([[0, 1]], [], 2), ([[0, 1]], [[0, 0]], 2), ([[0, 1]], None, 0),
])
def test_invalid_cache_requests(truth, prediction, capacity):
    with pytest.raises(ValueError):
        list(simulate_layer(truth, prediction, capacity, 4))


def metadata(source="text", requests=1):
    return dict(type="metadata", format="moe-cache-trace-v1", mode="same_token", phase="decode",
                source=source, prediction_k=2, expected_requests=requests,
                layers=[dict(moe=0, name="layers.0", experts=4, top_k=2, source_moe=None),
                        dict(moe=1, name="layers.1", experts=4, top_k=2, source_moe=0)])


def request(index=0):
    return dict(type="request", id=f"sample-{index}", sample_index=index, decode_tokens=2,
                layers=[dict(moe=0, truth=[[0, 1], [2, 3]], prediction=None),
                        dict(moe=1, truth=[[0, 1], [0, 2]], prediction=[[0, 1], [2, 3]])])


def write_trace(path, records=None, header=None, complete=True):
    records = [request()] if records is None else records
    header = metadata(requests=len(records)) if header is None else header
    lines = [header, *records]
    if complete:
        lines.append(dict(type="complete", requests=len(records)))
    path.write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")
    return path


def test_report_rates_cold_start_and_independent_requests(tmp_path):
    path = write_trace(tmp_path / "trace.jsonl", [request(0), request(1)])
    report = simulate_files([path], [2, 4])
    source = report["sources"]["text"]
    assert source["requests"] == 2 and source["decode_tokens"] == 4
    row = source["global_curve"][0]
    assert row["baseline_loads"] == 14
    assert row["prefetch_loads"] == 8
    assert row["demand_loads"] == 10
    assert row["total_loads_per_token"] == 4.5  # Sum over layers, not the per-layer mean.
    assert row["first_token_total_loads_per_request"] == 4
    assert row["after_first_total_loads_per_token"] == 5
    assert row["extra_loads"] == 4
    assert row["io_amplification"] == 18 / 14
    assert row["demand_load_reduction"] == 1 - 10 / 14
    assert source["per_layer"]["1"][0]["baseline_loads_per_token"] == 1.5
    assert source["per_layer"]["0"][0]["prefetch_loads"] == 0
    filtered = simulate_files([path], [2], layers=[1])["sources"]["text"]
    assert filtered["global_curve"] == filtered["per_layer"]["1"]


def test_sources_shards_and_zero_decode_requests(tmp_path):
    first = write_trace(tmp_path / "text0.jsonl")
    second = write_trace(tmp_path / "text1.jsonl", [request(1)])
    empty = request()
    empty["decode_tokens"] = 0
    for row in empty["layers"]:
        row["truth"] = []
        if row["prediction"] is not None:
            row["prediction"] = []
    third = write_trace(tmp_path / "vl0.jsonl", [empty], metadata(source="vl"))
    sources = simulate_files([first, second, third], [2])["sources"]
    assert sources["text"]["requests"] == 2
    row = sources["vl"]["global_curve"][0]
    assert row["decode_tokens"] == row["total_loads"] == 0
    assert row["total_loads_per_token"] is row["io_amplification"] is None


@pytest.mark.parametrize("change", ["length", "missing_layer", "prediction", "mode", "footer", "count"])
def test_reject_incomplete_or_incompatible_traces(tmp_path, change):
    header, record = metadata(), request()
    if change == "length":
        record["layers"][1]["truth"].pop()
    elif change == "missing_layer":
        record["layers"].pop()
    elif change == "prediction":
        record["layers"][1]["prediction"] = None
    elif change == "mode":
        header["mode"] = "previous_token"
    elif change == "count":
        header["expected_requests"] = 2
    path = write_trace(tmp_path / "trace.jsonl", [record], header, complete=change != "footer")
    with pytest.raises(ValueError):
        simulate_files([path], [2])


def test_duplicate_requests_and_changed_checkpoint_are_rejected(tmp_path):
    first = write_trace(tmp_path / "first.jsonl")
    second = write_trace(tmp_path / "second.jsonl")
    with pytest.raises(ValueError, match="Duplicate"):
        simulate_files([first, second], [2])
    other = metadata()
    other["checkpoint"] = "different"
    write_trace(second, [request(1)], other)
    with pytest.raises(ValueError, match="same model"):
        simulate_files([first, second], [2])


@pytest.mark.parametrize("capacities,layers", [([1], None), ([], None), ([2], [99]), ([2], [1, 1])])
def test_invalid_simulation_configuration(tmp_path, capacities, layers):
    path = write_trace(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError):
        simulate_files([path], capacities, layers)


def test_offline_cli_and_plots(tmp_path, monkeypatch, capsys):
    from prefetch.evaluation.cache.simulate import main
    write_trace(tmp_path / "trace.jsonl")
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["simulate", "--traces", str(tmp_path / "*.jsonl"),
                                     "--capacities", "2", "4", "--output", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["sources"]["text"]["global_curve"][0]["total_loads"] == 9
    assert "prefetch_loads_per_token" in (output / "summary.csv").read_text()
    for name in ("text-global.png", "text-0.png", "text-1.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
