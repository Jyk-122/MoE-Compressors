import json

import pytest

from prefetch.prerouter.configuration import layer_pairs
from prefetch.evaluation.metrics import histogram_curve, report_histograms, save_report
from prefetch.datasets.prepare import cog_records, validation_group, write_splits


def test_layer_mapping():
    assert len(layer_pairs(35, distance=1)) == 34
    assert layer_pairs(35, distance=2)[0] == (0, 2)
    assert layer_pairs(35, "previous_token")[-1] == (33, 34)
    assert layer_pairs(35, distance=2, targets=[3, 5]) == [(1, 3), (3, 5)]


@pytest.mark.parametrize("kwargs", [{"distance": 0}, {"mode": "previous_token", "distance": 2},
                                    {"targets": [0]}, {"targets": []}, {"targets": [2, 2]}])
def test_invalid_layer_mapping(kwargs):
    with pytest.raises(ValueError):
        layer_pairs(35, **kwargs)


def test_histograms_and_denominators(tmp_path):
    # Two tokens, K=2, true expert ranks [1,3] and [2,4].
    ranks, maximum = [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]
    rows = histogram_curve(ranks, maximum, [1, 2, 3, 4])
    assert [r["recall"] for r in rows] == [0.25, 0.5, 0.75, 1]
    assert [r["full_coverage"] for r in rows] == [0, 0, 0.5, 1]
    report = report_histograms({"teacher_forcing": [[ranks, maximum]]},
                               {"layers": [{"target_moe": 1}]}, [2, 4])
    save_report(report, tmp_path / "metrics.json")
    assert (tmp_path / "metrics.csv").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["phases"]["teacher_forcing"]["global"][-1]["recall"] == 1


def test_empty_metrics_are_explicit():
    row = histogram_curve([0] * 5, [0] * 5, [4])[0]
    assert row["recall"] is None and row["full_coverage"] is None
    assert row["token_layer_pairs"] == 0


def test_split_is_stable_and_grouped(tmp_path):
    assert validation_group("same-image", 0.5, 42) == validation_group("same-image", 0.5, 42)
    records = [dict(id=f"example-{i}", group_id="same-image") for i in range(2)]
    counts = write_splits(records, tmp_path / "split", 0.5, 42)
    assert sorted(counts.values()) == [0, 2]
    with pytest.raises(FileExistsError):
        write_splits(records, tmp_path / "split", 0.5, 42)


def test_cog_caption_and_conversation_normalization(tmp_path):
    for folder in ("caption", "dialogue"):
        (tmp_path / folder / "labels").mkdir(parents=True)
        (tmp_path / folder / "images").mkdir()
        (tmp_path / folder / "images" / "1.jpg").write_bytes(b"same-image-bytes")
    (tmp_path / "caption/labels/1.json").write_text(json.dumps({"captions": [
        {"role": "caption", "content": "A cat."}, {"role": "caption", "content": "A kitten."}]}))
    (tmp_path / "dialogue/labels/1.json").write_text(json.dumps({"conversations": [
        {"role": "user", "content": "<image> What is here?"}, {"role": "assistant", "content": "A cat."}]}))
    records = list(cog_records(tmp_path, "Describe."))
    assert len(records) == 3
    assert len({r["group_id"] for r in records}) == 1
    assert records[-1]["messages"][0]["content"] == "What is here?"
