import json

import pytest

from prefetch.prerouter.configuration import layer_pairs
from prefetch.evaluation.metrics import histogram_curve, report_histograms, save_report
from prefetch.datasets.prepare import cog_records, main, validation_group, write_splits


def test_layer_mapping():
    assert len(layer_pairs(35, distance=1)) == 34
    assert layer_pairs(35, distance=2)[0] == (0, 2)
    assert layer_pairs(35, "previous_token")[-1] == (33, 34)
    assert layer_pairs(35, "previous_token", distance=0) == [(i, i) for i in range(35)]
    assert layer_pairs(35, "previous_token", distance=0, targets=[0, 3]) == [(0, 0), (3, 3)]
    assert layer_pairs(35, "previous_token", distance=2)[0] == (0, 2)
    assert layer_pairs(35, distance=2, targets=[3, 5]) == [(1, 3), (3, 5)]


@pytest.mark.parametrize("kwargs", [{"distance": 0}, {"mode": "previous_token", "distance": -1},
                                    {"targets": [0]}, {"targets": []}, {"targets": [2, 2]},
                                    {"mode": "previous_token", "distance": 0, "targets": [-1]},
                                    {"mode": "previous_token", "distance": 0, "targets": [35]}])
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


@pytest.fixture
def cog_root(tmp_path):
    for folder in ("caption", "dialogue"):
        for language in ("en", "zh"):
            (tmp_path / folder / f"labels_{language}").mkdir(parents=True)
        (tmp_path / folder / "images").mkdir()
        (tmp_path / folder / "images" / "1.jpg").write_bytes(b"same-image-bytes")
    for language, captions, question in (
        ("en", ["A cat.", "A kitten."], "What is here?"),
        ("zh", ["一只猫。", "一只小猫。"], "这里有什么？"),
    ):
        (tmp_path / f"caption/labels_{language}/1.json").write_text(json.dumps({"captions": [
            {"role": "caption", "content": content} for content in captions]}), encoding="utf-8")
        (tmp_path / f"dialogue/labels_{language}/1.json").write_text(json.dumps({"conversations": [
            {"role": "user", "content": f"<image> {question}"},
            {"role": "assistant", "content": captions[0]}]}), encoding="utf-8")
    return tmp_path


def test_cog_caption_and_conversation_normalization(cog_root):
    records = list(cog_records(cog_root, "Describe."))
    assert len(records) == len({r["id"] for r in records}) == 6
    assert len({r["group_id"] for r in records}) == 1
    assert records[0]["images"] == records[2]["images"]
    assert records[0]["messages"][0]["content"] == "Describe."
    assert records[2]["messages"][0]["content"] == "请详细描述这张图片。"
    assert records[2]["messages"][1]["content"] == "一只猫。"
    assert records[-2]["messages"][0]["content"] == "What is here?"
    assert records[-1]["messages"][0]["content"] == "这里有什么？"
    counts = write_splits(records, cog_root / "split", 0.5, 42)
    assert sorted(counts.values()) == [0, 6]


@pytest.mark.parametrize("limit", [None, 0, 1, 2, 3, 4, 6, 10])
def test_cog_record_limit(cog_root, limit):
    records = list(cog_records(cog_root, "Describe."))
    assert list(cog_records(cog_root, "Describe.", limit)) == records[:limit]


def test_cog_limit_stops_before_reading_next_image(cog_root):
    (cog_root / "dialogue/images/1.jpg").unlink()
    assert len(list(cog_records(cog_root, "Describe.", limit=4))) == 4


def test_cog_languages_are_grouped_by_image(cog_root):
    (cog_root / "caption/images/2.jpg").write_bytes(b"another-image")
    for language in ("en", "zh"):
        directory = cog_root / f"caption/labels_{language}"
        (directory / "2.json").write_text((directory / "1.json").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    records = list(cog_records(cog_root, "Describe.", limit=4))
    assert [r["id"] for r in records] == [
        f"cog/caption/labels_{language}/1.json/{index}"
        for language in ("en", "zh") for index in (0, 1)]


def test_cog_plain_labels_directory(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "images/1.jpg").write_bytes(b"image")
    (tmp_path / "labels/1.json").write_text(json.dumps({"captions": [{"content": "A cat."}]}))
    records = list(cog_records(tmp_path, "Describe."))
    assert len(records) == 1
    assert records[0]["messages"][0]["content"] == "Describe."


@pytest.mark.parametrize("limit", [None, 0, 1, 2])
def test_cog_cli_limit(cog_root, monkeypatch, capsys, limit):
    output = cog_root / "output"
    argv = ["prepare", "cog", "--root", str(cog_root), "--output", str(output),
            "--caption-prompt-zh", "描述图片。"]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    monkeypatch.setattr("sys.argv", argv)
    main()
    progress = capsys.readouterr().err
    assert "Prepare cog" in progress
    assert "sample" in progress
    if limit:
        assert f"{limit}/{limit}" in progress
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    expected = 6 if limit is None else limit
    assert metadata["limit"] == limit
    assert metadata["train"] + metadata["validation"] == expected
    records = [json.loads(line) for split in ("train", "validation")
               for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == expected
    if limit is None:
        captions_zh = [r for r in records if "caption/labels_zh/" in r["id"]]
        assert len(captions_zh) == 2
        assert all(r["messages"][0]["content"] == "描述图片。" for r in captions_zh)


@pytest.mark.parametrize("total", [None, 2])
def test_write_splits_progress(tmp_path, capsys, total):
    records = (dict(id=f"example-{i}", group_id="same-image") for i in range(2))
    counts = write_splits(records, tmp_path / "split", 0.5, 42, total=total, desc="Prepare tulu")
    assert sum(counts.values()) == 2
    progress = capsys.readouterr().err
    assert "Prepare tulu" in progress
    assert "sample" in progress
    assert ("2/2" if total is not None else "2sample") in progress
