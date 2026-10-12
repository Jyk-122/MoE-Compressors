"""Mixed calibration records preserve conversations, image pairs, and input masks."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from prefetch.datasets.dataset import SFTCollator
from prefetch.datasets.prepare_calibration import (
    SOURCES, chart_examples, cog_examples, conversation, export_mixture, instruction_records,
    interleave_sources, pack_instructions, record, source_streams, text_chunks,
)
from prefetch.tests.test_data import ToyProcessor


class CharTokenizer:
    all_special_ids = [1, 2, 3, 4]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


class Processor(ToyProcessor):
    tokenizer = CharTokenizer()

    def __init__(self):
        self.image_pixels = []

    def __call__(self, text, **kwargs):
        self.image_pixels.extend(image.getpixel((0, 0)) for image in kwargs.get("images", []))
        return super().__call__(text, **kwargs)


@pytest.mark.parametrize("column", ["text", "content"])
def test_raw_text_becomes_bounded_continuation_chunks(column):
    processor = Processor()
    rows = [{column: "alpha beta gamma delta " * 15}, {column: "more text " * 10}]
    result = list(text_chunks(rows, processor, max_length=128, prefix_tokens=12))
    assert len(result) > 2
    for item in result:
        assert not item["images"]
        assert item["messages"][0]["content"].startswith("Continue the following text:\n")
        assert item["messages"][1]["role"] == "assistant"
        batch = SFTCollator(processor, max_length=128)([item])
        assert batch["input_ids"].shape[1] <= 128
        assert batch["router_mask"].any()


def test_raw_text_column_selection_and_empty_rows(caplog):
    rows = [{"id": "metadata", "text": value} for value in (None, "", "  ", "actual body")]
    with caplog.at_level("INFO"):
        result = list(text_chunks(rows, Processor(), max_length=128, prefix_tokens=3))
    assert len(result) == 1
    assert result[0]["messages"] == conversation("Continue the following text:\nact", "ual body\n\n")
    assert "SlimPajama text column: text" in caplog.text


@pytest.mark.parametrize("source,suffix", [("codealpaca", ""), ("alpaca_zh", "_zh")])
def test_instruction_fields_and_complete_turn_packing(source, suffix):
    rows = [{f"instruction{suffix}": "solve", f"input{suffix}": "abc", f"output{suffix}": "done"}
            for _ in range(5)]
    normalized = list(instruction_records(rows, source))
    assert normalized[0]["messages"] == conversation("solve\nabc", "done")
    packed = list(pack_instructions(normalized, Processor(), max_length=42))
    assert len(packed) == 3
    assert sum(len(item["members"]) for item in packed) == len(rows)
    for item in packed:
        assert len(item["messages"]) % 2 == 0
        assert SFTCollator(Processor(), max_length=42)([item])["router_mask"].any()


def test_cog_english_annotation_matches_image(tmp_path):
    (tmp_path / "images").mkdir()
    for language in ("en", "zh"):
        (tmp_path / f"labels_{language}").mkdir()
        (tmp_path / f"labels_{language}" / "cat.json").write_text(
            json.dumps({"conversations": conversation("<image>\nwhat?", language + " cat")}), encoding="utf-8")
    Image.new("RGB", (5, 5), (12, 34, 56)).save(tmp_path / "images" / "cat.png")
    rows = list(cog_examples(tmp_path))
    assert len(rows) == 1
    assert rows[0]["messages"] == conversation("what?", "en cat")
    assert Image.open(rows[0]["images"][0]).getpixel((0, 0)) == (12, 34, 56)


def make_streams(image_path):
    return {
        "slimpajama": iter([record("slimpajama", 0, conversation("prefix", "continuation"))]),
        "codealpaca": iter([record("codealpaca", 0, conversation("code", "result"))]),
        "cogvlm": iter([record("cogvlm", i, conversation("describe", f"image {i}"), [str(image_path)])
                       for i in range(2)]),
        "chartqa": chart_examples([
            {"question": "long " * 100, "answer": "skip", "image": Image.new("RGB", (5, 5), "green")},
            {"question": "how many?", "answer": 42, "image": Image.new("RGB", (5, 5), "blue")},
        ], "test"),
        "alpaca_zh": iter([record("alpaca_zh", 0, conversation("question", "answer"))]),
    }


def test_export_image_pairs_can_be_read_directly_by_calibration(tmp_path):
    from prefetch.compensation.calibration import collect
    from prefetch.compensation.optimal_scale.calibrate import ScaleStats
    from prefetch.prerouter.patch import patch
    from prefetch.tests.test_patch import ToyModel, config

    source_image = tmp_path / "original.png"
    Image.new("RGB", (5, 5), "red").save(source_image)
    output = tmp_path / "calib" / "mixed.jsonl"
    processor = Processor()
    data = {"max_length": 128, "router_tokens": "assistant", "max_image_side": None}
    report = export_mixture(make_streams(source_image), [1, 1, 2, 1, 1],
                            processor, data, output, seed=42)
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["samples"] == 6 and report["image_samples"] == 3
    assert report["sources"]["chartqa"]["overlength"] == 1
    assert Counter(item["source"] for item in saved) == dict(zip(SOURCES, [1, 1, 2, 1, 1]))
    for item in saved:
        assert all(Path(path).is_absolute() and Path(path).parent == output.parent / "images"
                   for path in item["images"])
        assert item["assistant_text_tokens"] > 0
        if item["source"] == "chartqa":
            assert item["messages"][1]["content"] == "42"
            assert Image.open(item["images"][0]).getpixel((0, 0)) == (0, 0, 255)
        elif item["source"] == "cogvlm":
            assert Image.open(item["images"][0]).getpixel((0, 0)) == (255, 0, 0)
    assert output.with_suffix(".meta.json").exists()

    model = ToyModel()
    model.embed = torch.nn.Embedding(256, 6)
    model.eval().requires_grad_(False)
    state = patch(model, config(mode="previous_token", distance=0, targets=[3]))
    args = SimpleNamespace(chunk_size=2, max_tokens=1024, limit=6, sample_file=output)
    collector = ScaleStats(8, 6, channel_chunk_size=2)
    processor.image_pixels.clear()
    counts = collect(model, processor, state, {"model": {}, "data": data}, "cpu", args, collector)
    assert counts[3] == sum(item["assistant_text_tokens"] for item in saved)
    assert (255, 0, 0) in processor.image_pixels and (0, 0, 255) in processor.image_pixels
    assert not state.active and state.route_observer is None


def test_interleave_preserves_proportions_in_short_prefix_and_seed():
    sizes = [32, 32, 64, 32, 32]
    groups = {source: [record(source, index, conversation("q", "a")) for index in range(size)]
              for source, size in zip(SOURCES, sizes)}
    rows = interleave_sources(groups, 42)
    assert len(rows) == 192
    assert rows == interleave_sources(groups, 42)
    assert rows != interleave_sources(groups, 43)
    prefix = Counter(item["source"] for item in rows[:32])
    assert set(prefix) == set(SOURCES)
    assert all(abs(prefix[source] - 32 * size / 192) < 1 for source, size in zip(SOURCES, sizes))


def test_short_source_reports_actual_valid_count(tmp_path):
    streams = {source: iter([]) for source in SOURCES}
    streams["slimpajama"] = iter([record("slimpajama", 0, conversation("q", "a"))])
    with pytest.raises(ValueError, match="only 1/2 valid samples"):
        export_mixture(streams, [2, 0, 0, 0, 0], Processor(), {}, tmp_path / "mixed.jsonl")
    assert not (tmp_path / "mixed.jsonl").exists()


def test_source_paths_splits_and_streaming(tmp_path, monkeypatch):
    calls = []
    class Rows(list):
        def shuffle(self, seed):
            calls.append(("shuffle", seed))
            return self
    def load(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        if Path(path).name == "slimpajama-200k":
            return Rows([{"text": "plain text " * 30}])
        if Path(path).name == "ChartQA":
            return Rows([{"question": "count", "answer": "one", "image": Image.new("RGB", (2, 2))}])
        suffix = "_zh" if Path(path).name == "alpaca-data-gpt4-chinese" else ""
        return Rows([{f"instruction{suffix}": "q", f"output{suffix}": "a"}])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    args = SimpleNamespace(datasets_root=tmp_path, cog_root=tmp_path,
                           text_prefix_tokens=12, chart_split="train", seed=11)
    streams = source_streams(args, Processor(), {"max_length": 128})
    for source in ("slimpajama", "codealpaca", "chartqa", "alpaca_zh"):
        assert next(streams[source])["source"] == source
    assert ("ChartQA", {"split": "train"}) in calls
    assert ("shuffle", 11) in calls
    for source in ("slimpajama-200k", "CodeAlpaca-20k", "alpaca-data-gpt4-chinese"):
        assert (source, {"split": "train", "streaming": True}) in calls


def test_cli_exports_using_checkpoint_processor_settings(tmp_path, monkeypatch, capsys):
    import prefetch.datasets.prepare_calibration as module
    import prefetch.evaluation.evaluate as evaluate
    processor = Processor()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)))
    monkeypatch.setattr(evaluate, "experiment_config", lambda *args:
                        {"model": {"path": "model"}, "data": {"max_length": 128, "max_image_side": None}})
    image = tmp_path / "source.png"
    Image.new("RGB", (5, 5), "red").save(image)
    monkeypatch.setattr(module, "source_streams", lambda *args: make_streams(image))
    output = tmp_path / "mixed.jsonl"
    monkeypatch.setattr(sys, "argv", ["prepare_calibration", "--checkpoint", "predictor",
                                    "--counts", "1", "1", "2", "1", "1", "--output", str(output)])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 6
    assert report["processor"] == "model" and report["chart_split"] == "test"
    assert report["data"]["max_image_side"] is None
