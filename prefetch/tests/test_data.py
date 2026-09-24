from types import SimpleNamespace
import json
import logging
import sys

import pytest

torch = pytest.importorskip("torch")

from prefetch.datasets.dataset import MissingUserTurn, OverlengthSample, SFTCollator, load_records, training_records
from prefetch.datasets.prepare import filter_records


class ToyProcessor:
    tokenizer = SimpleNamespace(all_special_ids=[1, 2, 3, 4])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += "U" if message["role"] == "user" else "A"
            for content in message["content"]:
                text += "I" if content["type"] == "image" else content["text"]
            text += "E"
        return text + ("A" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        # A processor that expands one image marker to several visual tokens.
        sequence = []
        for char in text[0]:
            sequence.extend([4, 4, 4] if char == "I" else [{"U": 1, "A": 2, "E": 3}.get(char, ord(char))])
        ids = torch.tensor([sequence])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


def example(images=None):
    return dict(id="one", images=images or [], messages=[dict(role="user", content="hello"),
                  dict(role="assistant", content="yes"), dict(role="user", content="why"),
                  dict(role="assistant", content="ok")])


def test_assistant_mask_matches_current_input_token():
    batch = SFTCollator(ToyProcessor())([example()])
    ids = batch["input_ids"][0]
    assert "".join(chr(x) for x in ids[batch["router_mask"][0]].tolist()) == "yesok"
    assert (batch["labels"] != -100).sum() == 7  # Includes both assistant EOS tokens.


def test_visual_expansion_is_preserved(tmp_path):
    from PIL import Image
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    batch = SFTCollator(ToyProcessor())([example([str(path)])])
    assert int((batch["input_ids"] == 4).sum()) == 3
    assert not batch["router_mask"][batch["input_ids"] == 4].any()


def test_router_supervision_requires_response_scope():
    with pytest.raises(ValueError, match="router_tokens must be assistant"):
        SFTCollator(ToyProcessor(), router_tokens="all_text")


def test_overlength_is_explicit():
    with pytest.raises(OverlengthSample):
        SFTCollator(ToyProcessor(), max_length=2)([example()])


@pytest.mark.parametrize("messages", [[], [dict(role="assistant", content="A cat.")],
                                       [dict(role="system", content="Describe images.")]])
def test_image_without_user_turn_is_identified_before_image_loading(messages):
    record = dict(id="missing-user", images=["not-opened.png"], messages=messages)
    with pytest.raises(MissingUserTurn, match="needs a user turn"):
        SFTCollator(ToyProcessor())([record])


def test_filter_skips_missing_user_and_continues(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: ToyProcessor())))
    records = [dict(example(), id="first"),
               dict(id="missing-user", images=["not-opened.png"], messages=[dict(role="assistant", content="A cat.")]),
               dict(id="long", images=[], messages=[dict(role="user", content="hello" * 100),
                                                     dict(role="assistant", content="yes")]),
               dict(example(), id="last")]
    source, output = tmp_path / "train.jsonl", tmp_path / "train.filtered.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    args = SimpleNamespace(input=str(source), output=str(output), model_path="toy", max_length=64, max_image_side=672)
    with caplog.at_level(logging.WARNING, logger="prefetch.datasets.prepare"):
        report = filter_records(args)
    assert report["kept"] == 2 and report["overlength"] == 1 and report["missing_user_turn"] == 1
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in saved] == ["first", "last"]
    assert all(record["num_tokens"] > 0 and record["assistant_text_tokens"] > 0 for record in saved)
    assert "missing-user" in caplog.text and f"{source}:2" in caplog.text
    assert "Filter train.jsonl" in capsys.readouterr().err


def test_filter_preserves_other_errors(tmp_path, monkeypatch):
    from prefetch.datasets import dataset

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: ToyProcessor())))

    def fail(self, examples):
        raise ValueError("chat-template prefixes do not align")

    monkeypatch.setattr(dataset.SFTCollator, "__call__", fail)
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(example()) + "\n", encoding="utf-8")
    args = SimpleNamespace(input=str(source), output=str(tmp_path / "filtered.jsonl"),
                           model_path="toy", max_length=64, max_image_side=672)
    with pytest.raises(ValueError, match="chat-template prefixes do not align"):
        filter_records(args)


@pytest.mark.parametrize("images", [[], ["image.jpg"]])
def test_image_paths_have_string_type(tmp_path, monkeypatch, images):
    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    record = dict(example(images), num_tokens=20, assistant_text_tokens=5)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_records(str(path), limit=1)
    assert loaded.features["images"] == datasets.Sequence(datasets.Value("string"))
    assert loaded[0] == record


def test_mix_text_and_vision_records(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    sources, expected = [], {}
    for source, images in (("cog", ["image.jpg"]), ("tulu", [])):
        record = dict(example(images), id=source, source=source, group_id=source,
                      num_tokens=20, assistant_text_tokens=5)
        path = tmp_path / f"{source}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        sources.append(dict(path=str(path), weight=0.5))
        expected[source] = record
    mixed = training_records(dict(train=sources, seed=42))
    assert mixed.features["images"] == datasets.Sequence(datasets.Value("string"))
    assert {row["source"] for row in mixed} == {"cog", "tulu"}
    assert all(row == expected[row["source"]] for row in mixed)
