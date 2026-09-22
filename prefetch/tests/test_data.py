from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefetch.datasets.dataset import OverlengthSample, SFTCollator


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
    batch = SFTCollator(ToyProcessor(), router_tokens="all_text")([example([str(path)])])
    assert int((batch["input_ids"] == 4).sum()) == 3
    assert not batch["router_mask"][batch["input_ids"] == 4].any()


def test_overlength_is_explicit():
    with pytest.raises(OverlengthSample):
        SFTCollator(ToyProcessor(), max_length=2)([example()])
