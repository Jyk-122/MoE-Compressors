"""Decode tracing observes native expert inputs and keeps all actual token calls."""
import json
import sys
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from prefetch.evaluation.cache.trace import DecodeTrace
from prefetch.prerouter.patch import patch
from prefetch.tests.test_patch import ToyModel, config


def test_trace_all_layers_tokens_and_prediction_ranking():
    model = ToyModel().eval().requires_grad_(False)
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([[3]])).logits
    state = patch(model, config(excluded_token_ids=[3], trace_limit=1))
    trace = DecodeTrace(state, prediction_k=2)
    with torch.no_grad(), trace.capture(model):
        model(input_ids=torch.tensor([[1, 2, 3]]))  # Prefill does not warm the simulated caches.
        actual = model(input_ids=torch.tensor([[3]])).logits
        truth = state.router_indices[1][0].tolist()
        block = state.blocks[1][1]
        scores = state.predictions[1].float().sigmoid() + block.e_score_correction_bias
        prediction = scores.sort(dim=-1, descending=True, stable=True).indices[0, :2].tolist()
        model(input_ids=torch.tensor([[5]]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    record = trace.request(id="sample", sample_index=0)
    assert record["decode_tokens"] == 2  # Includes special ID 3 and ignores trace_limit=1.
    assert len(record["layers"]) == 4
    assert record["layers"][0]["prediction"] is None
    assert len(record["layers"][0]["truth"]) == 2
    assert record["layers"][1]["truth"][0] == truth
    assert record["layers"][1]["prediction"][0] == prediction
    assert not state.generation and not state.active
    assert all(not block.experts._forward_pre_hooks for _, block in state.blocks)


def test_requests_are_independent_and_zero_decode_is_valid():
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config())
    trace = DecodeTrace(state, prediction_k=2)
    with torch.no_grad(), trace.capture(model):
        model(input_ids=torch.tensor([[1, 2]]))
        model(input_ids=torch.tensor([[3]]))
    first = trace.request(id="first", sample_index=0)
    with torch.no_grad(), trace.capture(model):
        model(input_ids=torch.tensor([[1, 2]]))
    second = trace.request(id="second", sample_index=1)
    assert first["decode_tokens"] == 1
    assert second["decode_tokens"] == 0
    assert all(not row["truth"] for row in second["layers"])


def test_observers_removed_after_exception():
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config())
    trace = DecodeTrace(state, prediction_k=2)
    with pytest.raises(RuntimeError, match="generation failed"):
        with trace.capture(model):
            raise RuntimeError("generation failed")
    assert all(not block.experts._forward_pre_hooks for _, block in state.blocks)
    assert not state.generation


def test_previous_token_trace_requires_event_timing_support():
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(mode="previous_token"))
    with pytest.raises(ValueError, match="same_token"):
        DecodeTrace(state)


def test_cache_trace_requires_native_execution():
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config())
    trace = DecodeTrace(state)
    state.config.prerouter_enabled = True
    with pytest.raises(ValueError, match="prerouter_enabled=False"):
        with trace.capture(model):
            pytest.fail("The routing policy must be checked before generation")


def test_collection_shards_complete_requests_for_offline_simulation(tmp_path, monkeypatch):
    import prefetch.evaluation.cache.collect as collection
    from prefetch.evaluation.cache.simulate import simulate_files
    configuration = dict(seed=42, model={"path": "toy", "quantization": "none"},
                         data={"validation": {"text": {"path": "samples"}}})
    monkeypatch.setattr(collection, "experiment_config", lambda *args: configuration)
    monkeypatch.setattr(collection, "read_metadata", lambda *args: {"config": {"mode": "same_token"}})
    monkeypatch.setattr(collection, "setup", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(collection, "seed_all", lambda *args: None)
    monkeypatch.setattr(collection, "world_size", lambda: 2)
    monkeypatch.setattr(collection, "load_records", lambda *args: [{"id": f"sample-{i}"} for i in range(3)])
    monkeypatch.setattr(collection, "generation_inputs", lambda *args: {"input_ids": torch.tensor([[1, 2]])})
    def install(model, **kwargs):
        assert kwargs["prerouter_enabled"] is False
        return patch(model, config(excluded_token_ids=[3], prerouter_enabled=True),
                     prerouter_enabled=kwargs["prerouter_enabled"])
    monkeypatch.setattr(collection, "patch", install)

    def load_model(*args):
        model = ToyModel().eval().requires_grad_(False)

        def generate(self, input_ids, max_new_tokens, **kwargs):
            self(input_ids=input_ids)
            for _ in range(max_new_tokens - 1):
                self(input_ids=torch.tensor([[3]]))
            return torch.cat([input_ids, torch.full((1, max_new_tokens), 3)], dim=1)

        model.generate = MethodType(generate, model)
        return model, None, {}

    monkeypatch.setattr(collection, "load_model", load_model)
    monkeypatch.setattr(sys, "argv", ["collect", "--checkpoint", str(tmp_path / "checkpoint"),
                                     "--output", str(tmp_path / "traces"), "--max-new-tokens", "3"])
    for process_rank in (0, 1):
        monkeypatch.setattr(collection, "rank", lambda: process_rank)
        collection.main()
    paths = sorted((tmp_path / "traces").glob("*.jsonl"))
    assert len(paths) == 2
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert records[-1]["type"] == "complete"
        assert all(row["decode_tokens"] == 2 for row in records[1:-1])
    report = simulate_files(paths, [8])
    assert report["sources"]["text"]["requests"] == 3
    assert report["sources"]["text"]["decode_tokens"] == 6
    assert all(len(row["truth"]) == 2 for row in records[1]["layers"])
