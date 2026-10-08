"""Generation requests own routing state; observers retain their collected metrics."""
import inspect
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefetch.evaluation.cache.trace import DecodeTrace
from prefetch.evaluation.routing import RoutingMetrics, capture_generation
from prefetch.prerouter.patch import patch, unpatch
from prefetch.tests.test_patch import ROUTING_CASES, ToyModel, batch, config
from prefetch.training.train import TrainingTask


class GeneratingModel(ToyModel):
    @torch.no_grad()
    def generate(self, input_ids, *, max_new_tokens=3, **kwargs):
        sequences = input_ids
        outputs = []
        for step in range(max_new_tokens):
            current = input_ids if step == 0 else sequences[:, -1:]
            output = self(input_ids=current, attention_mask=torch.ones_like(sequences))
            outputs.append(output.logits)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            sequences = torch.cat([sequences, token], dim=1)
        return SimpleNamespace(sequences=sequences, logits=outputs, options=kwargs)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return GeneratingModel().eval().requires_grad_(False)


def assert_idle(state):
    assert not state.generation and not state.active and not state.train_prerouter
    assert state.phase == "teacher_forcing" and state.forward_index == -1
    assert state.token_offset == state.sequence_length == 0
    assert state.router_mask is None and state.valid_mask is None
    assert not state.predictions and not state.next_predictions
    assert not state.router_logits and not state.router_indices
    assert not state.source_hidden and state.token_embeddings is None


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prompt", [(1,), (1, 3, 5)])
def test_direct_generate_isolates_requests(model, mode, distance, enabled, prompt):
    ids = torch.tensor([prompt])
    expected = model.generate(ids)
    state = patch(model, config(mode=mode, distance=distance, prerouter_enabled=enabled))
    phases = []

    def observe(module, args, output):
        phases.append((state.phase, state.forward_index, state.token_offset))
        if state.phase == "prefill":
            assert not state.predictions and not state.router_indices
            expected_targets = state.targets if mode == "previous_token" else set()
            assert set(state.next_predictions) == expected_targets

    handle = model.register_forward_hook(observe)
    for _ in range(2):
        # Simulate a preceding teacher-forcing sample; generate owns its new state.
        state.reset(train_prerouter=True, router_mask=torch.ones_like(ids, dtype=torch.bool))
        model(input_ids=ids)
        phases.clear()
        actual = model.generate(ids)
        assert phases == [("prefill", 0, 0), ("decode", 1, len(prompt)),
                          ("decode", 2, len(prompt) + 1)]
        torch.testing.assert_close(actual.logits[0], expected.logits[0], rtol=0, atol=0)
        if not enabled:
            torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
            for result, reference in zip(actual.logits, expected.logits):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert_idle(state)
        assert state.config.prerouter_enabled is enabled
    handle.remove()


@pytest.mark.parametrize("instance_override", [False, True])
def test_generate_signature_passthrough_and_restore(model, instance_override):
    result = object()
    marker = object()
    if instance_override:
        def generate(self, inputs, *, option):
            assert self is model and inputs is marker and option is marker
            assert self.prerouter_state.generation
            return result
        model.generate = MethodType(generate, model)
    original = model.generate
    signature = inspect.signature(original)
    for _ in range(2):
        state = patch(model, config())
        assert inspect.signature(model.generate) == signature
        assert model.generate.__wrapped__ == original
        if instance_override:
            assert model.generate(marker, option=marker) is result
        else:
            output = model.generate(input_ids=torch.tensor([[1]]), max_new_tokens=1, option=marker)
            assert output.options["option"] is marker
        assert_idle(state)
        unpatch(model)
        assert model.generate == original
        assert ("generate" in model.__dict__) is instance_override
        assert not hasattr(model, "_prerouter_generate")


@pytest.mark.parametrize("after_forward", [False, True])
def test_generate_exception_cleans_state_and_allows_next_request(model, after_forward):
    original = model.generate
    failure = RuntimeError("generation failed")

    def generate(self, input_ids, *, fail=False):
        if after_forward:
            output = original(input_ids)
        if fail:
            raise failure
        return output if after_forward else original(input_ids)

    model.generate = MethodType(generate, model)
    state = patch(model, config(mode="previous_token", distance=0, prerouter_enabled=True))
    with pytest.raises(RuntimeError, match="generation failed") as caught:
        model.generate(torch.tensor([[1, 3]]), fail=True)
    assert caught.value is failure
    assert_idle(state)
    model.generate(torch.tensor([[5, 7]]))
    assert_idle(state)


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_generation_metrics_survive_cleanup(model, mode, distance):
    state = patch(model, config(mode=mode, distance=distance, trace_limit=20))
    meter = RoutingMetrics(state)
    for request in range(2):
        with capture_generation(model, state, meter):
            model.generate(torch.tensor([[1, 3]]))
            assert_idle(state)
        assert not model._forward_hooks
        assert int(meter.counts[0].sum()) == 0
        assert int(meter.counts[1, :, 1].sum()) == (request + 1) * 2 * len(state.pairs)
        assert {row["target_token"] for row in meter.trace} == {2, 3}


def test_generation_cache_trace_survives_cleanup(model):
    state = patch(model, config())
    trace = DecodeTrace(state, prediction_k=2)
    for index in range(2):
        with trace.capture(model):
            model.generate(torch.tensor([[1, 3]]))
            assert_idle(state)
        record = trace.request(id=str(index))
        assert record["decode_tokens"] == 2
        assert all(len(layer["truth"]) == 2 for layer in record["layers"])
        assert all(not block.experts._forward_pre_hooks for _, block in state.blocks)


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_training_after_generate(model, mode, distance):
    state = patch(model, config(mode=mode, distance=distance, prerouter_enabled=True))
    model.generate(torch.tensor([[1, 3]]))
    task = TrainingTask(model, state, "router")
    task(batch(), record_metrics=True).backward()
    assert state.phase == "teacher_forcing"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state.parameters())
