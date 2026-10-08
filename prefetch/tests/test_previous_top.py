"""Token-conditioned prediction, alignment, and decode-entry scheduling."""
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from prefetch.evaluation.routing import RoutingMetrics
from prefetch.prerouter.block import Prerouter
from prefetch.prerouter.checkpoint import load_predictor, predictor_state_dict, save_predictor
from prefetch.prerouter.patch import PrefetchConfig, patch, unpatch
from prefetch.tests.test_generation_patch import GeneratingModel, assert_idle
from prefetch.tests.test_patch import ToyModel, config


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ToyModel().eval().requires_grad_(False)


def condition_heads(state):
    for head in state.prerouters.values():
        nn.init.normal_(head.token_proj.weight, std=0.2)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_conditioned_head_matches_formula(kind):
    head = Prerouter(6, 8, head=kind, hidden_dim=5, embedding_dim=4)
    hidden = torch.randn(3, 6, requires_grad=True)
    tokens = torch.randn(3, 4, requires_grad=True)
    assert torch.count_nonzero(head.token_proj.weight) == 0
    torch.testing.assert_close(head(hidden, tokens), head(hidden), rtol=0, atol=0)
    nn.init.normal_(head.token_proj.weight, std=0.2)
    token_features = F.linear(tokens, head.token_proj.weight)
    if kind == "linear":
        expected = F.linear(hidden, head.net.weight) + token_features
    else:
        features = F.linear(hidden, head.net[0].weight, head.net[0].bias) + token_features
        expected = F.linear(F.gelu(features), head.net[2].weight, head.net[2].bias)
    actual = head(hidden, tokens)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, head(hidden, tokens + 1))
    actual.square().mean().backward()
    assert hidden.grad is not None and tokens.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize("distance", [0, 1, 2])
@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("enabled", [False, True])
def test_all_predictions_ready_before_decode_layers(model, distance, kind, enabled):
    ids = torch.tensor([[1, 3, 5]])
    native = model(ids).logits
    state = patch(model, PrefetchConfig(mode="previous_top", distance=distance, head=kind,
                                        hidden_dim=5, prerouter_enabled=enabled, ks=[2, 8]))
    condition_heads(state)
    state.reset(generation=True)
    calls, handles = [], []
    for source, head in state.prerouters.items():
        handles.append(head.register_forward_hook(
            lambda module, args, output, source=source: calls.append(source)))
    torch.testing.assert_close(model(ids).logits, native, rtol=0, atol=0)
    assert not calls and not state.predictions and not state.next_predictions
    assert state.token_embeddings is None
    assert set(state.source_hidden) == set(state.prerouters)
    for value in state.source_hidden.values():
        assert value.shape == (1, 6) and not value.requires_grad and value._base is None

    saved = dict(state.source_hidden)
    token = torch.tensor([[7]])
    with torch.no_grad():
        expected = {target: state.prerouters[source](saved[source], model.embed(token)[0])
                    for source, target in state.pairs}
    calls.clear()

    def before_first_layer(module, args):
        assert state.phase == "decode"
        assert calls == list(state.prerouters)
        assert not state.source_hidden
        assert set(state.predictions) == state.targets
        for target in state.targets:
            torch.testing.assert_close(state.predictions[target], expected[target], rtol=0, atol=0)

    handle = model.layers[0].register_forward_pre_hook(before_first_layer)
    model(token)
    assert calls == list(state.prerouters)  # Each head executes only at decode entry.
    assert all(state.source_hidden[source] is not saved[source] for source in saved)
    for observer in handles + [handle]:
        observer.remove()


@pytest.mark.parametrize("distance", [0, 1, 2])
@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("enabled", [False, True])
def test_conditioned_teacher_forcing_matches_decode(model, distance, kind, enabled):
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    state = patch(model, PrefetchConfig(mode="previous_top", distance=distance, head=kind, hidden_dim=5,
                                        prerouter_enabled=enabled, trace_limit=50, ks=[2, 8]))
    condition_heads(state)
    meter = RoutingMetrics(state)
    state.reset(router_mask=torch.tensor([[False, False, True, True, True]]))
    full = model(ids).logits
    expected = {target: value.detach().clone() for target, value in state.predictions.items()}
    meter.update(state)
    full_counts = meter.counts[0].clone()
    meter.reset()
    state.reset(generation=True)
    outputs = [model(ids[:, :2]).logits]
    for position in range(2, ids.shape[1]):
        outputs.append(model(ids[:, position:position + 1]).logits)
        for target in state.targets:
            torch.testing.assert_close(state.predictions[target], expected[target][position - 1:position])
        meter.update(state)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    torch.testing.assert_close(meter.counts[1], full_counts, rtol=0, atol=0)
    assert all(row["source_token"] == row["target_token"] - 1 for row in meter.trace)
    assert meter.report()["metadata"]["producer_timing"] == "decode_start_after_token_embedding"


def test_sampled_token_changes_prediction_with_fixed_context(model):
    state = patch(model, config(mode="previous_top", distance=0))
    condition_heads(state)
    predictions = []
    for token in (5, 7):
        state.reset(generation=True)
        model(torch.tensor([[1, 3]]))
        model(torch.tensor([[token]]))
        predictions.append(dict(state.predictions))
    for target in state.targets:
        assert not torch.equal(predictions[0][target], predictions[1][target])


def test_failure_at_decode_entry_clears_features(monkeypatch):
    model = GeneratingModel().eval().requires_grad_(False)
    state = patch(model, config(mode="previous_top", distance=0))
    def fail(*args):
        raise RuntimeError("conditioned head failed")
    with monkeypatch.context() as context:
        context.setattr(state.prerouters[0], "forward", fail)
        with pytest.raises(RuntimeError, match="conditioned head failed"):
            model.generate(torch.tensor([[1, 3]]))
    assert_idle(state)
    model.generate(torch.tensor([[5, 7]]))
    assert_idle(state)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("mode", ["same_token", "previous_token", "previous_top"])
def test_checkpoint_key_mapping_and_restore(model, monkeypatch, kind, mode):
    cfg = PrefetchConfig(mode=mode, distance=1, targets=[3, 1], head=kind, hidden_dim=5)
    state = patch(model, cfg)
    if mode == "previous_top":
        condition_heads(state)
    weights = {key: value.clone() for key, value in predictor_state_dict(state).items()}
    expected_keys = {f"{target}.{name}" for source, target in state.pairs
                     for name in state.prerouters[source].net.state_dict()}
    if mode == "previous_top":
        expected_keys |= {"3.token_proj.weight", "1.token_proj.weight"}
    assert set(weights) == expected_keys
    ids = torch.tensor([[1, 3, 5]])
    model(ids)
    predictions = dict(state.predictions)
    unpatch(model)
    state = patch(model, cfg)
    for parameter in state.parameters():
        nn.init.zeros_(parameter)
    # Isolate checkpoint key handling from the optional safetensors file transport.
    transport = ModuleType("safetensors.torch")
    transport.load_file = lambda path: weights
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", transport)
    load_predictor(state, "unused", {"layers": state.layers})
    for key, value in predictor_state_dict(state).items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    model(ids)
    for target in state.targets:
        torch.testing.assert_close(state.predictions[target], predictions[target], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_conditioning_weights_checkpoint_roundtrip(model, tmp_path, kind):
    pytest.importorskip("safetensors")
    state = patch(model, PrefetchConfig(mode="previous_top", distance=0, targets=[3, 0], head=kind))
    condition_heads(state)
    weights = {key: value.clone() for key, value in predictor_state_dict(state).items()}
    assert {key for key in weights if "token_proj" in key} == {"3.token_proj.weight", "0.token_proj.weight"}
    assert all(".net." not in key for key in weights)
    ids = torch.tensor([[1, 3, 5]])
    model(ids)
    predictions = dict(state.predictions)
    save_predictor(state, tmp_path)
    unpatch(model)
    state = patch(model, checkpoint=tmp_path)
    assert state.config.mode == "previous_top"
    for key, value in predictor_state_dict(state).items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    model(ids)
    for target in state.targets:
        torch.testing.assert_close(state.predictions[target], predictions[target], rtol=0, atol=0)
