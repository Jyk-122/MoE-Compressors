"""Predicted expert execution with native weights and independent routing labels."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefetch.prerouter.checkpoint import save_predictor
from prefetch.prerouter.patch import patch, unpatch
from prefetch.prerouter.routing import select_experts
from prefetch.evaluation.routing import RoutingMetrics
from prefetch.tests.test_patch import ROUTING_CASES, ToyModel, config


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ToyModel().eval().requires_grad_(False)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_predicted_ids_gather_native_scores(normalize, scale):
    block = SimpleNamespace(top_k=2, norm_topk_prob=normalize, routed_scaling_factor=scale,
                            e_score_correction_bias=torch.tensor([0., 0., 0., 0., 0., 0.3]))
    native = torch.tensor([[3., 2., 1., -1., -2., -3.]], requires_grad=True)
    predicted = torch.tensor([[0., -3., -2., 1., 3., 0.]], requires_grad=True)
    native_indices = torch.tensor([[0, 1]])
    native_weights = native.sigmoid().gather(1, native_indices)
    if normalize:
        native_weights = native_weights / native_weights.sum(-1, keepdim=True)
    original = (native_indices, native_weights * scale)

    indices, weights = select_experts(block, native, original, predicted)
    # Expert 5 outranks 3 only after adding the choice correction bias.
    assert indices.tolist() == [[4, 5]]
    expected = native.sigmoid()[:, [4, 5]]
    if normalize:
        expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected * scale)
    weights[0, 0].backward()
    assert native.grad is not None and native.grad.abs().sum() > 0
    assert predicted.grad is None
    assert select_experts(block, native, original) is original


def test_prediction_ties_follow_metric_order():
    block = SimpleNamespace(top_k=2, norm_topk_prob=True, routed_scaling_factor=1.0,
                            e_score_correction_bias=torch.zeros(4))
    original = (torch.tensor([[2, 3]]), torch.tensor([[0.5, 0.5]]))
    indices, _ = select_experts(block, torch.zeros(1, 4), original, torch.zeros(1, 4))
    assert indices.tolist() == [[0, 1]]


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("enabled", [False, True])
def test_execution_and_teacher_are_separate(model, mode, distance, enabled):
    ids = torch.tensor([[0, 1, 3, 5, 7, 0]])
    attention = ids != 0
    original = model(ids, attention_mask=attention).logits
    state = patch(model, config(mode=mode, distance=distance, excluded_token_ids=[3],
                                prerouter_enabled=enabled))
    response_mask = torch.tensor([[False, False, True, True, True, False]])
    state.reset(router_mask=response_mask)
    captured, native_logits, handles = {}, {}, []
    for index, (_, block) in enumerate(state.blocks):
        handles.append(block.gate.register_forward_hook(
            lambda module, args, output, index=index: native_logits.update({index: output.detach()})))
        handles.append(block.experts.register_forward_pre_hook(
            lambda module, args, index=index: captured.update({index: (args[1].clone(), args[2].clone())})))
    output = model(ids, attention_mask=attention).logits
    available = attention[0].clone()
    if mode == "previous_token":
        available[0] = False
        available[1:] &= attention[0, :-1]
    available &= response_mask[0] & (ids[0] != 3)
    start = int(mode == "previous_token")
    assert state.target_start == start
    differences = 0
    for target in state.targets:
        block = state.blocks[target][1]
        teacher = native_logits[target]
        native_indices, native_weights = block.route_tokens_to_experts(teacher)
        torch.testing.assert_close(state.router_logits[target], teacher[start:], rtol=0, atol=0)
        torch.testing.assert_close(state.router_indices[target], native_indices[start:], rtol=0, atol=0)
        indices, weights = captured[target]
        if enabled:
            scores = state.predictions[target].float().sigmoid() + block.e_score_correction_bias
            predicted = scores.sort(dim=-1, descending=True, stable=True).indices[:, :block.top_k]
            assert predicted.shape == (ids.shape[1] - start, block.top_k)
            expected_indices = native_indices.clone()
            positions = available.nonzero().flatten()
            expected_indices[positions] = predicted[available[start:]]
            expected_weights = teacher.sigmoid().gather(1, expected_indices)
            differences += int((indices.sort(-1).values != native_indices.sort(-1).values).sum())
        else:
            expected_indices, expected_weights = native_indices, native_weights
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    if enabled:
        assert differences > 0
        assert not torch.equal(output, original)
    else:
        torch.testing.assert_close(output, original, rtol=0, atol=0)
    meter = RoutingMetrics(state)
    meter.update(state)
    assert meter.report()["metadata"]["config"]["prerouter_enabled"] is enabled
    state.config.prerouter_enabled = False
    state.reset()
    torch.testing.assert_close(model(ids, attention_mask=attention).logits, original, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_enabled_full_sequence_matches_incremental(model, mode, distance):
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    state = patch(model, config(mode=mode, distance=distance, prerouter_enabled=True))
    state.reset(router_mask=torch.tensor([[False, False, True, True, True]]))
    full = model(ids).logits
    state.reset(generation=True)
    outputs = [model(ids[:, :2]).logits]
    for position in range(2, ids.shape[1]):
        prior = dict(state.next_predictions)
        outputs.append(model(ids[:, position:position + 1]).logits)
        if mode == "previous_token":
            assert all(state.predictions[target] is prior[target] for target in state.targets)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    state.reset(generation=True)
    torch.testing.assert_close(model(ids[:, :2]).logits, outputs[0], rtol=0, atol=0)
    assert state.config.prerouter_enabled


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_untargeted_layers_keep_native_route(model, mode, distance):
    state = patch(model, config(mode=mode, distance=distance, targets=[3], prerouter_enabled=True))
    state.reset(router_mask=torch.ones(1, 3, dtype=torch.bool))
    handles = []
    native = {}
    for index in range(3):
        block = state.blocks[index][1]
        handles.append(block.gate.register_forward_hook(
            lambda module, args, output, index=index, block=block:
                native.update({index: block.route_tokens_to_experts(output)})))
        def check(module, args, index=index):
            torch.testing.assert_close(args[1], native[index][0], rtol=0, atol=0)
            torch.testing.assert_close(args[2], native[index][1], rtol=0, atol=0)
        handles.append(block.experts.register_forward_pre_hook(check))
    model(torch.tensor([[1, 3, 5]]))
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_native_tail_prediction_and_decode_metrics(model, mode, distance, enabled):
    ids = torch.tensor([[1, 5, 7]])
    original = model(ids).logits
    state = patch(model, config(mode=mode, distance=distance, prerouter_enabled=enabled,
                                excluded_token_ids=[3], trace_limit=20))
    meter = RoutingMetrics(state)
    calls, handles = [], []
    for head in state.prerouters.values():
        handles.append(head.register_forward_hook(
            lambda module, args, output: calls.append((args[0].shape[0], output.shape[0]))))
    state.reset(generation=True)
    torch.testing.assert_close(model(ids).logits, original, rtol=0, atol=0)
    meter.update(state)
    assert not state.predictions and not state.router_logits and not state.router_indices
    assert int(meter.counts.sum()) == 0 and not meter.trace
    if mode == "previous_token":
        assert calls == [(1, 1)] * len(state.pairs)
        assert all(value.shape == (1, 8) for value in state.next_predictions.values())
    else:
        assert not calls and not state.next_predictions
    # Every actual decode call is counted, even an ID excluded from text supervision.
    model(torch.tensor([[3]]))
    meter.update(state)
    assert int(meter.counts[1, :, 1].sum()) == len(state.pairs)
    assert all(row["phase"] == "decode" and row["target_token"] == 3 for row in meter.trace)
    for handle in handles:
        handle.remove()


def test_teacher_forcing_requires_response_scope(model):
    state = patch(model, config(prerouter_enabled=True))
    with pytest.raises(ValueError, match="response_mask"):
        model(torch.tensor([[1, 3, 5]]))
    state.config.prerouter_enabled = False
    model(torch.tensor([[1, 3, 5]]))
    with pytest.raises(ValueError, match="response router_mask"):
        RoutingMetrics(state).update(state)


@pytest.mark.parametrize("saved_enabled", [None, False, True])
def test_checkpoint_execution_policy_override(model, tmp_path, saved_enabled):
    pytest.importorskip("safetensors")
    state = patch(model, config(prerouter_enabled=bool(saved_enabled)))
    save_predictor(state, tmp_path)
    if saved_enabled is None:
        path = tmp_path / "prefetch_config.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["config"].pop("prerouter_enabled")
        path.write_text(json.dumps(metadata), encoding="utf-8")
    unpatch(model)
    state = patch(model, checkpoint=tmp_path)
    assert state.config.prerouter_enabled is bool(saved_enabled)
    unpatch(model)
    state = patch(model, {"prerouter_enabled": True}, checkpoint=tmp_path)
    assert state.config.prerouter_enabled
    unpatch(model)
    state = patch(model, {"prerouter_enabled": True}, checkpoint=tmp_path, prerouter_enabled=False)
    assert not state.config.prerouter_enabled


@pytest.mark.parametrize("module_name", ["prefetch.examples.infer_prefetch_demo", "prefetch.evaluation.evaluate"])
@pytest.mark.parametrize("flag,expected", [(None, None), ("--prerouter-enabled", True),
                                         ("--no-prerouter-enabled", False)])
def test_cli_passes_execution_override(monkeypatch, module_name, flag, expected):
    import importlib
    import sys
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "experiment_config", lambda *args: {"prefetch": {}})
    monkeypatch.setattr(module, "load_model", lambda *args: (None, None, {}))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    if hasattr(module, "setup"):
        monkeypatch.setattr(module, "setup", lambda *args: "cpu")
    class ReachedPatch(Exception):
        pass
    def check_patch(*args, **kwargs):
        assert kwargs["prerouter_enabled"] is expected
        raise ReachedPatch
    monkeypatch.setattr(module, "patch", check_patch)
    argv = [module_name, "--checkpoint", "unused", "--output", "unused"]
    monkeypatch.setattr(sys, "argv", argv + ([flag] if flag else []))
    with pytest.raises(ReachedPatch):
        module.main()
