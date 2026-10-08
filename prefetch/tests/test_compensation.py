"""Numerical compensation, calibration, and patched routing contracts."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefetch.compensation import OWA, ExFold
from prefetch.compensation.artifacts import save_artifact, load_artifact
from prefetch.compensation.calibration import CalibrationObserver, expert_outputs, collect
from prefetch.compensation.owa.calibrate import OWACalibration
from prefetch.compensation.exfold.calibrate import PairwiseStats, ExFoldCalibration
from prefetch.prerouter.checkpoint import save_predictor
from prefetch.prerouter.patch import patch, unpatch
from prefetch.tests.test_patch import ToyModel, ToyMoE, ROUTING_CASES, config


def test_owa_partial_hit_uses_native_mass_and_scales_once():
    native = torch.tensor([[0, 1, 2]])
    predicted = torch.tensor([[3, 1, 0]])
    weights = torch.tensor([[1.25, 0.75, 0.5]])  # Native sum includes 2.5x scaling.
    baseline = torch.tensor([[0.5, 1.0, 1.0]])
    actual = OWA(alpha1=2, alpha2=0.8)(native, weights, predicted, baseline)
    # Hit mass=2, missing=.5; unnormalized predicted-order weights [.5, 1.875, 3.125].
    expected = torch.tensor([[0.5, 1.875, 3.125]]) * (2.0 / 5.5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.tensor([2.0]))


@pytest.mark.parametrize("method", ["owa", "exfold"])
def test_full_hit_permutation_and_empty_rows(method):
    module = OWA(2, 3) if method == "owa" else ExFold(torch.randn(4, 4), torch.rand(4, 4))
    ids, selected = torch.tensor([[0, 2]]), torch.tensor([[2, 0]])
    actual = module(ids, torch.tensor([[0.7, 0.3]]), selected, torch.ones(1, 2))
    torch.testing.assert_close(actual, torch.tensor([[0.3, 0.7]]))
    assert module(ids[:0], actual[:0], selected[:0], actual[:0]).shape == (0, 2)


def test_owa_zero_hits_and_configured_trigger():
    ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    selected = torch.tensor([[3, 4, 5], [0, 4, 5]])
    native = torch.tensor([[0.5, 0.3, 0.2]]).expand(2, -1)
    baseline = torch.tensor([[0.2, 0.2, 0.1]]).expand(2, -1)
    torch.testing.assert_close(OWA(2, 3, hit_min=2)(ids, native, selected, baseline), baseline)


def test_exfold_accumulates_signed_missing_contributions():
    c = torch.zeros(6, 6)
    loss = torch.full_like(c, 1e30)
    c[1, 0], loss[1, 0] = 2, 0.1
    c[2, 0], loss[2, 0] = -1, 0.2
    c[2, 4], loss[2, 4] = 3, 0.3
    ids, selected = torch.tensor([[0, 1, 2]]), torch.tensor([[4, 0, 5]])
    weights = ExFold(c, loss)(ids, torch.tensor([[0.5, 0.3, 0.2]]), selected, torch.ones(1, 3))
    # 0 retains .5 and receives .3*2 + .2*(-1); unused prefetched experts have weight zero.
    torch.testing.assert_close(weights, torch.tensor([[0., 0.9, 0.]]))


def test_exfold_zero_hit_and_unobserved_pair():
    c, loss = torch.zeros(4, 4), torch.full((4, 4), 1e30)
    c[0, 2], loss[0, 2] = -2, 0.1
    result = ExFold(c, loss)(torch.tensor([[0, 1]]), torch.tensor([[0.4, 0.6]]),
                             torch.tensor([[2, 3]]), torch.ones(1, 2))
    torch.testing.assert_close(result, torch.tensor([[-0.8, 0.]]))


def test_exfold_weighted_regression_matches_independent_objective():
    stats = PairwiseStats(3)
    u = torch.tensor([[[2., 1.]], [[3., -1.]], [[1., 4.]]])
    v = torch.tensor([[[1., 0.]], [[2., 1.]], [[-1., 2.]]])
    stats.update(torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1, dtype=torch.long), u, v)
    result = stats.finalize(ridge=0)
    weight = u.double().norm(dim=-1)
    c = (weight * (u.double() * v).sum(-1)).sum() / (weight * v.double().square().sum(-1)).sum()
    error = (weight * (u.double() - c * v).square().sum(-1)).sum()
    energy = (weight * u.double().square().sum(-1)).sum()
    torch.testing.assert_close(result["coefficients"][0, 1], c.float())
    torch.testing.assert_close(result["loss"][0, 1], (error / energy).float())
    assert result["count"][0, 1] == 3
    assert result["coefficients"][1, 0] == 0
    assert result["loss"][1, 0] >= 1e30
    clipped = stats.finalize(ridge=0, clip=0.2)
    assert clipped["coefficients"][0, 1] == pytest.approx(0.2)
    residual = (weight * (u.double() - 0.2 * v).square().sum(-1)).sum()
    assert float(clipped["loss"][0, 1]) == pytest.approx(float(residual / energy))


def test_owa_grid_search_recovers_output_optimum():
    collector = OWACalibration(alpha1=[1, 2], alpha2=[1])
    native = torch.tensor([[0, 1]]), torch.tensor([[0.5, 0.5]])
    predicted = torch.tensor([[0, 2]]), torch.tensor([[0.5, 0.5]])
    # alpha1=2 gives [.8,.2], whose output [.8,.2] equals the native mixture.
    u = torch.tensor([[[1., 0.], [0.6, 0.4]]])
    v = torch.tensor([[[1., 0.], [0., 1.]]])
    collector.update(0, native, predicted, u, v)
    result = collector.result()
    assert result["parameters"]["alpha1"] == 2
    assert result["normalized_mse"] < 1e-12
    assert result["baseline_normalized_mse"] > 0


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("method", ["owa", "exfold"])
def test_compensated_execution_respects_masks_ids_and_native_labels(mode, distance, method):
    torch.manual_seed(11)
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(mode=mode, distance=distance, targets=[3], excluded_token_ids=[3],
                                execution_mode="compensated", compensation={"method": "owa"}))
    if method == "exfold":
        state.compensators[3] = ExFold(torch.randn(8, 8), torch.rand(8, 8))
    ids = torch.tensor([[0, 1, 3, 5, 7, 0]])
    response = torch.tensor([[False, False, True, True, True, False]])
    block = state.blocks[3][1]
    recorded, observed = [], []
    handle = block.experts.register_forward_pre_hook(lambda module, args: recorded.append(args))
    state.route_observer = lambda *args: observed.append(args)
    state.reset(router_mask=response)
    model(ids, attention_mask=ids != 0)
    assert len(recorded) == 1  # Online compensation invokes only the final expert mixture.
    _, actual_ids, actual_weights = recorded[0]
    target, _, _, native, predicted = observed[0]
    assert target == 3
    expected = state.compensators[3](*native, *predicted)
    rows = torch.tensor([3, 4])
    torch.testing.assert_close(actual_ids[rows], predicted[0])
    torch.testing.assert_close(actual_weights[rows], expected)
    logits = state.router_logits[3]
    labels, weights = block.route_tokens_to_experts(logits)
    torch.testing.assert_close(state.router_indices[3], labels)
    start = int(mode != "same_token")
    unchanged = torch.tensor([i for i in range(start, 6) if i not in [3, 4]])
    torch.testing.assert_close(actual_ids[unchanged], labels[unchanged - start])
    torch.testing.assert_close(actual_weights[unchanged], weights[unchanged - start])
    handle.remove()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_calibration_observes_native_trajectory_and_aligned_response(mode, distance):
    torch.manual_seed(13)
    model = ToyModel().eval().requires_grad_(False)
    ids = torch.tensor([[0, 1, 3, 5, 7, 0]])
    original = model(ids, attention_mask=ids != 0).logits
    state = patch(model, config(mode=mode, distance=distance, excluded_token_ids=[3]))
    collector = ExFoldCalibration(state.layers)
    observer = CalibrationObserver(collector, chunk_size=1, max_tokens=1)
    state.route_observer = observer
    for _ in range(2):
        state.reset(router_mask=torch.tensor([[False, False, True, True, True, False]]))
        output = model(ids, attention_mask=ids != 0).logits
        torch.testing.assert_close(output, original, rtol=0, atol=0)
    assert all(observer.tokens[target] == 1 for target in state.targets)
    assert all(int(stats.count.sum()) == 4 for stats in collector.stats.values())


@pytest.mark.parametrize("sampling", ["prefetch", "co_routed"])
def test_calibration_artifact_roundtrip_and_layer_mapping(tmp_path, sampling):
    pytest.importorskip("safetensors")
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(targets=[3, 1]))
    collector = ExFoldCalibration(state.layers, sampling)
    state.route_observer = CalibrationObserver(collector, chunk_size=2)
    state.reset(router_mask=torch.ones(1, 3, dtype=torch.bool))
    model(torch.tensor([[1, 3, 5]]))
    tables, _ = collector.result()
    path = tmp_path / "exfold.safetensors"
    save_artifact(path, tables, dict(method="exfold", layers=state.layers))
    restored, _ = load_artifact(path)
    for key in tables:
        torch.testing.assert_close(tables[key], restored[key])
    save_predictor(state, tmp_path / "predictor")
    unpatch(model)
    state = patch(model, checkpoint=tmp_path / "predictor", execution_mode="compensated",
                  compensation={"path": str(path)})
    for target, module in state.compensators.items():
        torch.testing.assert_close(module.coefficients, tables[f"{target}.coefficients"])
    unpatch(model)
    with pytest.raises(ValueError, match="target layers"):
        patch(model, config(targets=[2]), execution_mode="compensated", compensation={"path": str(path)})


def test_owa_artifact_and_checkpoint_execution_overrides(tmp_path):
    pytest.importorskip("safetensors")
    model = ToyModel().eval()
    state = patch(model, config())
    path = tmp_path / "owa.safetensors"
    save_artifact(path, {"alpha": torch.tensor([2., 0.8])},
                  dict(method="owa", layers=state.layers, parameters={"hit_min": 1, "hit_max": 1}))
    unpatch(model)
    state = patch(model, config(execution_mode="compensated", compensation={"path": str(path)}))
    assert all(m.alpha1 == 2 for m in state.compensators.values())
    save_predictor(state, tmp_path / "predictor")
    unpatch(model)
    state = patch(model, checkpoint=tmp_path / "predictor", prerouter_enabled=False)
    assert state.config.execution == "native"
    unpatch(model)
    state = patch(model, {"execution_mode": "predicted"}, checkpoint=tmp_path / "predictor")
    assert state.config.execution == "predicted"
    unpatch(model)
    state = patch(model, checkpoint=tmp_path / "predictor", prerouter_enabled=False,
                  execution_mode="compensated")
    assert state.config.execution == "compensated"


def test_expert_output_collection_matches_backend_mixture():
    block = ToyMoE()
    x, ids = torch.randn(3, 6), torch.tensor([[0, 2], [1, 3], [2, 5]])
    weights = torch.tensor([[1., -2.], [0.1, 0.4], [0.5, 0.5]])
    outputs = expert_outputs(block, x, ids)
    torch.testing.assert_close((outputs * weights.unsqueeze(-1)).sum(1), block.experts(x, ids, weights))


def test_jsonl_collection_restores_observer_and_respects_limit(tmp_path, monkeypatch):
    import prefetch.training.runtime as runtime
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(mode="previous_top", distance=0))
    samples = tmp_path / "calibration.jsonl"
    samples.write_text(json.dumps({"ids": [1, 3, 5]}) + "\n", encoding="utf-8")
    def collate(records):
        ids = torch.tensor([records[0]["ids"]])
        return dict(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids),
                    router_mask=torch.tensor([[False, True, True]]))
    monkeypatch.setattr(runtime, "make_collator", lambda *args: collate)
    args = SimpleNamespace(chunk_size=1, max_tokens=2, limit=1, sample_file=samples)
    collector = OWACalibration()
    counts = collect(model, None, state, {"model": {}, "data": {}}, "cpu", args, collector)
    assert set(counts.values()) == {2}
    assert state.route_observer is None and not state.active
    assert collector.result()["tokens"] == 8


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("method", ["owa", "exfold"])
def test_generation_prefill_native_and_decode_compensated(mode, distance, method):
    torch.manual_seed(21)
    model = ToyModel().eval().requires_grad_(False)
    prompt = torch.tensor([[1, 5, 7]])
    reference = model(prompt).logits
    state = patch(model, config(mode=mode, distance=distance, excluded_token_ids=[3],
                                execution_mode="compensated", compensation={"method": "owa"}))
    calls, handles = [], []
    for target in state.targets:
        if method == "exfold":
            state.compensators[target] = ExFold(torch.randn(8, 8), torch.rand(8, 8))
        handles.append(state.compensators[target].register_forward_hook(
            lambda module, args, output, target=target: calls.append((target, args, output))))
    state.reset(generation=True)
    torch.testing.assert_close(model(prompt).logits, reference, rtol=0, atol=0)
    assert not calls
    for token in [3, 9]:
        calls.clear()
        model(torch.tensor([[token]]))
        assert len(calls) == len(state.targets)
        for target, args, output in calls:
            torch.testing.assert_close(args[0], state.router_indices[target])
            assert output.shape == (1, 2)
    for handle in handles:
        handle.remove()


def test_reference_expert_backend_applies_signed_compensation():
    import ast
    from pathlib import Path
    from torch import nn
    path = Path(__file__).resolve().parents[1] / "assets" / "modeling.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "OpenPXXV2Experts")
    namespace = dict(torch=torch, nn=nn, F=torch.nn.functional,
                     ACT2FN={"silu": torch.nn.functional.silu}, use_experts_implementation=lambda cls: cls)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(n_routed_experts=4, hidden_size=6, moe_intermediate_size=4, hidden_act="silu")
    experts = namespace["OpenPXXV2Experts"](cfg)
    for parameter in experts.parameters():
        nn.init.normal_(parameter, std=0.1)
    x = torch.randn(2, 6)
    native_ids = torch.tensor([[0, 1], [1, 0]])
    native_weights = torch.tensor([[1.5, 1.0], [1.5, 1.0]])
    predicted_ids = torch.tensor([[0, 2], [0, 2]])
    c, loss = torch.zeros(4, 4), torch.full((4, 4), 1e30)
    c[1, 2], loss[1, 2] = -2, 0.1
    weights = ExFold(c, loss)(native_ids, native_weights, predicted_ids, native_weights)
    outputs = expert_outputs(SimpleNamespace(experts=experts), x, predicted_ids)
    actual = experts(x, predicted_ids, weights)
    expected = outputs[:, 0] * torch.tensor([[1.5], [1.]]) - outputs[:, 1] * torch.tensor([[2.], [3.]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method,save", [("owa", False), ("owa", True), ("exfold", True)])
def test_calibration_cli_runs_collection_and_emits_results(tmp_path, monkeypatch, capsys, method, save):
    import importlib
    import sys
    import prefetch.training.runtime as runtime
    module = importlib.import_module(f"prefetch.compensation.{method}.calibrate")
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(mode="previous_token", distance=0))
    samples, output = tmp_path / "samples.jsonl", tmp_path / "result.safetensors"
    samples.write_text(json.dumps({"ids": [1, 3, 5]}) + "\n", encoding="utf-8")
    def collate(records):
        ids = torch.tensor([records[0]["ids"]])
        return dict(input_ids=ids, labels=ids, router_mask=torch.ones_like(ids, dtype=torch.bool))
    monkeypatch.setattr(runtime, "make_collator", lambda *args: collate)
    monkeypatch.setattr(module, "load_calibration",
                        lambda args: (model, None, state, {"model": {}, "data": {}},
                                      "cpu", {"layers": state.layers}))
    argv = [method, "--checkpoint", "trained", "--sample-file", str(samples), "--limit", "1"]
    if save:
        argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == method
    if save:
        values, metadata = load_artifact(output)
        assert metadata["method"] == method
        assert values
    if method == "owa":
        assert set(report["parameters"]) == {"alpha1", "alpha2", "hit_min", "hit_max"}
        assert report["tokens"] == 8
    else:
        assert len(report["pair_coverage"]) == 4
