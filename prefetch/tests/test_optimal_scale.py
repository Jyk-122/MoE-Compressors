"""Joint regression, output scaling, and single-pass calibration."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.compensation import OptimalScale
from prefetch.compensation.artifacts import load_artifact, save_artifact
from prefetch.compensation.optimal_scale.calibrate import ScaleStats, calibrate_layers
from prefetch.prerouter.patch import patch, unpatch
from prefetch.tests.test_patch import ToyModel, ToyMoE, ROUTING_CASES, config
from prefetch.tests.test_generation_patch import GeneratingModel, assert_idle


@pytest.mark.parametrize("center", [0.0, 1.0])
def test_joint_regression_matches_augmented_least_squares(center):
    torch.manual_seed(31)
    tokens, experts, hidden = 20, 5, 4
    outputs = torch.randn(tokens, experts, hidden, dtype=torch.float64)
    native_ids = torch.tensor([[0, 1]]).expand(tokens, -1)
    predicted_ids = torch.tensor([[1, 2], [2, 3]]).repeat(tokens // 2, 1)
    g = torch.softmax(torch.randn(tokens, 2, dtype=torch.float64), -1) * 2.5
    b = torch.softmax(torch.randn(tokens, 2, dtype=torch.float64), -1) * 2.5
    u = outputs.gather(1, native_ids[..., None].expand(-1, -1, hidden))
    v = outputs.gather(1, predicted_ids[..., None].expand(-1, -1, hidden))
    # An unobserved channel should retain scale 1 even when other channels were seen.
    v[:, :, -1] = 0
    stats = ScaleStats(experts, hidden, channel_chunk_size=3)
    for chunk in (slice(0, 7), slice(7, None)):
        stats.update(0, (native_ids[chunk], g[chunk]), (predicted_ids[chunk], b[chunk]), u[chunk], v[chunk])
    ridge = 0.02
    scale, report = stats.solve(ridge, center)

    y = (u * g[..., None]).sum(1)
    design = torch.zeros(tokens, experts, hidden, dtype=torch.float64)
    design.scatter_(1, predicted_ids[..., None].expand_as(v), v * b[..., None])
    expected = torch.empty(experts, hidden, dtype=torch.float64)
    for h in range(hidden):
        matrix = torch.cat([design[:, :, h], ridge ** 0.5 * torch.eye(experts)])
        target = torch.cat([y[:, h], torch.full((experts,), ridge ** 0.5 * center)])
        expected[:, h] = torch.linalg.lstsq(matrix, target).solution
    observed = design.square().sum(0) > 0
    expected[~observed] = 1
    torch.testing.assert_close(scale, expected.float())
    assert scale[0].eq(1).all() and scale[4].eq(1).all()
    assert scale[:, -1].eq(1).all()
    prediction = (design * scale.double()).sum(1)
    error = (prediction - y).square().sum() / y.square().sum()
    baseline = (design.sum(1) - y).square().sum() / y.square().sum()
    assert report["normalized_mse"] == pytest.approx(float(error), abs=1e-7)
    assert report["baseline_normalized_mse"] == pytest.approx(float(baseline))
    assert report["tokens"] == tokens
    assert report["observed_experts"] == 3
    if center == 1:
        assert error <= baseline


def make_experts(kind, dtype):
    if kind == "toy":
        return ToyMoE(hidden=6, experts=8).experts.to(dtype)
    source = Path(__file__).resolve().parents[1] / "assets" / "modeling.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "OpenPXXV2Experts")
    namespace = dict(torch=torch, nn=nn, ACT2FN={"silu": torch.nn.functional.silu},
                     use_experts_implementation=lambda cls: cls)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    cfg = SimpleNamespace(n_routed_experts=8, hidden_size=6, moe_intermediate_size=4, hidden_act="silu")
    experts = namespace["OpenPXXV2Experts"](cfg).to(dtype)
    for parameter in experts.parameters():
        nn.init.normal_(parameter, std=0.2)
    if kind == "nf4_interface":
        # Exercise NF4Experts.forward with CPU linears; actual quantized CUDA is separate.
        from prefetch.backbone.quantization import NF4Experts
        adapter = NF4Experts.__new__(NF4Experts)
        nn.Module.__init__(adapter)
        adapter.act_fn = torch.nn.functional.silu
        adapter.gate_up = nn.ModuleList([nn.Linear(6, 8, bias=False, dtype=dtype) for _ in range(8)])
        adapter.down = nn.ModuleList([nn.Linear(4, 6, bias=False, dtype=dtype) for _ in range(8)])
        with torch.no_grad():
            for e in range(8):
                adapter.gate_up[e].weight.copy_(experts.gate_up_proj[e])
                adapter.down[e].weight.copy_(experts.down_proj[e])
        return adapter
    return experts


@pytest.mark.parametrize("kind,dtype", [
    ("toy", torch.float32),
    ("bf16", torch.float32), ("bf16", torch.bfloat16),
    ("nf4_interface", torch.float32), ("nf4_interface", torch.bfloat16),
])
def test_scale_before_sum_with_signed_channels_and_native_rows(kind, dtype):
    torch.manual_seed(4)
    experts = make_experts(kind, dtype)
    x = torch.randn(4, 6).to(dtype)
    ids = torch.tensor([[0, 2], [2, 3], [1, 2], [3, 0]])
    weights = torch.tensor([[1., 1.5], [0.5, 2.], [1.3, 1.2], [2., 0.5]])
    scale = torch.randn(8, 6)
    rows = torch.tensor([1, 3])
    expected = torch.zeros_like(x)
    for e in ids.unique():
        token, slot = torch.where(ids == e)
        values = experts(x[token], ids[token, slot, None], torch.ones(len(token), 1)).float()
        factors = torch.ones_like(values)
        factors[torch.isin(token, rows)] = scale[e]
        values = values * factors * weights[token, slot, None]
        expected.index_add_(0, token, values.to(dtype))
    calls = []
    handle = experts.register_forward_pre_hook(lambda module, args: calls.append(args))
    actual = OptimalScale(scale)(experts, x, ids, weights, rows)
    handle.remove()
    torch.testing.assert_close(actual, expected)
    assert len(calls) == ids.unique().numel()
    assert sum(args[1].numel() for args in calls) == ids.numel()
    assert all(args[1].unique().numel() == 1 for args in calls)
    native = experts(x, ids, weights)
    torch.testing.assert_close(actual[[0, 2]], native[[0, 2]])
    torch.testing.assert_close(OptimalScale(torch.ones_like(scale))(experts, x, ids, weights, rows), native)


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_patch_applies_scale_only_to_aligned_response_rows(mode, distance):
    torch.manual_seed(8)
    model = ToyModel().eval().requires_grad_(False)
    ids = torch.tensor([[0, 1, 3, 5, 7, 0]])
    reference = model(ids).logits
    state = patch(model, config(mode=mode, distance=distance, targets=[3], excluded_token_ids=[3],
                                execution_mode="compensated", compensation={"method": "owa"}))
    scale = torch.randn(8, 6)
    state.compensators[3] = OptimalScale(scale)
    block = state.blocks[3][1]
    inputs, routes = [], []
    handle = block.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    state.route_observer = lambda *args: routes.append(args)
    response = torch.tensor([[False, False, True, True, True, False]])
    state.reset(router_mask=response)
    actual = model(ids, attention_mask=ids != 0).logits
    handle.remove()

    hidden = inputs[0][0]
    native_ids, native_weights = block.route_tokens_to_experts(block.gate(inputs[0]))
    all_ids, all_weights = native_ids.clone(), native_weights.clone()
    selected, baseline = routes[0][-1]
    rows = torch.tensor([3, 4])
    all_ids[rows], all_weights[rows] = selected, baseline
    outputs = torch.einsum("eij,tj->tei", block.experts.weight, hidden)
    selected_outputs = outputs.gather(1, all_ids[..., None].expand(-1, -1, 6))
    factors = torch.ones_like(selected_outputs)
    factors[rows] = scale[selected]
    expected = (selected_outputs * factors * all_weights[..., None]).sum(1)
    expected += block.shared_experts(hidden)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[:, [0, 1, 2, 5]], reference[:, [0, 1, 2, 5]])

    calls = []
    handle = state.compensators[3].register_forward_hook(lambda *args: calls.append(1))
    state.reset(router_mask=torch.zeros_like(response))
    torch.testing.assert_close(model(ids).logits, reference)
    handle.remove()
    assert not calls


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_identity_scale_generation_matches_predicted_and_resets(mode, distance):
    torch.manual_seed(9)
    model = GeneratingModel().eval().requires_grad_(False)
    state = patch(model, config(mode=mode, distance=distance, execution_mode="predicted"))
    prompt = torch.tensor([[1, 5, 7]])
    expected = model.generate(prompt)
    state.config.execution_mode = "compensated"
    state.compensators = {target: OptimalScale(torch.ones(8, 6)) for target in state.targets}
    phases, handles = [], []
    for module in state.compensators.values():
        handles.append(module.register_forward_hook(lambda *args: phases.append(state.phase)))
    for _ in range(2):
        actual = model.generate(prompt)
        torch.testing.assert_close(actual.sequences, expected.sequences)
        for result, reference in zip(actual.logits, expected.logits):
            torch.testing.assert_close(result, reference)
        assert_idle(state)
    assert phases and set(phases) == {"decode"}
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("max_tokens", [3, 9])
def test_single_pass_matches_separate_collection_and_releases_stats(
        tmp_path, monkeypatch, mode, distance, max_tokens):
    import prefetch.compensation.optimal_scale.calibrate as calibration
    import prefetch.training.runtime as runtime

    torch.manual_seed(14)
    model = ToyModel().eval().requires_grad_(False)
    ids = torch.tensor([[1, 3, 5]])
    original = model(ids).logits
    state = patch(model, config(mode=mode, distance=distance, targets=[2, 3]))
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"ids": [1, 3, 5]}\n' * 3, encoding="utf-8")

    def collate(records):
        return dict(input_ids=ids, labels=ids, router_mask=torch.ones_like(ids, dtype=torch.bool))
    monkeypatch.setattr(runtime, "make_collator", lambda *args: collate)
    args = SimpleNamespace(chunk_size=1, max_tokens=max_tokens, limit=2, sample_file=samples,
                           channel_chunk_size=2, ridge=1e-4, ridge_center=1.0,
                           output=tmp_path / "scale.safetensors")
    expected = {}
    for target in (2, 3):
        stats = ScaleStats(8, 6, args.channel_chunk_size)
        calibration.collect(model, None, state, {"model": {}, "data": {}}, "cpu", args,
                            stats, targets=[target])
        expected[target] = stats.solve(args.ridge, args.ridge_center)
    del stats

    tokens_per_example = 3 if mode == "same_token" else 2
    forward_count = min(args.limit, (max_tokens + tokens_per_example - 1) // tokens_per_example)
    token_count = min(max_tokens, args.limit * tokens_per_example)
    live, passes = [], []
    class TrackedStats(ScaleStats):
        def __init__(self, *args):
            super().__init__(*args)
            live.append(weakref.ref(self))

        def solve(self, *args):
            # All layers finish collection before the first solve.
            assert len(passes) == forward_count
            assert len(live) == 2
            assert all(ref().tokens == token_count for ref in live if ref() is not None)
            return super().solve(*args)

    monkeypatch.setattr(calibration, "ScaleStats", TrackedStats)
    collect_calls = []
    real_collect = calibration.collect
    def collect(*args, **kwargs):
        collect_calls.append(1)
        return real_collect(*args, **kwargs)
    monkeypatch.setattr(calibration, "collect", collect)

    def check_native(module, args, output):
        torch.testing.assert_close(output.logits, original)
        passes.append(1)
    handle = model.register_forward_hook(check_native)
    saved_layers = []
    def save(path, tensors, report):
        assert all(key.endswith(".scale") for key in tensors)
        saved_layers.append([row["target_moe"] for row in report["layers"]])
        assert sum(ref() is not None for ref in live) == 2 - len(saved_layers)
        save_artifact(path, tensors, report)
    monkeypatch.setattr(calibration, "save_artifact", save)
    report = calibrate_layers(model, None, state, {"model": {}, "data": {}}, "cpu", args,
                              {"layers": state.layers})
    handle.remove()
    assert len(collect_calls) == 1
    assert len(passes) == forward_count
    assert all(reference() is None for reference in live)
    assert saved_layers == [[2], [2, 3]]
    assert report["tokens"] == {2: token_count, 3: token_count}
    assert report["collection"] == "single_pass"
    values, metadata = load_artifact(args.output)
    assert set(values) == {"2.scale", "3.scale"}
    for target, (scale, metrics) in expected.items():
        torch.testing.assert_close(values[f"{target}.scale"], scale, rtol=0, atol=0)
        assert report["fit_metrics"][target] == pytest.approx(metrics)
        assert metrics["normalized_mse"] <= metrics["baseline_normalized_mse"] + 1e-8
    assert all(value.shape == (8, 6) and value.dtype == torch.float32 for value in values.values())
    assert state.route_observer is None and not state.compensators
    unpatch(model)
    state = patch(model, config(targets=[2, 3]), execution_mode="compensated",
                  compensation={"method": "optimal_scale", "path": str(args.output)})
    for target, module in state.compensators.items():
        torch.testing.assert_close(module.scale, values[f"{target}.scale"])
    assert metadata["ridge_center"] == 1.0


def test_calibration_cli_uses_single_pass_collection(tmp_path, monkeypatch, capsys):
    import sys
    import prefetch.compensation.optimal_scale.calibrate as calibration
    import prefetch.training.runtime as runtime
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(targets=[3]))
    ids = torch.tensor([[1, 3, 5]])
    monkeypatch.setattr(runtime, "make_collator", lambda *args: lambda records:
                        dict(input_ids=ids, router_mask=torch.ones_like(ids, dtype=torch.bool)))
    monkeypatch.setattr(calibration, "load_calibration",
                        lambda args: (model, None, state, {"model": {}, "data": {}},
                                      "cpu", {"layers": state.layers}))
    source, output = tmp_path / "data.jsonl", tmp_path / "scale.safetensors"
    source.write_text('{}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["calibrate", "--checkpoint", "trained",
                                    "--sample-file", str(source), "--limit", "1",
                                    "--ridge-center", "0", "--output", str(output)])
    calibration.main()
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == "optimal_scale" and report["ridge_center"] == 0
    assert report["max_tokens"] == 65536
    assert report["collection"] == "single_pass"
    assert report["tokens"] == {"3": 3}
    assert set(load_artifact(output)[0]) == {"3.scale"}
