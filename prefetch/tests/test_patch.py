from types import SimpleNamespace
import os
import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prefetch.prerouter.block import Prerouter
from prefetch.prerouter.checkpoint import predictor_state_dict, save_predictor
from prefetch.backbone.loading import install_lora, load_lora, save_lora
from prefetch.prerouter.patch import PrefetchConfig, patch, unpatch
from prefetch.evaluation.routing import RoutingMetrics, capture_generation
from prefetch.training.train import TrainingTask, compute_prerouter_loss, router_loss


ROUTING_CASES = [("same_token", 1), ("previous_token", 0),
                 ("previous_token", 1), ("previous_token", 2)]


class ToyRouter(nn.Linear):
    def forward(self, x):
        return super().forward(x).reshape(-1, self.out_features)


class ToyExperts(nn.Module):
    def __init__(self, hidden, experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, hidden, hidden) * 0.1)

    def forward(self, x, indices, weights):
        output = torch.einsum("nkij,nj->nki", self.weight[indices], x)
        return (output * weights.unsqueeze(-1)).sum(1)


class ToyMoE(nn.Module):
    """Same forward contract as assets/modeling.py, including shared experts."""
    def __init__(self, hidden=6, experts=8, top_k=2):
        super().__init__()
        self.gate = ToyRouter(hidden, experts, bias=False)
        self.experts = ToyExperts(hidden, experts)
        self.shared_experts = nn.Linear(hidden, hidden, bias=False)
        self.n_group, self.top_k = 1, top_k
        self.register_buffer("e_score_correction_bias", torch.linspace(-0.1, 0.1, experts))

    def route_tokens_to_experts(self, logits):
        index = (logits.sigmoid() + self.e_score_correction_bias).topk(self.top_k, dim=-1).indices
        return index, logits.sigmoid().gather(-1, index)

    def forward(self, x):
        logits = self.gate(x)
        indices, weights = self.route_tokens_to_experts(logits)
        output = self.experts(x.view(-1, x.shape[-1]), indices, weights).view_as(x)
        return output + self.shared_experts(x)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(6, 6, bias=False)
        self.self_attn.o_proj = nn.Linear(6, 6, bias=False)
        self.mlp = ToyMoE()

    def forward(self, x):
        return self.mlp(x + self.self_attn.o_proj(self.self_attn.qkv_proj(x)))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 6)
        self.layers = nn.ModuleList([ToyLayer() for _ in range(4)])

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=x, loss=x.float().square().mean())


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ToyModel().eval().requires_grad_(False)


def config(**kwargs):
    return PrefetchConfig(head="linear", ks=[1, 2, 4, 8], **kwargs)


def batch(ids=(1, 3, 5)):
    ids = torch.tensor([ids])
    return dict(input_ids=ids, attention_mask=ids != 0, router_mask=ids != 0, labels=ids)


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_prerouter_is_an_independent_module(head):
    head = Prerouter(6, 8, head=head, hidden_dim=5)
    x = torch.randn(4, 6, requires_grad=True)
    output = head(x)
    assert output.shape == (4, 8)
    output.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in head.parameters())


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_original_output_preserved_and_patch_removable(model, mode, distance):
    ids = torch.tensor([[1, 3, 5, 7]])
    original = model(ids).logits
    signature = inspect.signature(model.forward)
    state = patch(model, config(mode=mode, distance=distance))
    assert model.prerouter_state is state
    assert inspect.signature(model.forward) == signature
    torch.testing.assert_close(model(ids).logits, original, rtol=0, atol=0)
    assert set(state.router_logits) == set(state.predictions) == set(range(distance, 4))
    assert all(layer.mlp.prerouter_state is state for layer in model.layers)
    unpatch(model)
    assert "forward" not in model.__dict__
    assert not hasattr(model, "prerouter_state")
    assert all("forward" not in layer.mlp.__dict__ and not hasattr(layer.mlp, "prerouter")
               for layer in model.layers)
    torch.testing.assert_close(model(ids).logits, original, rtol=0, atol=0)


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_reference_model_moe_forward_and_routes(model, mode, distance):
    # Execute the four self-contained reference classes at a small dimension.
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "assets" / "modeling.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"OpenPXXV2MLP", "OpenPXXV2Experts", "OpenPXXV2TopkRouter", "OpenPXXV2SparseMoeBlock"}
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)
                               and node.name in names], type_ignores=[])
    namespace = dict(torch=torch, nn=nn, F=torch.nn.functional,
                     ACT2FN={"silu": torch.nn.functional.silu}, use_experts_implementation=lambda cls: cls)
    exec(compile(selected, str(path), "exec"), namespace)
    cfg = SimpleNamespace(hidden_size=6, intermediate_size=4, moe_intermediate_size=4,
                          hidden_act="silu", n_routed_experts=8, n_shared_experts=1,
                          num_experts_per_tok=2, topk_group=1, norm_topk_prob=True,
                          routed_scaling_factor=2.5)
    for layer in model.layers:
        layer.mlp = namespace["OpenPXXV2SparseMoeBlock"](cfg).requires_grad_(False)
        for parameter in layer.mlp.parameters():
            nn.init.normal_(parameter, std=0.1)
    values = batch()
    original = model(values["input_ids"]).logits
    state = patch(model, config(mode=mode, distance=distance))
    task = TrainingTask(model, state, "router")
    task(values, record_metrics=True).backward()
    torch.testing.assert_close(model(values["input_ids"]).logits, original, rtol=0, atol=0)
    for target, logits in state.router_logits.items():
        expected = model.layers[target].mlp.route_tokens_to_experts(logits)[0]
        torch.testing.assert_close(state.router_indices[target], expected, rtol=0, atol=0)


def test_exception_clears_routing_state(model, monkeypatch):
    state = patch(model, config())
    def fail(*args):
        raise RuntimeError("expert failure")
    monkeypatch.setattr(model.layers[1].mlp.experts, "forward", fail)
    with pytest.raises(RuntimeError, match="expert failure"):
        model(input_ids=torch.tensor([[1, 3, 5]]))
    assert not state.active
    assert not state.predictions and not state.next_predictions and not state.router_logits


@pytest.mark.parametrize("generation", [False, True])
def test_patch_enforces_batch_size_one(model, generation):
    state = patch(model, config())
    state.reset(generation=generation)
    with pytest.raises(ValueError, match="batch size must be 1"):
        model(input_ids=torch.tensor([[1, 3], [5, 7]]))


@pytest.mark.parametrize("stage", ["router", "lora"])
def test_training_enforces_batch_size_one(model, stage):
    state = patch(model, config()) if stage == "router" else None
    task = TrainingTask(model, state, stage)
    with pytest.raises(ValueError, match="batch size 1"):
        task(dict(input_ids=torch.ones(2, 3, dtype=torch.long)))


@pytest.mark.parametrize("distance", [1, 2])
def test_publish_before_experts_and_target_routing(model, distance, monkeypatch):
    state = patch(model, config(distance=distance))
    published, consumed = [], []
    for index, (_, block) in enumerate(state.blocks):
        original = block.route_tokens_to_experts
        def route(logits, index=index, original=original):
            if index in state.targets:
                assert index in state.predictions
                consumed.append(index)
            return original(logits)
        monkeypatch.setattr(block, "route_tokens_to_experts", route)
    handles = []
    for source, target in state.pairs:
        def before_experts(module, args, source=source, target=target):
            assert target in state.predictions
            published.append(source)
        handles.append(state.blocks[source][1].experts.register_forward_pre_hook(before_experts))
    model(input_ids=torch.tensor([[1, 3, 5]]))
    assert published == [source for source, _ in state.pairs]
    assert consumed == [target for _, target in state.pairs]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_full_sequence_equals_prefill_then_incremental(model, mode, distance):
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    state = patch(model, config(mode=mode, distance=distance, trace_limit=20))
    meter = RoutingMetrics(state)
    model(input_ids=ids)
    meter.update(state)
    full = meter.counts[0].clone()
    meter.reset()
    with capture_generation(model, state, meter):
        model(input_ids=ids[:, :2], attention_mask=torch.ones(1, 2))
        for i in range(2, ids.shape[1]):
            model(input_ids=ids[:, i:i + 1], attention_mask=torch.ones(1, i + 1))
    torch.testing.assert_close(full, meter.counts[1:].sum(0), rtol=0, atol=0)
    if mode == "previous_token":
        first = next(row for row in meter.trace if row["phase"] == "decode")
        assert first["source_token"] == 1 and first["target_token"] == 2
        assert first["forward_index"] == 1
        assert first["target_moe"] - first["source_moe"] == distance
    assert not model._forward_hooks


@pytest.mark.parametrize("distance", [0, 1, 2])
def test_cross_token_handoff_keeps_current_and_next_predictions(model, distance):
    state = patch(model, config(mode="previous_token", distance=distance))
    state.reset(generation=True)
    model(input_ids=torch.tensor([[1, 3]]))
    prior = dict(state.next_predictions)
    model(input_ids=torch.tensor([[5]]), attention_mask=torch.ones(1, 3))
    for target in state.targets:
        assert state.predictions[target] is prior[target]
        assert state.next_predictions[target] is not prior[target]
        assert state.next_predictions[target].shape == (1, 8)
        assert not state.next_predictions[target].requires_grad


@pytest.mark.parametrize("distance", [0, 1, 2])
def test_reset_prevents_cross_request_pairing(model, distance):
    state = patch(model, config(mode="previous_token", distance=distance))
    meter = RoutingMetrics(state)
    for _ in range(2):
        with capture_generation(model, state, meter):
            model(input_ids=torch.tensor([[1]]))
    assert int(meter.counts.sum()) == 0
    with capture_generation(model, state, meter):
        model(input_ids=torch.tensor([[1]]))
        model(input_ids=torch.tensor([[2]]))
    assert int(meter.counts[2, :, 1].sum()) == len(state.pairs)


@pytest.mark.parametrize("distance", [0, 1, 2])
def test_padding_exclusions_and_target_mask(model, distance):
    state = patch(model, config(mode="previous_token", distance=distance, excluded_token_ids=[3]))
    task = TrainingTask(model, state, "router")
    values = batch((0, 1, 2, 3, 4, 0))
    values["router_mask"] = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.bool)
    task(values, record_metrics=True).backward()
    rows = task.meter.report()["phases"]["teacher_forcing"]["global"]
    assert rows[-1]["token_layer_pairs"] == 2 * len(state.pairs)
    assert rows[-1]["recall"] == rows[-1]["full_coverage"] == 1
    for name, param in model.named_parameters():
        assert (param.grad is not None) == (".prerouter." in name)


@pytest.mark.parametrize("distance", [0, 1, 2])
def test_zero_valid_tokens_produce_connected_zero_loss(model, distance):
    state = patch(model, config(mode="previous_token", distance=distance))
    task = TrainingTask(model, state, "router")
    loss = task(batch((1,)), record_metrics=True)
    loss.backward()
    assert loss.item() == 0
    assert all(p.grad is not None for p in state.parameters())


def test_metrics_read_actual_selection_without_rerouting(model, monkeypatch):
    state = patch(model, config(targets=[1]))
    block = model.layers[1].mlp
    block.e_score_correction_bias[0] = 3.0
    model(input_ids=torch.tensor([[1, 3, 5]]))
    assert (state.router_indices[1][:, 0] == 0).all()
    def fail(logits):
        raise AssertionError("metrics must read collected expert indices")
    monkeypatch.setattr(block, "route_tokens_to_experts", fail)
    meter = RoutingMetrics(state)
    meter.update(state)
    assert meter.report()["phases"]["teacher_forcing"]["global"][-1]["full_coverage"] == 1


@pytest.mark.parametrize("kind", ["score_kl", "logit_kl"])
def test_kl_teacher_detached(model, kind):
    pred = torch.randn(5, 8, requires_grad=True)
    teacher = torch.randn(5, 8, requires_grad=True)
    loss = router_loss(pred, teacher, model.layers[0].mlp, 0.1, kind)
    loss.backward()
    assert teacher.grad is None
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
@pytest.mark.parametrize("head", ["linear", "mlp"])
@pytest.mark.parametrize("loss_kind", ["score_kl", "logit_kl"])
def test_loss_and_gradients_match_independent_reference(model, mode, distance, head, loss_kind):
    cfg = PrefetchConfig(mode=mode, distance=distance, head=head, hidden_dim=5, loss=loss_kind, ks=[2, 8])
    values = batch((0, 1, 3, 5, 0))
    values["router_mask"] = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    captured, handles = {}, []
    for index, layer in enumerate(model.layers):
        def capture(gate, args, logits, index=index):
            captured[index] = (args[0][0].detach(), logits.detach())
        handles.append(layer.mlp.gate.register_forward_hook(capture))
    state = patch(model, cfg)
    task = TrainingTask(model, state, "router")
    actual = task(values)
    actual.backward()
    gradients = [parameter.grad.clone() for parameter in state.parameters()]
    assert all(g.abs().sum() > 0 for g in gradients)
    assert model.embed.weight.grad is None
    assert all(not x.requires_grad for x in state.router_logits.values())
    for parameter in state.parameters():
        parameter.grad = None
    for handle in handles:
        handle.remove()

    losses = []
    for source, target in state.pairs:
        prediction = state.prerouters[source](captured[source][0])
        teacher = captured[target][1]
        mask = values["router_mask"][0] & values["attention_mask"][0]
        if mode == "previous_token":
            prediction, teacher = prediction[:-1], teacher[1:]
            mask = mask[1:] & values["attention_mask"][0, :-1]
        prediction, teacher = prediction[mask], teacher[mask]
        if loss_kind == "score_kl":
            bias = model.layers[target].mlp.e_score_correction_bias
            prediction, teacher = prediction.sigmoid() + bias, teacher.sigmoid() + bias
        log_q = (prediction / cfg.temperature).log_softmax(-1)
        log_p = (teacher / cfg.temperature).log_softmax(-1)
        losses.append((log_p.exp() * (log_p - log_q)).sum(-1).mean())
    expected = torch.stack(losses).mean()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for parameter, gradient in zip(state.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)


def test_forward_collects_data_for_external_loss_and_metrics(model):
    state = patch(model, config())
    meter = RoutingMetrics(state)
    state.reset(train_prerouter=True)
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 3, 5]]))
    assert int(meter.counts.sum()) == 0
    loss = compute_prerouter_loss(state, torch.ones(1, 3, dtype=torch.bool))
    loss.backward()
    meter.update(state)
    assert all(p.grad is not None for p in state.parameters())
    assert int(meter.counts.sum()) > 0


def test_eval_creates_no_head_graph(model):
    state = patch(model, config())
    task = TrainingTask(model, state, "router")
    with torch.no_grad():
        loss = task(batch(), record_metrics=True)
    assert not loss.requires_grad
    assert all(not x.requires_grad for x in state.predictions.values())
    assert task.meter.report()["phases"]["teacher_forcing"]["global"][-1]["recall"] == 1


def test_trainable_parameter_order_and_source_ownership(model):
    state = patch(model, config(targets=[3, 1]))
    assert list(state.prerouters) == [2, 0]
    assert model.layers[2].mlp.prerouter is state.prerouters[2]
    assert model.layers[0].mlp.prerouter is state.prerouters[0]
    expected = list(state.prerouters[2].parameters()) + list(state.prerouters[0].parameters())
    task = TrainingTask(model, state, "router")
    assert list(map(id, task.parameters())) == list(map(id, expected))


@pytest.mark.parametrize("head", ["linear", "mlp"])
@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_checkpoint_roundtrip_and_target_keys(model, tmp_path, head, mode, distance):
    pytest.importorskip("safetensors")
    targets = [3, distance]
    state = patch(model, PrefetchConfig(mode=mode, distance=distance, head=head, hidden_dim=5,
                                       targets=targets, ks=[2, 8]))
    with torch.no_grad():
        next(state.parameters()).add_(0.5)
    weights = {k: v.clone() for k, v in predictor_state_dict(state).items()}
    assert all(int(key.split(".")[0]) in targets and ".net." not in key for key in weights)
    save_predictor(state, tmp_path)
    unpatch(model)
    restored = patch(model, checkpoint=tmp_path)
    for key, value in predictor_state_dict(restored).items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_target_key_checkpoint_loading(model, tmp_path):
    pytest.importorskip("safetensors")
    from dataclasses import asdict
    import json
    from safetensors.torch import save_file
    state = patch(model, config(targets=[3, 1]))
    weights = {f"{target}.weight": torch.randn(8, 6) for _, target in state.pairs}
    save_file(weights, str(tmp_path / "predictor.safetensors"))
    (tmp_path / "prefetch_config.json").write_text(json.dumps({"config": asdict(state.config),
                                                               "layers": state.layers}))
    unpatch(model)
    restored = patch(model, checkpoint=tmp_path)
    for name, value in predictor_state_dict(restored).items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_attention_lora_zero_init_and_save(model, tmp_path):
    pytest.importorskip("safetensors")
    ids = torch.tensor([[1, 3, 5]])
    original = model(ids).logits
    lora = dict(rank=2, alpha=4, dropout=0.0)
    names = install_lora(model, lora)
    assert len(names) == 8
    torch.testing.assert_close(model(ids).logits, original, rtol=0, atol=0)
    model(ids).loss.backward()
    assert model.layers[0].self_attn.qkv_proj.lora_B.grad.abs().sum() > 0
    save_lora(model, tmp_path, lora)
    load_lora(model, tmp_path)


def _ddp_worker(process_rank, init_url, mode, distance):
    from contextlib import nullcontext
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from prefetch.training.runtime import EvalShard
    dist.init_process_group("gloo", init_method=init_url, rank=process_rank, world_size=2)
    torch.manual_seed(19)
    model = ToyModel().eval().requires_grad_(False)
    state = patch(model, config(mode=mode, distance=distance))
    task = TrainingTask(model, state, "router")
    ddp = DistributedDataParallel(task, broadcast_buffers=False)
    optimizer = torch.optim.SGD(task.parameters(), lr=0.01)
    values = batch()
    values["router_mask"] = torch.full((1, 3), process_rank == 0, dtype=torch.bool)
    for micro in range(2):
        with ddp.no_sync() if micro == 0 else nullcontext():
            (ddp(values, record_metrics=True) / 2).backward()
    optimizer.step()
    weights = torch.cat([p.detach().flatten() for p in task.parameters()])
    replicas = [torch.empty_like(weights) for _ in range(2)]
    dist.all_gather(replicas, weights)
    torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    assert list(EvalShard(range(3))) == ([0, 2] if process_rank == 0 else [1])
    report = task.meter.report(distributed=True)
    expected = 2 * (3 if mode == "same_token" else 2) * len(state.pairs)
    metrics = report["phases"]["teacher_forcing"]
    assert metrics["global"][-1]["token_layer_pairs"] == expected
    required = metrics["required_k"]["global"]
    assert required["token_layer_pairs"] == expected
    assert sum(row["count"] for row in required["distribution"]) == expected
    assert required["distribution"][-1]["cdf"] == 1
    for row in metrics["global"]:
        assert row["full_coverage"] == required["distribution"][row["k"] - 1]["cdf"]
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_DDP_TESTS") != "1", reason="Set RUN_DDP_TESTS=1 for the two-process test")
@pytest.mark.parametrize("mode,distance", ROUTING_CASES)
def test_two_process_trainable_ddp_and_count_reduction(tmp_path, mode, distance):
    torch.multiprocessing.spawn(_ddp_worker, args=((tmp_path / "gloo_init").resolve().as_uri(), mode, distance),
                                nprocs=2, join=True)
