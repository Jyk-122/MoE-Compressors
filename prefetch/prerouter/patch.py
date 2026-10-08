"""Install Prerouter modules, forward patches, and generation request lifecycle."""
from __future__ import annotations

from dataclasses import asdict
from functools import wraps
from inspect import signature
from types import MethodType

import torch

from prefetch.prerouter.block import Prerouter
from prefetch.prerouter.checkpoint import load_predictor, read_metadata
from prefetch.prerouter.configuration import PrefetchConfig
from prefetch.backbone.structure import find_moe_blocks
from prefetch.prerouter.routing import select_experts
from prefetch.prerouter.state import PrerouterState


def produce_prediction(block, hidden_states):
    """Produce a layer prediction, or retain source features for previous_top."""
    state, index = block.prerouter_state, block.moe_index
    if state.phase == "prefill" and state.config.mode == "same_token":
        return
    if hidden_states.shape[:2] != (1, state.sequence_length):
        raise ValueError("MoE input must match processor input_ids [1, sequence]")
    source = hidden_states[0].detach()
    if state.config.mode == "previous_top" and state.generation:
        state.source_hidden[index] = source[-1:].clone()
        return
    # Only the prerouter builds a training graph; the backbone stays frozen.
    with torch.set_grad_enabled(state.train_prerouter):
        if state.config.mode == "previous_top":
            prediction = block.prerouter(source[:-1], state.token_embeddings[1:])
        else:
            source = source[-1:] if state.phase == "prefill" else source
            prediction = block.prerouter(source)
        state.publish(index, prediction)


def prepare_token_predictions(state, embedding, input_ids):
    """Condition every previous_top head on the sampled token before decode layers."""
    with torch.no_grad():
        state.token_embeddings = embedding(input_ids)[0]
    if state.phase == "decode":
        with torch.set_grad_enabled(state.train_prerouter):
            for source, target in state.pairs:
                state.predictions[target] = state.prerouters[source](
                    state.source_hidden[source], state.token_embeddings)
        state.source_hidden = {}  # Consumed; this forward will save the next source features.


def moe_forward(self, hidden_states):
    """OpenPXX SparseMoeBlock forward, with routing capture and a prerouter branch."""
    state, index = self.prerouter_state, self.moe_index
    router_logits = self.gate(hidden_states)
    original_route = self.route_tokens_to_experts(router_logits)
    if state.active:
        state.record(index, router_logits, original_route[0])
        if index in state.prerouters:
            produce_prediction(self, hidden_states)

    topk_indices, topk_weights = original_route
    execution = state.config.execution
    if (state.active and index in state.predictions and
            (execution != "native" or state.route_observer is not None)):
        prediction = state.predictions[index]
        if state.phase == "decode":
            rows = torch.arange(router_logits.shape[0], device=router_logits.device)
        else:
            # Prompt rows keep native routing; response rows emulate decode execution.
            mask = state.valid_mask & state.router_mask[0, state.target_start:].bool()
            rows = mask.nonzero().flatten() + state.target_start
            prediction = prediction[mask]
        native = topk_indices[rows], topk_weights[rows]
        predicted = select_experts(self, router_logits[rows], native, prediction)
        if state.route_observer is not None:
            state.route_observer(index, self, hidden_states.reshape(-1, hidden_states.shape[-1])[rows],
                                 native, predicted)
        if execution != "native":
            indices, weights = predicted
            if execution == "compensated":
                weights = state.compensators[index](*native, indices, weights)
            topk_indices, topk_weights = topk_indices.clone(), topk_weights.clone()
            topk_indices[rows], topk_weights[rows] = indices, weights
    output = self.experts(hidden_states.view(-1, hidden_states.shape[-1]),
                          topk_indices, topk_weights).view_as(hidden_states)
    return output + self.shared_experts(hidden_states)


def patch_moe_block(block, index, state):
    block._prerouter_forward = ("forward" in block.__dict__, block.forward)
    block.prerouter_state, block.moe_index = state, index
    if index in state.prerouters:
        block.prerouter = state.prerouters[index]
    block.forward = MethodType(moe_forward, block)


def patch(model, config=None, checkpoint=None, *, prerouter_enabled=None,
          execution_mode=None, compensation=None):
    """Return the global state; losses and metrics are computed by callers."""
    if hasattr(model, "prerouter_state"):
        raise ValueError("Model already has a prerouter patch")
    # Execution policy can be changed while checkpoint architecture stays fixed.
    requested = config if isinstance(config, dict) else asdict(config) if config is not None else {}
    # An explicit legacy CLI switch also overrides a saved execution_mode.
    if execution_mode is None and prerouter_enabled is None:
        execution_mode = requested.get("execution_mode")
    if prerouter_enabled is None:
        prerouter_enabled = requested.get("prerouter_enabled")
    if compensation is None:
        compensation = requested.get("compensation")
    metadata = read_metadata(checkpoint) if checkpoint else None
    config = metadata["config"] if metadata else config
    config = PrefetchConfig(**config) if isinstance(config, dict) else config or PrefetchConfig()
    if prerouter_enabled is not None:
        config.prerouter_enabled = prerouter_enabled
        config.execution_mode = None
    if execution_mode is not None:
        config.execution_mode = execution_mode
    if compensation is not None:
        config.compensation = compensation
    state = PrerouterState(find_moe_blocks(model), config)
    if any(block.n_group != 1 for _, block in state.blocks):
        raise ValueError("Prerouter ranking requires n_group=1")
    if len({layer["experts"] for layer in state.layers}) != 1:
        raise ValueError("Prediction targets must have the same expert count")
    embedding = model.get_input_embeddings() if config.mode == "previous_top" else None
    embedding_dim = embedding.weight.shape[1] if embedding is not None else None
    for source, target in state.pairs:
        source_gate, target_gate = state.blocks[source][1].gate, state.blocks[target][1].gate
        head = Prerouter(source_gate.weight.shape[1], target_gate.weight.shape[0],
                         config.head, config.hidden_dim, embedding_dim=embedding_dim).to(
                             device=source_gate.weight.device, dtype=torch.float32)
        if config.head == "linear":
            with torch.no_grad():
                head.net.weight.copy_(target_gate.weight)
        state.prerouters[source] = head
    if checkpoint:
        load_predictor(state, checkpoint, metadata)
    if config.execution == "compensated":
        from prefetch.compensation.artifacts import build_compensators
        state.compensators = build_compensators(state)
    for index, (_, block) in enumerate(state.blocks):
        patch_moe_block(block, index, state)

    original_forward = model.forward
    model._prerouter_forward = ("forward" in model.__dict__, original_forward)
    model.prerouter_state = state

    @wraps(original_forward)
    def forward(*args, **kwargs):
        try:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            state.begin_forward(input_ids,
                                kwargs.get("attention_mask", args[1] if len(args) > 1 else None))
            if config.mode == "previous_top" and state.phase != "prefill":
                prepare_token_predictions(state, embedding, input_ids)
            return original_forward(*args, **kwargs)
        except Exception:
            state.reset()
            raise
        finally:
            state.active = False

    model.forward = forward
    if hasattr(model, "generate"):
        original_generate = model.generate
        model._prerouter_generate = ("generate" in model.__dict__, original_generate)

        @wraps(original_generate)
        def generate(*args, **kwargs):
            state.reset(generation=True)
            try:
                return original_generate(*args, **kwargs)
            finally:
                state.reset()

        # Keep the bound signature through decorators such as torch.no_grad.
        generate.__signature__ = signature(original_generate)
        model.generate = generate
    return state


def unpatch(model):
    state = model.prerouter_state
    if hasattr(model, "_prerouter_generate"):
        had_override, original = model._prerouter_generate
        if had_override:
            model.generate = original
        else:
            del model.generate
        del model._prerouter_generate
    for module in [model] + [block for _, block in state.blocks]:
        had_override, original = module._prerouter_forward
        if had_override:
            module.forward = original
        else:
            del module.forward
        del module._prerouter_forward, module.prerouter_state
    for source, _ in state.pairs:
        del state.blocks[source][1].prerouter
    for _, block in state.blocks:
        del block.moe_index
    state.reset()
