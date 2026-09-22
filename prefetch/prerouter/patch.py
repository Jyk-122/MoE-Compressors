"""Install Prerouter modules and explicit MoE/model forward patches."""
from __future__ import annotations

from functools import wraps
from types import MethodType

import torch

from prefetch.prerouter.block import Prerouter
from prefetch.prerouter.checkpoint import load_predictor, read_metadata
from prefetch.prerouter.configuration import PrefetchConfig
from prefetch.backbone.structure import find_moe_blocks
from prefetch.prerouter.state import PrerouterState


def moe_forward(self, hidden_states):
    """OpenPXX SparseMoeBlock forward, with routing capture and a prerouter branch."""
    state, index = self.prerouter_state, self.moe_index
    router_logits = self.gate(hidden_states)
    # Expert selection lives here. The current experiment uses the native router.
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    if state.active:
        state.record(index, router_logits, topk_indices)
        if index in state.prerouters:
            if hidden_states.shape[:2] != (1, state.sequence_length):
                raise ValueError("MoE input must match processor input_ids [1, sequence]")
            # Only the prerouter builds a training graph; the backbone stays frozen.
            with torch.set_grad_enabled(state.train_prerouter):
                prediction = self.prerouter(hidden_states[0].detach())
                state.publish(index, prediction)

    output = self.experts(hidden_states.view(-1, hidden_states.shape[-1]),
                          topk_indices, topk_weights).view_as(hidden_states)
    return output + self.shared_experts(hidden_states)


def patch_moe_block(block, index, state):
    block._prerouter_forward = ("forward" in block.__dict__, block.forward)
    block.prerouter_state, block.moe_index = state, index
    if index in state.prerouters:
        block.prerouter = state.prerouters[index]
    block.forward = MethodType(moe_forward, block)


def patch(model, config=None, checkpoint=None):
    """Return the global state; losses and metrics are computed by callers."""
    if hasattr(model, "prerouter_state"):
        raise ValueError("Model already has a prerouter patch")
    metadata = read_metadata(checkpoint) if checkpoint else None
    config = metadata["config"] if metadata else config
    config = PrefetchConfig(**config) if isinstance(config, dict) else config or PrefetchConfig()
    state = PrerouterState(find_moe_blocks(model), config)
    if any(block.n_group != 1 for _, block in state.blocks):
        raise ValueError("Prerouter ranking requires n_group=1")
    if len({layer["experts"] for layer in state.layers}) != 1:
        raise ValueError("Prediction targets must have the same expert count")
    for source, target in state.pairs:
        source_gate, target_gate = state.blocks[source][1].gate, state.blocks[target][1].gate
        head = Prerouter(source_gate.weight.shape[1], target_gate.weight.shape[0],
                         config.head, config.hidden_dim).to(device=source_gate.weight.device, dtype=torch.float32)
        if config.head == "linear":
            with torch.no_grad():
                head.net.weight.copy_(target_gate.weight)
        state.prerouters[source] = head
    if checkpoint:
        load_predictor(state, checkpoint, metadata)
    for index, (_, block) in enumerate(state.blocks):
        patch_moe_block(block, index, state)

    original_forward = model.forward
    model._prerouter_forward = ("forward" in model.__dict__, original_forward)
    model.prerouter_state = state

    @wraps(original_forward)
    def forward(*args, **kwargs):
        try:
            state.begin_forward(kwargs.get("input_ids", args[0] if args else None),
                                kwargs.get("attention_mask", args[1] if len(args) > 1 else None))
            return original_forward(*args, **kwargs)
        except Exception:
            state.reset()
            raise
        finally:
            state.active = False

    model.forward = forward
    return state


def unpatch(model):
    state = model.prerouter_state
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
