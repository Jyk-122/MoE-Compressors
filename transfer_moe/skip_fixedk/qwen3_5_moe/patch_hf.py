"""Fixed-K using the native router, expert dispatch and shared-expert branch."""

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from ...utils import attach_activation_stats


def record_routes(gate, args, output):
    ids = output[2]
    gate.activation_stats.record_fixed(gate.activation_layer_id, ids.shape[0], ids.shape[1])


def patch(model, *, topk):
    """Set the routed expert count on a native HF model before compilation."""
    blocks = [module for module in model.modules()
              if isinstance(module, Qwen3_5MoeSparseMoeBlock)]
    if not blocks:
        raise ValueError("No Qwen3.5 MoE layers found")
    native_topk = model.config.get_text_config().num_experts_per_tok
    if type(topk) is not int or not 1 <= topk <= native_topk:
        raise ValueError(f"topk must be an integer in [1, {native_topk}]")
    stats = attach_activation_stats(
        model, [(index, block.gate, native_topk, block.gate.weight.device)
                for index, block in enumerate(blocks)])
    for block in blocks:
        block.gate.top_k = topk
        if hasattr(block.gate, "activation_stats_hook"):
            block.gate.activation_stats_hook.remove()
            del block.gate.activation_stats_hook
        if stats is not None:
            block.gate.activation_stats_hook = block.gate.register_forward_hook(record_routes)
    return model
