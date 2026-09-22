"""MoE discovery and native routing scores shared by patching and evaluation."""

def find_moe_blocks(model):
    blocks = [(name, module) for name, module in model.named_modules()
              if hasattr(module, "route_tokens_to_experts")
              and hasattr(module, "gate") and hasattr(module, "experts")]
    if not blocks:
        raise ValueError("No MoE blocks exposing gate/experts/route_tokens_to_experts found")
    return blocks


def choice_scores(logits, block):
    return logits.float().sigmoid() + block.e_score_correction_bias.detach().float()
