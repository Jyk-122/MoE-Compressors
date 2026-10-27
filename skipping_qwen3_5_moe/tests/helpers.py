"""Small causal models built from local reference Qwen MoE classes."""
from types import SimpleNamespace

import torch
from torch import nn

from .reference_moe import Qwen3_5MoeSparseMoeBlock
from ..calibrate import calibrate_outputs
from ..experts import all_expert_outputs


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(19, 6)
        cfg = SimpleNamespace(hidden_size=6, num_experts=5, num_experts_per_tok=4,
                              moe_intermediate_size=7, shared_expert_intermediate_size=7, hidden_act="silu")
        self.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Module()
            layer.mlp = Qwen3_5MoeSparseMoeBlock(cfg)
            layer.mix = nn.Linear(6, 6, bias=False)
            self.layers.append(layer)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.25)
        for layer in self.layers:
            nn.init.zeros_(layer.mlp.gate.weight)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        x = x.cumsum(1)
        for layer in self.layers:
            x = x + layer.mlp(x)
            x = x + layer.mix(x).tanh()
        return SimpleNamespace(last_hidden_state=x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(6, 19, bias=False)
        self.config = SimpleNamespace(_name_or_path="tiny-qwen")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.model(input_ids, attention_mask, **kwargs).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def tiny_generation_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=8, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
        num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    for layer in model.model.layers:
        layer.mlp = Qwen3_5MoeSparseMoeBlock(config)
        for parameter in layer.mlp.parameters():
            nn.init.normal_(parameter, std=0.2)
    return model


def calibration_for(model):
    from ..model import find_moe_layers
    result = {}
    for index, layer in find_moe_layers(model):
        block = layer.mlp
        h = torch.randn(32, block.experts.gate_up_proj.shape[-1])
        state = calibrate_outputs(all_expert_outputs(block.experts, h))
        state["top_k"] = block.gate.top_k
        result[index] = state
    return result


def batch():
    ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]])
    labels = ids.clone()
    labels[:, :3] = -100
    return {"input_ids": ids, "attention_mask": ids.ne(0), "labels": labels}
