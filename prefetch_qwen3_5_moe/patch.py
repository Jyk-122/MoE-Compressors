"""patch(model) 将各层 MoE 的 forward 绑定到本文件的 moe_forward。"""
from functools import wraps
from inspect import signature
import json
from types import MethodType

import torch

from .router_cache import RouterCache


def moe_blocks(model):
    """找到文本模型的各层 MoE。"""
    backbone = model
    if hasattr(backbone, "model"):
        backbone = backbone.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model

    blocks = []
    for layer in backbone.layers:
        blocks.append(layer.mlp)
    return blocks


def moe_forward(self, hidden_states):
    """调用 model(...) 时，每层 MoE 会执行这里的预测、存储和专家计算。"""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    cache = self.router_cache
    hidden = hidden_states.reshape(-1, hidden_dim)

    # 1. 确定本次输入的起始 token 位置和有效 token。
    # 例如 token_start=8、sequence_length=1，表示本次处理绝对位置 8。
    token_start = cache.token_start

    # 模型的 forward wrapper 已准备好本次输入的 bool mask。
    attention_mask = cache.attention_mask

    # 2. 用当前层的 hidden states 预测未来位置，并保存到目标位置。
    if self.gate_prefetch is not None:
        layers_ahead, tokens_ahead = self.mode_prefetch
        prefetch_logits, prefetch_scores, prefetch_indices = self.gate_prefetch(hidden.detach().float())

        prefetch_logits = prefetch_logits.reshape(batch_size, sequence_length, -1)
        prefetch_scores = prefetch_scores.reshape(batch_size, sequence_length, -1)
        prefetch_indices = prefetch_indices.reshape(batch_size, sequence_length, -1)

        target_token_start = token_start + tokens_ahead
        target_layer_id = self.layer_id + layers_ahead
        cache.save_prefetch(
            token_start=target_token_start,
            layer_id=target_layer_id,
            router_logits=prefetch_logits,
            router_scores=prefetch_scores,
            router_indices=prefetch_indices,
            valid_mask=attention_mask,
        )

    # 3. 当前层一次读取整段预测，专家编号形状为 [batch, sequence, top_k]。
    prediction = cache.get_prefetch_range(token_start, sequence_length, self.layer_id)
    if prediction is None:
        self.prefetch_indices = None
        self.prefetch_valid_mask = None
    else:
        self.prefetch_indices = prediction.router_indices
        self.prefetch_valid_mask = prediction.valid_mask

    # 4. 计算当前层的真实路由，保存到当前 token、当前层。
    shared_output = self.shared_expert(hidden)
    router_logits, router_scores, router_indices = self.gate(hidden)

    gt_logits = router_logits.reshape(batch_size, sequence_length, -1)
    gt_scores = router_scores.reshape(batch_size, sequence_length, -1)
    gt_indices = router_indices.reshape(batch_size, sequence_length, -1)

    cache.save_gt(
        token_start=token_start,
        layer_id=self.layer_id,
        router_logits=gt_logits,
        router_scores=gt_scores,
        router_indices=gt_indices,
        valid_mask=attention_mask,
    )

    # 5. 用真实路由执行专家，返回这一层的输出。
    expert_output = self.experts(hidden, router_indices, router_scores)
    shared_output = torch.sigmoid(self.shared_expert_gate(hidden)) * shared_output
    output = expert_output + shared_output
    return output.reshape(batch_size, sequence_length, hidden_dim)


def wrap_generation(model):
    """generate 管请求，forward 管本轮输入；生成算法由原生 generate 执行。"""
    original_generate = model.generate
    original_forward = model.forward
    forward_signature = signature(original_forward)
    cache = model.router_cache

    @wraps(original_generate)
    def generate(*args, router_cache_max_tokens=2048, **kwargs):
        cache.clear(max_tokens=router_cache_max_tokens)
        return original_generate(*args, **kwargs)

    @wraps(original_forward)
    def forward(*args, **kwargs):
        arguments = forward_signature.bind(*args, **kwargs).arguments
        inputs = arguments.get("input_ids")
        if inputs is None:
            inputs = arguments["inputs_embeds"]
        batch_size, sequence_length = inputs.shape[:2]

        # 在原生 forward 更新 KV cache 之前，读取已处理的 token 数。
        token_start = 0
        past_key_values = arguments.get("past_key_values")
        if past_key_values is not None:
            token_start = int(past_key_values.get_seq_length())

        if arguments.get("use_cache") is False:
            # 训练或整段重算：本轮使用新缓存，容量包含预测的未来 token。
            max_tokens = token_start + sequence_length + model.mode_prefetch[1]
            cache.clear(max_tokens=max_tokens)

        attention_mask = arguments.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, sequence_length, device=inputs.device, dtype=torch.bool)
        else:
            if attention_mask.ndim != 2:
                raise ValueError("RouterCache generation expects a 2D attention_mask; use the default dynamic cache")
            attention_mask = attention_mask[:, -sequence_length:].bool()
        cache.token_start = token_start
        cache.attention_mask = attention_mask
        return original_forward(*args, **kwargs)

    # Transformers 会检查 forward 的参数；保留原方法的签名和 __wrapped__。
    generate.__signature__ = signature(original_generate)
    forward.__signature__ = forward_signature
    model.generate = generate
    model.forward = forward


def patch(model, mode_prefetch=(1, 0)):
    """安装预测头、MoE forward 和原生 generate 的缓存 wrapper。"""
    if hasattr(model, "router_cache"):
        raise ValueError("This model already has a RouterCache")

    mode_prefetch = tuple(mode_prefetch)
    if len(mode_prefetch) != 2:
        raise ValueError("mode_prefetch must be (layers_ahead, tokens_ahead)")
    for offset in mode_prefetch:
        if type(offset) is not int or offset < 0:
            raise ValueError("Prefetch offsets must be nonnegative integers")
    if mode_prefetch == (0, 0):
        raise ValueError("At least one prefetch offset must be positive")

    blocks = moe_blocks(model)
    layers_ahead, tokens_ahead = mode_prefetch
    if layers_ahead >= len(blocks):
        raise ValueError("layers_ahead must be smaller than the number of layers")
    for block in blocks:
        if block.gate.weight.is_meta:
            raise ValueError("Load all router weights onto a compute device before patching")

    model.eval()
    model.requires_grad_(False)

    cache = RouterCache(num_hidden_layers=len(blocks))
    model.router_cache = cache
    model.mode_prefetch = mode_prefetch

    for layer_id, block in enumerate(blocks):
        # 所有层共用这一份 cache。
        block.router_cache = cache
        block.layer_id = layer_id
        block.mode_prefetch = mode_prefetch

        # 每个源层的预测头，用目标层的真实 gate 权重初始化。
        block.gate_prefetch = None
        target_layer_id = layer_id + layers_ahead
        if target_layer_id < len(blocks):
            target_gate = blocks[target_layer_id].gate
            # 这里的 router_class 就是模型使用的 Qwen3_5MoeTopKRouter。
            router_class = type(target_gate)
            router_config = blocks[target_layer_id].shared_expert.config
            block.gate_prefetch = router_class(router_config)
            block.gate_prefetch = block.gate_prefetch.to(
                device=block.gate.weight.device, dtype=torch.float32)
            with torch.no_grad():
                block.gate_prefetch.weight.copy_(target_gate.weight)

        # 绑定后，模型执行这个 block 时就会进入 moe_forward
        patched_forward = MethodType(moe_forward, block)
        if hasattr(block, "_hf_hook"):
            # Accelerate 的设备分派 wrapper 会调用 _old_forward。
            block._old_forward = patched_forward
        else:
            block.forward = patched_forward
    if hasattr(model, "generate"):
        wrap_generation(model)
    return cache


def save_prefetch(model, path):
    """保存各源层的预测头权重，以及它们使用的 mode_prefetch。"""
    from safetensors.torch import save_file

    blocks = moe_blocks(model)
    heads = {}
    for layer_id, block in enumerate(blocks):
        if block.gate_prefetch is not None:
            weight = block.gate_prefetch.weight.detach().cpu().contiguous()
            heads[str(layer_id)] = weight

    metadata = {
        "mode_prefetch": json.dumps(model.mode_prefetch),
        "num_hidden_layers": str(len(blocks)),
    }
    save_file(heads, str(path), metadata=metadata)


def load_prefetch(model, path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        heads = {}
        for key in checkpoint.keys():
            heads[key] = checkpoint.get_tensor(key)

    blocks = moe_blocks(model)
    if int(metadata["num_hidden_layers"]) != len(blocks):
        raise ValueError("Checkpoint layer count does not match the model")

    mode_prefetch = json.loads(metadata["mode_prefetch"])
    cache = patch(model, mode_prefetch)

    expected_keys = set()
    for layer_id, block in enumerate(blocks):
        if block.gate_prefetch is not None:
            expected_keys.add(str(layer_id))
    if set(heads) != expected_keys:
        raise ValueError("Checkpoint head keys do not match mode_prefetch")

    for layer_id, block in enumerate(blocks):
        if block.gate_prefetch is not None:
            block.gate_prefetch.load_state_dict({"weight": heads[str(layer_id)]})
    return cache
