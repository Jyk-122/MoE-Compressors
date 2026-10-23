"""每层保存 [batch, max_tokens, ...] 张量，按 token 区间读写。"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class RouterResult:
    router_logits: torch.Tensor   # 整层为 [batch, tokens, experts]；单 token 查询为 [batch, experts]
    router_scores: torch.Tensor   # 整层为 [batch, tokens, top_k]
    router_indices: torch.Tensor  # 整层为 [batch, tokens, top_k]
    valid_mask: torch.Tensor      # 整层为 [batch, tokens]，未写入或 padding 位置为 False
    token_start: int              # 已写入区间的起点，包含此位置
    token_end: int                # 已写入区间的终点，不包含此位置

class RouterCache:
    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers
        self.clear()

    def clear(self, max_tokens=2048):
        """新 batch / 新请求开始时，指定可容纳的绝对 token 位置数。"""
        self.max_tokens = max_tokens
        self.cache_gt = [None] * self.num_hidden_layers
        self.cache_pre = [None] * self.num_hidden_layers
        self.token_start = 0
        self.attention_mask = None

    def _allocate_layer(self, token_start, router_logits, router_scores, router_indices):
        """该层第一次写入时，根据实际设备、dtype 和形状分配整层存储。"""
        batch_size = router_logits.shape[0]
        num_experts = router_logits.shape[2]
        top_k = router_indices.shape[2]
        return RouterResult(
            router_logits=router_logits.new_zeros(batch_size, self.max_tokens, num_experts),
            router_scores=router_scores.new_zeros(batch_size, self.max_tokens, top_k),
            router_indices=router_indices.new_full((batch_size, self.max_tokens, top_k), -1),
            valid_mask=torch.zeros(
                batch_size, self.max_tokens, device=router_logits.device, dtype=torch.bool),
            token_start=token_start,
            token_end=token_start,
        )

    def _check_write(self, layer_id, token_start, router_logits, valid_mask):
        if layer_id < 0 or layer_id >= self.num_hidden_layers:
            raise ValueError("layer_id is outside the model")
        if router_logits.ndim != 3:
            raise ValueError("router_logits must have shape [batch, sequence, experts]")
        if valid_mask.shape != router_logits.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, sequence]")
        sequence_length = router_logits.shape[1]
        if token_start < 0 or sequence_length == 0:
            raise ValueError("Write a nonempty sequence at a nonnegative token_start")
        if token_start + sequence_length > self.max_tokens:
            raise ValueError("Cache capacity exceeded; call clear(max_tokens=...) before the request")

    def save_gt(self, token_start, layer_id, router_logits, router_scores,
                router_indices, valid_mask):
        """一次保存当前层的整段真实路由，输入形状为 [batch, sequence, ...]。"""
        self._check_write(layer_id, token_start, router_logits, valid_mask)
        if self.cache_gt[layer_id] is None:
            self.cache_gt[layer_id] = self._allocate_layer(
                token_start, router_logits, router_scores, router_indices)
        layer = self.cache_gt[layer_id]
        if token_start != layer.token_end:
            raise ValueError("Append ground-truth routes at the previous token_end")
        if router_logits.shape[0] != layer.router_logits.shape[0]:
            raise ValueError("Batch size must stay fixed within a request")

        token_end = token_start + router_logits.shape[1]
        layer.router_logits[:, token_start:token_end] = router_logits.detach()
        layer.router_scores[:, token_start:token_end] = router_scores.detach()
        layer.router_indices[:, token_start:token_end] = router_indices.detach()
        layer.valid_mask[:, token_start:token_end] = valid_mask
        layer.token_end = token_end

    def save_prefetch(self, token_start, layer_id, router_logits, router_scores,
                      router_indices, valid_mask):
        """一次保存整段预测；token_start 和 layer_id 已由调用方换算成目标位置。"""
        self._check_write(layer_id, token_start, router_logits, valid_mask)
        if self.cache_pre[layer_id] is None:
            self.cache_pre[layer_id] = self._allocate_layer(
                token_start, router_logits, router_scores, router_indices)
        layer = self.cache_pre[layer_id]
        if token_start != layer.token_end:
            raise ValueError("Append predictions at the previous token_end")
        if router_logits.shape[0] != layer.router_logits.shape[0]:
            raise ValueError("Batch size must stay fixed within a request")

        token_end = token_start + router_logits.shape[1]
        # PyTorch 会记录切片赋值的梯度，loss.backward() 可回到 gate_prefetch。
        layer.router_logits[:, token_start:token_end] = router_logits
        layer.router_scores[:, token_start:token_end] = router_scores.detach()
        layer.router_indices[:, token_start:token_end] = router_indices.detach()
        layer.valid_mask[:, token_start:token_end] = valid_mask
        layer.token_end = token_end

    def get_gt(self, token_idx, layer_id):
        """查询一个绝对 token 位置，返回 [batch, ...]。"""
        layer = self.cache_gt[layer_id]
        if layer is None:
            return None
        if token_idx < layer.token_start or token_idx >= layer.token_end:
            return None
        return RouterResult(
            router_logits=layer.router_logits[:, token_idx],
            router_scores=layer.router_scores[:, token_idx],
            router_indices=layer.router_indices[:, token_idx],
            valid_mask=layer.valid_mask[:, token_idx],
            token_start=token_idx,
            token_end=token_idx + 1,
        )

    def get_prefetch(self, token_idx, layer_id):
        """查询一个绝对 token 位置，返回 [batch, ...]。"""
        layer = self.cache_pre[layer_id]
        if layer is None:
            return None
        if token_idx < layer.token_start or token_idx >= layer.token_end:
            return None
        return RouterResult(
            router_logits=layer.router_logits[:, token_idx],
            router_scores=layer.router_scores[:, token_idx],
            router_indices=layer.router_indices[:, token_idx],
            valid_mask=layer.valid_mask[:, token_idx],
            token_start=token_idx,
            token_end=token_idx + 1,
        )

    def get_prefetch_range(self, token_start, sequence_length, layer_id):
        """MoE 一次读取整段预测；用 valid_mask 判断各位置是否有效。"""
        token_end = token_start + sequence_length
        if token_start < 0 or sequence_length <= 0 or token_end > self.max_tokens:
            raise ValueError("Requested token range is outside cache capacity")
        layer = self.cache_pre[layer_id]
        if layer is None:
            return None
        return self._slice_layer(layer, token_start, token_end)

    def _slice_layer(self, layer, start, end):
        """以绝对位置切片整层缓存，返回 [batch, end-start, ...] 视图。"""
        return RouterResult(
            router_logits=layer.router_logits[:, start:end],
            router_scores=layer.router_scores[:, start:end],
            router_indices=layer.router_indices[:, start:end],
            valid_mask=layer.valid_mask[:, start:end],
            token_start=start,
            token_end=end,
        )

    def matched(self):
        """按层返回预测与真实结果的重合区间，张量形状为 [batch, tokens, ...]。"""
        for layer_id in range(self.num_hidden_layers):
            prediction = self.cache_pre[layer_id]
            teacher = self.cache_gt[layer_id]
            if prediction is None or teacher is None:
                continue

            start = max(prediction.token_start, teacher.token_start)
            end = min(prediction.token_end, teacher.token_end)
            if start < end:
                prediction_range = self._slice_layer(prediction, start, end)
                teacher_range = self._slice_layer(teacher, start, end)
                yield layer_id, prediction_range, teacher_range

    @torch.no_grad()
    def prefetch_hit_stats(self, prefetch_k=(8, 12, 16, 32), router_mask=None,
                           mode="logits", lambda_=None, temperature=1.0):
        """统计平均命中数量；指定 lambda_ 时，融合预测与同层上一 token 的真实路由。"""
        if mode not in ("logits", "prob"):
            raise ValueError("mode must be logits or prob")
        if lambda_ is not None:
            if not math.isfinite(lambda_) or not 0 <= lambda_ <= 1:
                raise ValueError("lambda_ must be between 0 and 1")
        if mode == "prob" and (not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Fusion temperature must be finite and positive")
        total_hits = {}
        for k in prefetch_k:
            if type(k) is not int or k <= 0:
                raise ValueError("prefetch_k must contain positive integers")
            total_hits[k] = 0
        if not total_hits:
            raise ValueError("prefetch_k must not be empty")

        max_k = max(total_hits)
        num_positions = 0
        for layer_id, prediction, teacher in self.matched():
            device = prediction.router_logits.device
            num_experts = prediction.router_logits.shape[-1]
            if max_k > num_experts:
                raise ValueError(f"prefetch_k={max_k} exceeds layer {layer_id}'s {num_experts} experts")

            if lambda_ is not None:
                # 目标 t 对齐同层真实路由 t-1；所有 lambda 使用同一组有效位置。
                full_teacher = self.cache_gt[layer_id]
                start = max(teacher.token_start, full_teacher.token_start + 1)
                end = teacher.token_end
                if start >= end:
                    continue
                prediction = self._slice_layer(self.cache_pre[layer_id], start, end)
                teacher = self._slice_layer(full_teacher, start, end)
                previous_token = self._slice_layer(full_teacher, start - 1, end - 1)

            # 源位置和目标位置都有效时，才计入统计。
            valid = prediction.valid_mask & teacher.valid_mask.to(device)
            if lambda_ is not None:
                valid = valid & previous_token.valid_mask.to(device)
            if router_mask is not None:
                target_mask = router_mask[:, teacher.token_start:teacher.token_end]
                valid = valid & target_mask.to(device=device, dtype=torch.bool)
            logits = prediction.router_logits[valid]
            if logits.shape[0] == 0:
                continue
            scores = logits
            if lambda_ is not None:
                layer_logits = logits.float()
                token_logits = previous_token.router_logits.to(device)[valid].float()
                # 两个端点直接按原始 logits 排名，保留完整的专家排序。
                if lambda_ == 1:
                    scores = layer_logits
                elif lambda_ == 0:
                    scores = token_logits
                elif mode == "logits":
                    scores = lambda_ * layer_logits + (1 - lambda_) * token_logits
                else:
                    # log(λ*p_layer + (1-λ)*p_token)：排名相同，同时避免小概率下溢。
                    layer_log_probs = (layer_logits / temperature).log_softmax(dim=-1)
                    token_log_probs = (token_logits / temperature).log_softmax(dim=-1)
                    scores = torch.logaddexp(
                        math.log(lambda_) + layer_log_probs,
                        math.log1p(-lambda_) + token_log_probs,
                    )
            expected = teacher.router_indices.to(device)[valid]
            num_positions += logits.shape[0]

            # 一次取最大的 k'，较小的 k' 直接使用排名前缀。
            predicted = scores.topk(max_k, dim=-1, sorted=True).indices
            matched = (predicted.unsqueeze(-1) == expected.unsqueeze(-2)).any(dim=-1)
            hits_at_each_rank = matched.sum(dim=0)
            hits_within_top_k = hits_at_each_rank.cumsum(dim=0).tolist()
            for k in total_hits:
                total_hits[k] += hits_within_top_k[k - 1]

        avg_hits = {}
        for k in total_hits:
            avg_hits[k] = None
            if num_positions > 0:
                avg_hits[k] = total_hits[k] / num_positions
        return {
            "num_positions": num_positions,
            "total_hits": total_hits,
            "avg_hits": avg_hits,
        }

    def loss(self, router_mask=None, temperature=1.0):
        """每层整段计算 KL，再对有效的 batch/token/layer 位置取平均。"""
        total_loss = None
        total_valid_tokens = 0
        for layer_id, prediction, teacher in self.matched():
            device = prediction.router_logits.device
            valid_mask = prediction.valid_mask & teacher.valid_mask.to(device)
            if router_mask is not None:
                target_mask = router_mask[:, teacher.token_start:teacher.token_end]
                valid_mask = valid_mask & target_mask.to(device=device, dtype=torch.bool)
            prediction_logits = prediction.router_logits[valid_mask].float()
            teacher_logits = teacher.router_logits.to(device)[valid_mask].float()
            valid_tokens = prediction_logits.shape[0]
            if valid_tokens == 0:
                continue

            layer_loss = F.kl_div(
                (prediction_logits / temperature).log_softmax(dim=-1),
                (teacher_logits / temperature).log_softmax(dim=-1),
                reduction="sum",
                log_target=True,
            )
            if total_loss is None:
                total_loss = layer_loss
            else:
                total_loss = total_loss + layer_loss.to(total_loss.device)
            total_valid_tokens += valid_tokens

        if total_valid_tokens == 0:
            raise ValueError("No matched valid routes; provide a longer sequence or a smaller prefetch offset")
        return total_loss / total_valid_tokens
