from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.expert_trace_io import (
    write_forward_block,
    write_sample_start,
    write_trace_header,
)


class MoEStatsCollector:
    """Collect runtime MoE routing stats for eval-time reports (pruning acceleration / skipping)."""
    STAGE_KEYS = ("prefill", "decode")

    def __init__(self, num_experts: int):
        self.num_experts = int(num_experts)
        self._layers: dict[int, dict[str, Any]] = {}
        self._stage_layers: dict[str, dict[int, dict[str, Any]]] = {
            key: {} for key in self.STAGE_KEYS
        }
        self._active_attention_mask: torch.Tensor | None = None

        self._expert_trace_path: Path | None = None
        self._expert_trace_write: bool = False
        self._expert_trace_fp: Any = None
        self._trace_forward_buffer: dict[int, np.ndarray] = {}
        self._trace_forward_seq_len: int | None = None
        self._ordered_layer_indices: list[int] = []

    def initialize_layers(self, layer_indices: list[int]) -> None:
        self._ordered_layer_indices = sorted(int(i) for i in layer_indices)
        for layer_idx in layer_indices:
            self._ensure_layer(self._layers, int(layer_idx))

    def _ensure_layer(self, bucket: dict[int, dict[str, Any]], layer_idx: int):
        if layer_idx not in bucket:
            bucket[layer_idx] = {
                "expert_activation_count": torch.zeros(self.num_experts, dtype=torch.long),
                "total_tokens": 0,
                "total_selected_before": 0,
                "total_selected_after": 0,
            }
        return bucket[layer_idx]

    def _update_bucket(
        self,
        bucket: dict[int, dict[str, Any]],
        *,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        default_top_k: int,
    ) -> None:
        layer = self._ensure_layer(bucket, layer_idx)
        valid = selected_indices[selected_indices >= 0]
        if valid.numel() > 0:
            layer["expert_activation_count"] += torch.bincount(
                valid.cpu(),
                minlength=self.num_experts,
            )

        num_tokens = int(selected_indices.shape[0])
        selected_after = int((selected_indices >= 0).sum().item())
        layer["total_tokens"] += num_tokens
        layer["total_selected_before"] += num_tokens * int(default_top_k)
        layer["total_selected_after"] += selected_after

    def _resolve_stage(self, sequence_length: int | None) -> str | None:
        if sequence_length is None:
            return None
        # 生成时 sequence_length==1 通常表示增量 decode；>1 视作 prefill。
        return "decode" if int(sequence_length) == 1 else "prefill"

    def set_active_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        if attention_mask is None:
            self._active_attention_mask = None
            return
        # 存为 bool，后续按当前 sequence_length 取尾段并展平过滤 padding token。
        self._active_attention_mask = attention_mask.detach().to(dtype=torch.bool)

    def _apply_active_mask(
        self,
        selected_indices: torch.LongTensor,
        sequence_length: int | None,
    ) -> torch.LongTensor:
        mask = self._active_attention_mask
        if mask is None or sequence_length is None:
            return selected_indices
        if mask.dim() != 2:
            return selected_indices

        # seq_len = int(sequence_length)
        # if seq_len <= 0 or mask.shape[1] < seq_len:
        #     return selected_indices

        # # 对齐当前 forward 的最后 seq_len 个位置（兼容 generate decode 时 attention_mask 长于 input）。
        # token_mask = mask[:, -seq_len:].reshape(-1)
        # if token_mask.numel() != selected_indices.shape[0]:
        #     return selected_indices
        
        token_mask = mask.flatten().bool()
        return selected_indices[token_mask]

    def update(
        self,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        default_top_k: int,
        sequence_length: int | None = None,
    ) -> None:
        # 根据 _active_attention_mask 来判断是否进行统计
        if self._active_attention_mask is None:
            return

        selected_indices = self._apply_active_mask(selected_indices, sequence_length)
        self._trace_maybe_buffer_layer(layer_idx, selected_indices, sequence_length)
        self._update_bucket(
            self._layers,
            layer_idx=layer_idx,
            selected_indices=selected_indices,
            default_top_k=default_top_k,
        )
        stage = self._resolve_stage(sequence_length)
        if stage is not None:
            stage_bucket = self._stage_layers[stage]
            self._update_bucket(
                stage_bucket,
                layer_idx=layer_idx,
                selected_indices=selected_indices,
                default_top_k=default_top_k,
            )

    def _summary_from_layers(self, layers_store: dict[int, dict[str, Any]]) -> dict[str, Any]:
        layers = {}
        global_before = 0
        global_after = 0
        for layer_idx, info in layers_store.items():
            before = int(info["total_selected_before"])
            after = int(info["total_selected_after"])
            reduction = 0.0 if before == 0 else 1.0 - (after / before)
            layers[str(layer_idx)] = {
                "total_tokens": int(info["total_tokens"]),
                "total_selected_before": before,
                "total_selected_after": after,
                "activation_reduction_ratio": reduction,
                "expert_activation_count": info["expert_activation_count"].tolist(),
            }
            global_before += before
            global_after += after

        global_reduction = 0.0 if global_before == 0 else 1.0 - (global_after / global_before)
        return {
            "enabled": bool(layers_store),
            "global": {
                "total_selected_before": global_before,
                "total_selected_after": global_after,
                "activation_reduction_ratio": global_reduction,
                "effective_selected_per_token_ratio": 0.0
                if global_before == 0
                else (global_after / global_before),
            },
            "layers": layers,
        }

    def _store_to_tensors(
        self,
        layers_store: dict[int, dict[str, Any]],
        layer_indices: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_layers = len(layer_indices)
        tokens = torch.zeros(num_layers, dtype=torch.long, device=device)
        before = torch.zeros(num_layers, dtype=torch.long, device=device)
        after = torch.zeros(num_layers, dtype=torch.long, device=device)
        activation = torch.zeros((num_layers, self.num_experts), dtype=torch.long, device=device)

        for i, layer_idx in enumerate(layer_indices):
            info = layers_store.get(layer_idx)
            if info is None:
                continue
            tokens[i] = int(info["total_tokens"])
            before[i] = int(info["total_selected_before"])
            after[i] = int(info["total_selected_after"])
            activation[i] = info["expert_activation_count"].to(device=device, dtype=torch.long)
        return tokens, before, after, activation

    def _summary_from_tensors(
        self,
        *,
        layer_indices: list[int],
        tokens: torch.Tensor,
        before: torch.Tensor,
        after: torch.Tensor,
        activation: torch.Tensor,
    ) -> dict[str, Any]:
        layers = {}
        global_before = int(before.sum().item())
        global_after = int(after.sum().item())
        global_tokens = int(tokens.sum().item())

        for i, layer_idx in enumerate(layer_indices):
            layer_before = int(before[i].item())
            layer_after = int(after[i].item())
            reduction = 0.0 if layer_before == 0 else 1.0 - (layer_after / layer_before)
            layers[str(layer_idx)] = {
                "total_tokens": int(tokens[i].item()),
                "total_selected_before": layer_before,
                "total_selected_after": layer_after,
                "activation_reduction_ratio": reduction,
                "expert_activation_count": activation[i].tolist(),
            }

        global_reduction = 0.0 if global_before == 0 else 1.0 - (global_after / global_before)
        return {
            "enabled": bool(global_tokens > 0),
            "global": {
                "total_selected_before": global_before,
                "total_selected_after": global_after,
                "activation_reduction_ratio": global_reduction,
                "effective_selected_per_token_ratio": 0.0
                if global_before == 0
                else (global_after / global_before),
            },
            "layers": layers,
        }

    def distributed_summary(self) -> dict[str, Any]:
        layer_indices = sorted(self._layers.keys())
        if not layer_indices:
            return self.summary()

        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        overall = self._store_to_tensors(self._layers, layer_indices, device)
        stage_tensors = {
            stage: self._store_to_tensors(self._stage_layers[stage], layer_indices, device)
            for stage in self.STAGE_KEYS
        }

        dist = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist = torch.distributed

        # 暂时不使用多卡allreduce，存在卡死的情况
        # if dist is not None:
        #     for t in overall:
        #         dist.all_reduce(t, op=dist.ReduceOp.SUM)
        #     for stage in self.STAGE_KEYS:
        #         tensors = stage_tensors[stage]
        #         for t in tensors:
        #             dist.all_reduce(t, op=dist.ReduceOp.SUM)

        summary = self._summary_from_tensors(
            layer_indices=layer_indices,
            tokens=overall[0],
            before=overall[1],
            after=overall[2],
            activation=overall[3],
        )
        by_stage = {
            stage: self._summary_from_tensors(
                layer_indices=layer_indices,
                tokens=stage_tensors[stage][0],
                before=stage_tensors[stage][1],
                after=stage_tensors[stage][2],
                activation=stage_tensors[stage][3],
            )
            for stage in self.STAGE_KEYS
        }
        return self._reformat_axes(summary, by_stage)

    def summary(self) -> dict[str, Any]:
        overall = self._summary_from_layers(self._layers)
        by_stage = {
            stage: self._summary_from_layers(self._stage_layers[stage])
            for stage in self.STAGE_KEYS
        }
        return self._reformat_axes(overall, by_stage)

    def _reformat_axes(
        self,
        overall: dict[str, Any],
        by_stage: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "enabled": bool(overall.get("enabled", False)),
            "global": {
                "all": overall.get("global", {}),
                "prefill": by_stage["prefill"].get("global", {}),
                "decode": by_stage["decode"].get("global", {}),
            },
            "layers": {
                "all": overall.get("layers", {}),
                "prefill": by_stage["prefill"].get("layers", {}),
                "decode": by_stage["decode"].get("layers", {}),
            },
        }

    def enable_expert_trace(self, path: str | Path, *, write: bool = True) -> None:
        """Append routing trace to ``path`` (binary). If write=False, skip recording (non-main ranks)."""
        self._expert_trace_path = Path(path)
        self._expert_trace_write = bool(write)
        self._expert_trace_fp = None
        self._trace_forward_buffer.clear()
        self._trace_forward_seq_len = None
        if self._expert_trace_write:
            self._expert_trace_path.parent.mkdir(parents=True, exist_ok=True)

    def close_expert_trace(self) -> None:
        if self._expert_trace_fp is not None:
            try:
                self._expert_trace_fp.close()
            finally:
                self._expert_trace_fp = None
        self._expert_trace_path = None
        self._expert_trace_write = False
        self._trace_forward_buffer.clear()
        self._trace_forward_seq_len = None

    def _trace_ensure_file_open(self) -> None:
        if not self._expert_trace_write or self._expert_trace_path is None:
            return
        if self._expert_trace_fp is not None:
            return
        path = self._expert_trace_path
        new_file = not path.exists() or path.stat().st_size == 0
        self._expert_trace_fp = path.open("ab" if not new_file else "wb")
        if new_file:
            num_layers = len(self._ordered_layer_indices) if self._ordered_layer_indices else len(self._layers)
            write_trace_header(self._expert_trace_fp, num_experts=self.num_experts, num_layers=num_layers)

    def _trace_maybe_buffer_layer(
        self,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        sequence_length: int | None,
    ) -> None:
        if self._expert_trace_path is None or not self._expert_trace_write:
            return
        if selected_indices.numel() == 0:
            return
        arr = selected_indices.detach().cpu().numpy().astype(np.int16, copy=True)
        self._trace_forward_buffer[int(layer_idx)] = arr
        if sequence_length is not None:
            self._trace_forward_seq_len = int(sequence_length)

    def finalize_forward_step(self) -> None:
        """Call once per model forward after all MoE layers have run (e.g. in forward wrapper finally)."""
        if self._expert_trace_path is None or not self._expert_trace_write:
            self._trace_forward_buffer.clear()
            self._trace_forward_seq_len = None
            return
        if not self._trace_forward_buffer:
            self._trace_forward_seq_len = None
            return
        self._trace_ensure_file_open()
        assert self._expert_trace_fp is not None
        seq_len = self._trace_forward_seq_len if self._trace_forward_seq_len is not None else 1
        if seq_len > 1:
            write_sample_start(self._expert_trace_fp)
        layers_sorted = sorted(self._trace_forward_buffer.items(), key=lambda x: x[0])
        layers_payload = [(idx, arr) for idx, arr in layers_sorted]
        write_forward_block(self._expert_trace_fp, sequence_length=seq_len, layers=layers_payload)
        self._expert_trace_fp.flush()
        self._trace_forward_buffer.clear()
        self._trace_forward_seq_len = None


def build_router_prob_hist(router_probs: torch.Tensor, bins: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hist, cdf) on [0,1] for flattened router probabilities."""
    probs = router_probs.detach().float().reshape(-1).cpu()
    hist = torch.histc(probs, bins=bins, min=0.0, max=1.0)
    cdf = torch.cumsum(hist / hist.sum().clamp_min(1.0), dim=0)
    return hist, cdf
