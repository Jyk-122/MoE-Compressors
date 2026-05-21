#!/usr/bin/env python3
"""
MoE 校准数据集专家激活分析与子集选择

功能：
  analyze - 对数据集进行 prefill 推理，统计每层每个专家的激活频率，
            计算覆盖度测度（归一化熵、Coverage Ratio、基尼系数），并可视化。
  select  - 基于专家激活指纹聚类，从候选池中选出最优校准子集。
            支持 cluster_stratified / greedy_entropy / greedy_coverage 三种方法。

用法:
  python dataset_analysis.py analyze \\
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    --dataset DKYoon/SlimPajama-200k \\
    --max_samples 20000 \\
    --max_context_len 2048 \\
    --output_dir ./outputs/dataset_analysis

  python dataset_analysis.py select \\
    --activation_file ./outputs/dataset_analysis/activation_data.pt \\
    --num_samples 256 \\
    --method cluster_stratified \\
    --n_clusters 16 \\
    --output_dir ./outputs/dataset_analysis/selected
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_analysis")

_DEVICE_ALIASES: dict[str, str] = {"auto": "cuda" if torch.cuda.is_available() else "cpu"}


def _get_moe_layers(model) -> list[tuple[int, Qwen3MoeSparseMoeBlock]]:
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            moe_layers.append((i, layer.mlp))
    return moe_layers


def _resolve_num_experts(model) -> int:
    if hasattr(model.config, "num_experts"):
        return int(model.config.num_experts)
    moe_layers = _get_moe_layers(model)
    if moe_layers:
        return int(moe_layers[0][1].num_experts)
    raise RuntimeError("无法确定 num_experts")


# ============================================================
# ExpertActivationCollector
# ============================================================

class ExpertActivationCollector:
    def __init__(self):
        self._current: dict[int, torch.Tensor] | None = None
        self.per_sample: list[dict[int, np.ndarray]] = []

    def start_sample(self) -> None:
        self._current = {}

    def record(self, layer_idx: int, router_indices: torch.Tensor, num_experts: int) -> None:
        if self._current is None:
            return
        if layer_idx not in self._current:
            self._current[layer_idx] = torch.zeros(num_experts, dtype=torch.long)
        self._current[layer_idx] += torch.bincount(
            router_indices.flatten().cpu(), minlength=num_experts
        )

    def finish_sample(self) -> None:
        if self._current is not None:
            sample_np = {k: v.numpy().astype(np.int64) for k, v in self._current.items()}
            self.per_sample.append(sample_np)
        self._current = None

    def to_array(self, layer_indices: list[int]) -> np.ndarray:
        num_layers = len(layer_indices)
        num_samples = len(self.per_sample)
        num_experts = list(self.per_sample[0].values())[0].shape[0]
        arr = np.zeros((num_samples, num_layers, num_experts), dtype=np.int64)
        for i, sample in enumerate(self.per_sample):
            for j, lidx in enumerate(layer_indices):
                if lidx in sample:
                    arr[i, j, :] = sample[lidx]
        return arr


# ============================================================
# Metrics
# ============================================================

def compute_normalized_entropy(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    n = len(counts)
    if n <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(n))


def compute_coverage_ratio(counts: np.ndarray, threshold: float = 0.0) -> float:
    n = len(counts)
    if n == 0:
        return 0.0
    return float(np.sum(counts > threshold) / n)


def compute_gini(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    if n == 0 or sorted_counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n)


def compute_layer_metrics(
    activation_array: np.ndarray,
    coverage_threshold: float = 0.0,
) -> dict:
    num_layers = activation_array.shape[1]
    metrics: dict[str, list[float]] = {
        "normalized_entropy": [],
        "coverage_ratio": [],
        "gini": [],
    }
    for lidx in range(num_layers):
        layer_counts = activation_array[:, lidx, :].sum(axis=0)
        metrics["normalized_entropy"].append(compute_normalized_entropy(layer_counts))
        metrics["coverage_ratio"].append(compute_coverage_ratio(layer_counts, coverage_threshold))
        metrics["gini"].append(compute_gini(layer_counts))
    return metrics


# ============================================================
# Data Loading
# ============================================================

def load_and_chunk_dataset(
    dataset_name: str,
    tokenizer,
    max_samples: int,
    max_context_len: int,
) -> list[str]:
    parts = dataset_name.split(":", 1)
    dataset_name_clean = parts[0]
    config_name = parts[1] if len(parts) > 1 else None

    logger.info("Loading dataset: %s (config=%s)", dataset_name_clean, config_name)
    ds = load_dataset(dataset_name_clean, config_name, split="train")

    formatted_texts: list[str] = []
    if "instruction" in ds.column_names:
        logger.info("Detected Alpaca-style dataset, formatting with chat template")
        for item in ds:
            user_content = item["instruction"]
            if "input" in item and item["input"]:
                user_content += "\n" + item["input"]
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item.get("output", "")},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
    else:
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        formatted_texts = [str(t) for t in ds[col] if t and str(t).strip()]

    logger.info("Raw texts: %d, building chunks of ~%d tokens", len(formatted_texts), max_context_len)

    chunks: list[str] = []
    buf = ""
    pbar = tqdm(total=max_samples, desc="Building chunks", unit="chunk")
    for text in formatted_texts:
        buf += text
        n_tokens = len(tokenizer.encode(buf, add_special_tokens=False))
        if n_tokens >= max_context_len:
            chunks.append(buf)
            buf = ""
            pbar.update(1)
            if len(chunks) >= max_samples:
                break
    pbar.close()
    logger.info("Built %d chunks", len(chunks))
    return chunks


# ============================================================
# Analysis (Part 1)
# ============================================================

def run_analysis(
    model_name_or_path: str,
    dataset_name: str,
    output_dir: Path,
    max_samples: int = 20000,
    max_context_len: int = 2048,
    device: str = "auto",
    dtype: torch.dtype | None = None,
) -> None:
    device = _DEVICE_ALIASES.get(device, device)
    dtype = dtype or torch.bfloat16

    logger.info("Loading model: %s", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_and_chunk_dataset(dataset_name, tokenizer, max_samples, max_context_len)
    if not texts:
        raise RuntimeError("No text chunks built")

    model.eval()
    moe_layers = _get_moe_layers(model)
    num_experts = _resolve_num_experts(model)
    layer_indices = [lidx for lidx, _ in moe_layers]
    num_moe_layers = len(moe_layers)
    logger.info("MoE layers: %d, experts per layer: %d", num_moe_layers, num_experts)

    original_forwards: dict[int, any] = {}
    for lidx, block in moe_layers:
        original_forwards[lidx] = block.forward

    collector = ExpertActivationCollector()

    for lidx, block in moe_layers:
        original = original_forwards[lidx]

        def _make_forward(_lidx: int, _orig):
            def _forward(self_block, hidden_states: torch.Tensor):
                h_flat = hidden_states.view(-1, hidden_states.shape[-1])
                router_logits = F.linear(h_flat, self_block.gate.weight)
                router_probs = F.softmax(router_logits.float(), dim=-1)
                _, router_indices = torch.topk(router_probs, self_block.gate.top_k, dim=-1)
                collector.record(_lidx, router_indices, num_experts)
                return _orig(hidden_states)
            return _forward

        block.forward = types.MethodType(_make_forward(lidx, original), block)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running prefill inference on %d samples ...", len(texts))

    for idx, text in enumerate(tqdm(texts, desc="Prefill inference", unit="sample")):
        collector.start_sample()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_context_len
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        collector.finish_sample()

    for lidx, block in moe_layers:
        block.forward = original_forwards[lidx]

    activation_array = collector.to_array(layer_indices)
    activation_path = output_dir / "activation_data.npz"
    np.savez_compressed(
        activation_path,
        activation_array=activation_array,
        layer_indices=np.array(layer_indices, dtype=np.int32),
    )
    logger.info("Activation data saved to %s (shape=%s)", activation_path, activation_array.shape)

    metrics = compute_layer_metrics(activation_array)
    metrics_path = output_dir / "metrics.json"
    metrics_serializable: dict = {
        "config": {
            "model": model_name_or_path,
            "dataset": dataset_name,
            "max_samples": max_samples,
            "max_context_len": max_context_len,
            "num_moe_layers": num_moe_layers,
            "num_experts": num_experts,
        },
        "metrics": {k: v for k, v in metrics.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("=== Per-Layer Metrics Summary ===")
    for k, vals in metrics.items():
        mean_val = np.mean(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        logger.info("  %s: mean=%.4f  min=%.4f  max=%.4f", k, mean_val, min_val, max_val)

    plot_metrics(metrics, layer_indices, output_dir / "metrics_plot.png")
    plot_heatmap(activation_array, layer_indices, output_dir / "expert_heatmap.png")
    logger.info("Visualization complete. All outputs in %s", output_dir)


# ============================================================
# Subset Selection (Part 2)
# ============================================================

def _load_activation_data(activation_file: Path) -> tuple[np.ndarray, list[int]]:
    data = np.load(activation_file)
    activation_array = data["activation_array"]
    layer_indices = data["layer_indices"].tolist()
    return activation_array, layer_indices


def _build_fingerprints(activation_array: np.ndarray) -> np.ndarray:
    num_samples, num_layers, num_experts = activation_array.shape
    fp = np.zeros((num_samples, num_layers * num_experts), dtype=np.float64)
    for s in range(num_samples):
        for l in range(num_layers):
            layer_counts = activation_array[s, l, :].astype(np.float64)
            total = layer_counts.sum()
            if total > 0:
                layer_counts /= total
            fp[s, l * num_experts : (l + 1) * num_experts] = layer_counts
    return fp


def _pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def _kmeans(
    X: np.ndarray,
    n_clusters: int,
    max_iters: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    centers = X[rng.choice(len(X), n_clusters, replace=False)].copy()
    for _ in range(max_iters):
        distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = X[rng.choice(len(X))]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, centers


def _select_per_cluster(
    fingerprints: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    quota: int,
) -> list[int]:
    selected: list[int] = []
    for k in range(len(centers)):
        mask = labels == k
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if quota >= len(indices):
            selected.extend(indices.tolist())
        else:
            center = centers[k]
            cluster_fp = fingerprints[indices]
            dists = np.sum((cluster_fp - center) ** 2, axis=1)
            top = np.argsort(dists)[:quota]
            selected.extend(indices[top].tolist())
    return selected


def select_subset_cluster_stratified(
    activation_array: np.ndarray,
    num_samples: int,
    n_clusters: int = 16,
    random_state: int = 42,
) -> list[int]:
    fingerprints = _build_fingerprints(activation_array)
    logger.info("Fingerprints shape: %s", fingerprints.shape)

    n_pca = min(n_clusters * 4, fingerprints.shape[1], fingerprints.shape[0] - 1)
    logger.info("PCA: %d -> %d dims", fingerprints.shape[1], n_pca)
    fp_reduced = _pca_reduce(fingerprints, n_pca)
    explained_var = np.var(fp_reduced, axis=0).sum() / np.var(fingerprints, axis=0).sum()
    logger.info("PCA explained variance ratio: %.4f", explained_var)

    labels, centers = _kmeans(fp_reduced, n_clusters, random_state=random_state)
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Cluster sizes: %s", dict(zip(unique.tolist(), counts.tolist())))

    cluster_quotas = np.zeros(n_clusters, dtype=int)
    for k in range(n_clusters):
        cluster_quotas[k] = max(1, int(round(num_samples * counts[k] / len(fingerprints))))
    diff = num_samples - cluster_quotas.sum()
    sorted_clusters = np.argsort(counts)[::-1]
    for i in range(abs(diff)):
        idx = sorted_clusters[i % len(sorted_clusters)]
        if diff > 0:
            cluster_quotas[idx] += 1
        else:
            if cluster_quotas[idx] > 1:
                cluster_quotas[idx] -= 1
    logger.info("Cluster quotas: %s (total=%d)", dict(enumerate(cluster_quotas.tolist())), cluster_quotas.sum())

    selected = _select_per_cluster(fingerprints, centers, labels, max(cluster_quotas))
    if len(selected) > num_samples:
        selected = selected[:num_samples]
    logger.info("Selected %d samples (clusters=%d)", len(selected), n_clusters)
    return selected


def select_subset_greedy_entropy(
    activation_array: np.ndarray,
    num_samples: int,
) -> list[int]:
    num_samples_total, num_layers, num_experts = activation_array.shape
    selected: list[int] = []
    available = set(range(num_samples_total))

    cum_counts = np.zeros((num_layers, num_experts), dtype=np.float64)

    pbar = tqdm(total=num_samples, desc="Greedy entropy selection", unit="sample")
    while len(selected) < num_samples and available:
        best_idx = -1
        best_gain = -1.0
        for idx in list(available)[:5000]:
            new_counts = cum_counts + activation_array[idx].astype(np.float64)
            gain = 0.0
            for l in range(num_layers):
                gain += compute_normalized_entropy(new_counts[l])
            gain /= num_layers
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        available.remove(best_idx)
        cum_counts += activation_array[best_idx].astype(np.float64)
        pbar.update(1)
    pbar.close()
    return selected


def select_subset_greedy_coverage(
    activation_array: np.ndarray,
    num_samples: int,
) -> list[int]:
    num_samples_total, num_layers, num_experts = activation_array.shape
    selected: list[int] = []
    available = set(range(num_samples_total))

    covered: np.ndarray = np.zeros((num_layers, num_experts), dtype=bool)

    pbar = tqdm(total=num_samples, desc="Greedy coverage selection", unit="sample")
    while len(selected) < num_samples and available:
        best_idx = -1
        best_gain = -1
        for idx in list(available)[:5000]:
            new_covered = (activation_array[idx] > 0).astype(bool)
            gain = (new_covered & ~covered).sum()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain == 0:
            break
        selected.append(best_idx)
        available.remove(best_idx)
        covered |= (activation_array[best_idx] > 0).astype(bool)
        pbar.update(1)
    pbar.close()
    return selected


def run_selection(
    activation_file: Path,
    output_dir: Path,
    num_samples: int = 256,
    method: str = "cluster_stratified",
    n_clusters: int = 16,
    random_state: int = 42,
) -> None:
    activation_array, layer_indices = _load_activation_data(activation_file)
    num_samples_total, num_layers, num_experts = activation_array.shape
    logger.info(
        "Loaded activation data: samples=%d layers=%d experts=%d",
        num_samples_total, num_layers, num_experts,
    )

    if method == "cluster_stratified":
        selected = select_subset_cluster_stratified(
            activation_array, num_samples, n_clusters, random_state,
        )
    elif method == "greedy_entropy":
        selected = select_subset_greedy_entropy(activation_array, num_samples)
    elif method == "greedy_coverage":
        selected = select_subset_greedy_coverage(activation_array, num_samples)
    else:
        raise ValueError(f"Unknown method: {method}")

    output_dir.mkdir(parents=True, exist_ok=True)

    indices_path = output_dir / "selected_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump({"method": method, "num_samples": len(selected), "indices": selected}, f, ensure_ascii=False, indent=2)
    logger.info("Selected indices saved to %s", indices_path)

    full_metrics = compute_layer_metrics(activation_array)
    subset_array = activation_array[selected]
    subset_metrics = compute_layer_metrics(subset_array)

    comparison_path = output_dir / "subset_comparison.json"
    comparison: dict = {
        "method": method,
        "num_selected": len(selected),
        "full": full_metrics,
        "subset": subset_metrics,
    }
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info("=== Full vs Subset Metrics ===")
    for metric_name in ["normalized_entropy", "coverage_ratio", "gini"]:
        full_mean = np.mean(full_metrics[metric_name])
        subset_mean = np.mean(subset_metrics[metric_name])
        logger.info(
            "  %s: full=%.4f  subset=%.4f  delta=%.4f",
            metric_name, full_mean, subset_mean, subset_mean - full_mean,
        )

    plot_comparison(full_metrics, subset_metrics, layer_indices, output_dir / "subset_comparison.png")
    logger.info("All outputs in %s", output_dir)


# ============================================================
# Visualization
# ============================================================

def plot_metrics(
    metrics: dict[str, list[float]],
    layer_indices: list[int],
    output_path: Path,
) -> None:
    metric_names = list(metrics.keys())
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, name, color in zip(axes, metric_names, colors[:n]):
        vals = metrics[name]
        ax.plot(layer_indices, vals, marker="o", markersize=4, linewidth=1.5, color=color)
        ax.set_xlabel("MoE Layer Index")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(f"Per-Layer {name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Metrics plot saved to %s", output_path)


def plot_heatmap(
    activation_array: np.ndarray,
    layer_indices: list[int],
    output_path: Path,
) -> None:
    num_layers = activation_array.shape[1]
    num_experts = activation_array.shape[2]
    layer_activation = np.zeros((num_layers, num_experts), dtype=np.float64)
    for l in range(num_layers):
        layer_activation[l] = activation_array[:, l, :].sum(axis=0)
    layer_activation = layer_activation / (layer_activation.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(max(12, num_experts * 0.15), max(6, num_layers * 0.2)))
    im = ax.imshow(layer_activation, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("MoE Layer Index")
    ax.set_title("Expert Activation Frequency Heatmap (per-layer normalized)")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(l) for l in layer_indices])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Activation Frequency")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Heatmap saved to %s", output_path)


def plot_comparison(
    full_metrics: dict[str, list[float]],
    subset_metrics: dict[str, list[float]],
    layer_indices: list[int],
    output_path: Path,
) -> None:
    metric_names = list(full_metrics.keys())
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colors = ["#1f77b4", "#d62728"]
    for ax, name in zip(axes, metric_names):
        ax.plot(layer_indices, full_metrics[name], marker="o", markersize=4,
                linewidth=1.5, color=colors[0], label="Full Dataset")
        ax.plot(layer_indices, subset_metrics[name], marker="s", markersize=4,
                linewidth=1.5, color=colors[1], label="Selected Subset")
        ax.set_xlabel("MoE Layer Index")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(f"{name.replace('_', ' ').title()}: Full vs Subset")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison plot saved to %s", output_path)


# ============================================================
# CLI
# ============================================================

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE 校准数据集专家激活分析与子集选择",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="运行推理，统计专家激活频率，计算测度")
    p_analyze.add_argument("--model", required=True, help="模型路径或 HF 名称")
    p_analyze.add_argument("--dataset", required=True, help="HF 数据集名，如 DKYoon/SlimPajama-200k")
    p_analyze.add_argument("--max_samples", type=int, default=20000, help="最大分析样本数")
    p_analyze.add_argument("--max_context_len", type=int, default=2048, help="每个样本最大 token 数")
    p_analyze.add_argument("--device", default="auto", help="推理设备，默认 auto")
    p_analyze.add_argument("--dtype", default="bfloat16", help="模型 dtype")
    p_analyze.add_argument("--output_dir", required=True, help="输出目录")

    p_select = sub.add_parser("select", help="从激活数据中选出最优校准子集")
    p_select.add_argument("--activation_file", required=True, help="analyze 生成的 .npz 文件")
    p_select.add_argument("--num_samples", type=int, default=256, help="子集大小")
    p_select.add_argument("--method", default="cluster_stratified",
                          choices=["cluster_stratified", "greedy_entropy", "greedy_coverage"],
                          help="选择算法")
    p_select.add_argument("--n_clusters", type=int, default=16, help="聚类数（cluster_stratified 时生效）")
    p_select.add_argument("--random_state", type=int, default=42, help="随机种子")
    p_select.add_argument("--output_dir", required=True, help="输出目录")

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
        run_analysis(
            model_name_or_path=args.model,
            dataset_name=args.dataset,
            output_dir=Path(args.output_dir),
            max_samples=args.max_samples,
            max_context_len=args.max_context_len,
            device=args.device,
            dtype=dtype,
        )
    elif args.command == "select":
        run_selection(
            activation_file=Path(args.activation_file),
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples,
            method=args.method,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()