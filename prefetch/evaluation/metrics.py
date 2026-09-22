"""Pure-Python reporting for rank histograms (also usable without PyTorch)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def histogram_curve(rank_hist, max_rank_hist, ks):
    """Rank bins are 1-based; bin zero is unused."""
    experts = len(rank_hist) - 1
    expert_count, tokens = sum(rank_hist), sum(max_rank_hist)
    rows = []
    for k in sorted(set(int(k) for k in ks if 1 <= int(k) <= experts)):
        hits, full = sum(rank_hist[:k + 1]), sum(max_rank_hist[:k + 1])
        rows.append(dict(k=k, recall=hits / expert_count if expert_count else None,
                         full_coverage=full / tokens if tokens else None,
                         expert_hits=hits, expert_count=expert_count,
                         full_tokens=full, token_layer_pairs=tokens))
    return rows


def report_histograms(histograms, metadata, ks):
    """histograms: phase -> [layer][rank/max-rank][bin]."""
    result = {"metadata": metadata, "phases": {}}
    for phase, layers in histograms.items():
        n_bins = len(layers[0][0])
        total = [[sum(layer[j][i] for layer in layers) for i in range(n_bins)]
                 for j in range(2)]
        result["phases"][phase] = {
            "global": histogram_curve(*total, ks),
            "per_layer": {str(m["target_moe"]): histogram_curve(*h, ks)
                          for m, h in zip(metadata["layers"], layers)},
            "histograms": layers,
        }
    return result


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for phase, stats in report["phases"].items():
        for layer, curve in {"global": stats["global"], **stats["per_layer"]}.items():
            rows.extend(dict(phase=phase, layer=layer, **row) for row in curve)
    if rows:
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
