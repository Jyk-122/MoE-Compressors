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
                         mean_hits=hits / tokens if tokens else None,
                         full_coverage=full / tokens if tokens else None,
                         expert_hits=hits, expert_count=expert_count,
                         full_tokens=full, token_layer_pairs=tokens))
    return rows


def required_k_statistics(max_rank_hist):
    """RequiredK is the largest predicted rank among a call's true experts."""
    count = sum(max_rank_hist)
    cumulative = 0
    distribution = []
    for k, frequency in enumerate(max_rank_hist[1:], start=1):
        cumulative += frequency
        distribution.append(dict(k=k, count=frequency,
                                 probability=frequency / count if count else None,
                                 cdf=cumulative / count if count else None))
    summary = dict(token_layer_pairs=count, mean=None, median=None, min=None, max=None,
                   p90=None, p95=None, p99=None)
    if count:
        occupied = [row for row in distribution if row["count"]]
        cumulative_counts = []
        cumulative = 0
        for row in occupied:
            cumulative += row["count"]
            cumulative_counts.append((row["k"], cumulative))

        def value_at(position):
            return next(k for k, total in cumulative_counts if total >= position)

        summary.update(mean=sum(row["k"] * row["count"] for row in occupied) / count,
                       median=(value_at((count + 1) // 2) + value_at((count + 2) // 2)) / 2,
                       min=occupied[0]["k"], max=occupied[-1]["k"])
        # Coverage-budget quantiles use the first integer k whose CDF reaches p.
        for percentile in (90, 95, 99):
            summary[f"p{percentile}"] = value_at((count * percentile + 99) // 100)
    return dict(**summary, distribution=distribution)


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
            "required_k": {
                "global": required_k_statistics(total[1]),
                "per_layer": {str(m["target_moe"]): required_k_statistics(h[1])
                              for m, h in zip(metadata["layers"], layers)},
            },
            "histograms": layers,
        }
    return result


def rebuild_report(report, layers=None):
    """Recompute all statistics from saved histograms, optionally selecting targets."""
    metadata = dict(report["metadata"])
    selected = [i for i, layer in enumerate(metadata["layers"])
                if layers is None or layer["target_moe"] in layers]
    if layers is not None and (not selected or len(selected) != len(layers)):
        raise ValueError("Requested common target layers are absent or repeated")
    metadata["layers"] = [metadata["layers"][i] for i in selected]
    histograms = {phase: [data["histograms"][i] for i in selected]
                  for phase, data in report["phases"].items()}
    ks = [row["k"] for row in next(iter(report["phases"].values()))["global"]]
    result = {**report, **report_histograms(histograms, metadata, ks)}
    if layers is not None and "trace" in result:
        result["trace"] = [row for row in result["trace"] if row["target_moe"] in layers]
    return result


def _write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows, summaries, distributions = [], [], []
    for phase, stats in report["phases"].items():
        for layer, curve in {"global": stats["global"], **stats["per_layer"]}.items():
            rows.extend(dict(phase=phase, layer=layer, **row) for row in curve)
        required = stats["required_k"]
        for layer, values in {"global": required["global"], **required["per_layer"]}.items():
            summaries.append(dict(phase=phase, layer=layer,
                                  **{k: v for k, v in values.items() if k != "distribution"}))
            distributions.extend(dict(phase=phase, layer=layer, **row) for row in values["distribution"])
    _write_csv(path.with_suffix(".csv"), rows)
    _write_csv(path.with_name(path.stem + "-required_k.csv"), summaries)
    _write_csv(path.with_name(path.stem + "-required_k_distribution.csv"), distributions)
