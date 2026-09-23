"""Cache I/O summaries and capacity curves, globally and per MoE layer."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def plot_curve(curve, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    points = [row for row in curve if row["decode_tokens"]]
    if points:
        xs = [row["capacity"] for row in points]
        for key, label in (("baseline_loads", "Demand-only oracle"), ("total_loads", "Prefetch + demand")):
            axes[0].plot(xs, [row[key + "_per_token"] for row in points], marker="o", label=label)
        axes[1].plot(xs, [row["prefetch_loads_per_token"] for row in points], marker="o", label="Prefetch loads")
        axes[1].plot(xs, [row["demand_loads_per_token"] for row in points], marker="o", label="Demand reloads")
        for axis in axes:
            axis.legend()
    for axis, subtitle in zip(axes, ("Total I/O", "Prefetch strategy breakdown")):
        axis.set(title=subtitle, xlabel="Cache capacity N (experts per layer)", ylim=(0, None))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Expert loads per decode token")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(report, output, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for source, data in report["sources"].items():
        for layer, curve in {"global": data["global_curve"], **data["per_layer"]}.items():
            rows.extend(dict(source=source, layer=layer, **row) for row in curve)
            if plots:
                scope = "all selected layers (sum)" if layer == "global" else f"MoE {layer}"
                plot_curve(curve, f"{source}: {scope}", output / f"{source}-{layer}.png")
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
