from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_report(report, output_prefix, layers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prefetch.evaluation.metrics import histogram_curve
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for metric, label in (("recall", "Recall"), ("full_coverage", "Full coverage")):
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        drawn = False
        for phase, data in report["phases"].items():
            curve = data["global"]
            if layers:
                mapping = report["metadata"]["layers"]
                selected = [hist for meta, hist in zip(mapping, data["histograms"]) if meta["target_moe"] in layers]
                if len(selected) != len(layers):
                    raise ValueError("Requested common target layers are absent in this report")
                combined = [[sum(h[j][i] for h in selected) for i in range(len(selected[0][0]))] for j in range(2)]
                curve = histogram_curve(*combined, [row["k"] for row in curve])
            points = [row for row in curve if row[metric] is not None]
            if points:
                axis.plot([p["k"] for p in points], [p[metric] for p in points], marker="o", label=phase)
                drawn = True
        axis.set(xlabel="Candidate experts k'", ylabel=label, ylim=(0, 1.01))
        axis.grid(alpha=0.25)
        if drawn:
            axis.legend()
        fig.tight_layout()
        fig.savefig(str(prefix) + f"-{metric}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--output-prefix")
    parser.add_argument("--layers", type=int, nargs="+", help="Common target MoE ordinals for distance comparisons")
    args = parser.parse_args()
    path = Path(args.metrics)
    plot_report(json.loads(path.read_text(encoding="utf-8")), args.output_prefix or path.with_suffix(""), args.layers)


if __name__ == "__main__":
    main()
