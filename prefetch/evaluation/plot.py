from __future__ import annotations

import argparse
import json
from pathlib import Path

from prefetch.evaluation.metrics import rebuild_report, save_report


def plot_report(report, output_prefix, layers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    report = rebuild_report(report, layers)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plots = (("recall", "recall", "Recall"),
             ("mean_hits", "mean_hits", "Mean true experts covered"),
             ("required_k_distribution", "probability", "Fraction of token-layer pairs"),
             ("required_k_cdf", "cdf", "Fraction with RequiredK <= k'"))
    for metric, field, label in plots:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        drawn = False
        is_distribution = field in {"probability", "cdf"}
        for phase, data in report["phases"].items():
            curve = data["required_k"]["global"]["distribution"] if is_distribution else data["global"]
            points = [row for row in curve if row[field] is not None]
            if points:
                xs, ys = [p["k"] for p in points], [p[field] for p in points]
                if is_distribution:
                    axis.step(xs, ys, where="post" if field == "cdf" else "mid", label=phase)
                else:
                    axis.plot(xs, ys, marker="o", label=phase)
                drawn = True
        axis.set(xlabel="RequiredK" if field == "probability" else "Candidate experts k'", ylabel=label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylim(bottom=0)
        if metric in {"recall", "required_k_cdf"}:
            axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.25)
        if drawn:
            axis.legend()
        fig.tight_layout()
        fig.savefig(str(prefix) + f"-{metric}.png", dpi=160)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description="Build statistics and plots from saved routing histograms; CPU only.")
    parser.add_argument("metrics", help="Full evaluation JSON containing histograms")
    parser.add_argument("--output-prefix", help="Output file prefix; default: <input stem>-report")
    parser.add_argument("--layers", type=int, nargs="+", help="Common target MoE ordinals for distance comparisons")
    args = parser.parse_args()
    path = Path(args.metrics)
    prefix = args.output_prefix or path.with_name(path.stem + "-report")
    report = plot_report(json.loads(path.read_text(encoding="utf-8")), prefix, args.layers)
    save_report(report, Path(str(prefix) + ".json"))
    summary = {phase: {k: v for k, v in data["required_k"]["global"].items() if k != "distribution"}
               for phase, data in report["phases"].items()}
    print(json.dumps({"output_prefix": str(prefix), "required_k": summary}), flush=True)


if __name__ == "__main__":
    main()
