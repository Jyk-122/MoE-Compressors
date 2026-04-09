#!/usr/bin/env python3
"""
Plot memory (expert cache capacity) vs average expert loads per token for one or more trace files.

Requires: matplotlib (`pip install matplotlib` or see requirements-expert-io.txt).

I/O metric: each time an expert is brought into the cache counts as one load (swap-in).
Per-layer independent caches; y-axis = sum of loads over layers / total token positions.

Example:
  python utils/plot_expert_io_curves.py --trace a.bin --trace b.bin --label reap --label topp --out out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.expert_cache_sim import capacity_curve, default_capacity_range


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot expert cache capacity vs I/O from one or more binary traces",
    )
    p.add_argument(
        "--trace",
        dest="traces",
        action="append",
        required=True,
        metavar="PATH",
        help="Binary trace path (repeat for multiple curves)",
    )
    p.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        metavar="TEXT",
        help="Legend label for the preceding order of --trace (repeat; default: file stem)",
    )
    p.add_argument("--out", type=str, required=True, help="Output image path (.png or .pdf)")
    p.add_argument("--policy", type=str, default="belady", choices=["belady", "lookahead_lru"])
    p.add_argument("--lookahead", type=int, default=64, help="Token horizon for lookahead_lru")
    p.add_argument("--cap_step", type=int, default=1)
    p.add_argument("--max_cap", type=int, default=None)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (parse trace + capacity sweep)",
    )
    args = p.parse_args()

    traces: list[str] = args.traces
    if args.labels is None:
        labels = [Path(t).stem for t in traces]
    else:
        labels = args.labels
        if len(labels) != len(traces):
            raise SystemExit(
                f"--label count ({len(labels)}) must match --trace count ({len(traces)})",
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    caps = default_capacity_range(traces[0], step=args.cap_step, max_cap=args.max_cap)
    if args.max_cap is not None:
        caps = [c for c in caps if c <= int(args.max_cap)]
    if not caps:
        raise SystemExit("Empty capacity list")

    show_p = not args.no_progress
    markers = ("o", "s", "^", "v", "D", "P", "X", "*")
    plt.figure(figsize=(7, 4.5))
    trace_iter = zip(traces, labels, strict=True)
    if show_p:
        try:
            from tqdm import tqdm

            trace_iter = tqdm(list(trace_iter), desc="Traces", unit="curve")
        except ImportError:
            pass
    for i, (trace_path, label) in enumerate(trace_iter):
        curve = capacity_curve(
            trace_path,
            caps,
            policy=args.policy,
            lookahead=int(args.lookahead),
            show_progress=show_p,
        )
        xs = [c for c, _ in curve]
        ys = [y for _, y in curve]
        m = markers[i % len(markers)]
        plt.plot(xs, ys, marker=m, markersize=2, label=label)

    plt.xlabel("Cache capacity (experts per layer)")
    plt.ylabel("Avg expert loads per token\n(sum over layers / #token steps)")
    plt.title(f"Expert I/O vs capacity ({args.policy})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
