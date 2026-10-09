"""Command-line compensation options shared by inference and evaluation."""
from __future__ import annotations

from prefetch.prerouter.checkpoint import read_metadata


def add_compensation_arguments(parser):
    group = parser.add_argument_group("compensation (used with --execution-mode compensated)")
    group.add_argument("--method", choices=["owa", "exfold"], help="Compensation method")
    group.add_argument("--path", help="Calibrated compensation safetensors file")
    group.add_argument("--alpha1", type=float, help="OWA overlap weight multiplier")
    group.add_argument("--alpha2", type=float, help="OWA total weight multiplier")
    group.add_argument("--hit-min", type=int, help="OWA minimum overlap count")
    group.add_argument("--hit-max", type=int, help="OWA maximum overlap count")


def compensation_from_args(args, prefetch=None, checkpoint=None):
    overrides = {key: getattr(args, key) for key in
                 ("method", "path", "alpha1", "alpha2", "hit_min", "hit_max")
                 if getattr(args, key) is not None}
    if not overrides:
        return None
    defaults = (prefetch or {}).get("compensation")
    if defaults is None and checkpoint:
        defaults = read_metadata(checkpoint)["config"].get("compensation")
    options = dict(defaults or {})
    method = overrides.get("method")
    if method is not None and method != options.get("method", method):
        options = {}  # Switching methods starts with that method's own options.
    options.update(overrides)
    return options
