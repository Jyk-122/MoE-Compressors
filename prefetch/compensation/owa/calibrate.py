"""Grid-search OWA coefficients against native routed-expert outputs."""
from __future__ import annotations

import argparse
from itertools import product
import json

import torch

from ..artifacts import save_artifact
from ..calibration import add_arguments, collect, load_calibration
from .compensate import OWA


class OWACalibration:
    sampling = "prefetch"

    def __init__(self, alpha1=(0.5, 1.0, 2.0), alpha2=(0.8, 1.0, 1.2), hit_min=1, hit_max=None):
        self.methods = [OWA(a, b, hit_min, hit_max) for a, b in product(alpha1, alpha2)]
        if not self.methods:
            raise ValueError("OWA search requires at least one coefficient pair")
        self.errors = torch.zeros(len(self.methods), dtype=torch.float64)
        self.baseline_error = self.energy = 0.0
        self.tokens = self.partial_hits = 0

    @torch.no_grad()
    def update(self, target, native, predicted, native_outputs, predicted_outputs):
        ids, weights = native
        selected, baseline = predicted
        reference = (native_outputs * weights.float().unsqueeze(-1)).sum(1)
        baseline_output = (predicted_outputs * baseline.float().unsqueeze(-1)).sum(1)
        self.energy += float(reference.double().square().sum())
        self.baseline_error += float((baseline_output - reference).double().square().sum())
        count = (selected.unsqueeze(2) == ids.unsqueeze(1)).any(-1).sum(-1)
        self.tokens += ids.shape[0]
        self.partial_hits += int(((count > 0) & (count < ids.shape[1])).sum())
        for index, method in enumerate(self.methods):
            adjusted = method(ids, weights, selected, baseline)
            output = (predicted_outputs * adjusted.float().unsqueeze(-1)).sum(1)
            self.errors[index] += (output - reference).double().square().sum().cpu()

    def result(self):
        if self.energy <= 0:
            raise ValueError("OWA calibration requires nonzero native output energy")
        index = int(self.errors.argmin())
        method = self.methods[index]
        return dict(parameters=dict(alpha1=method.alpha1, alpha2=method.alpha2,
                                    hit_min=method.hit_min, hit_max=method.hit_max),
                    normalized_mse=float(self.errors[index]) / self.energy,
                    baseline_normalized_mse=self.baseline_error / self.energy,
                    tokens=self.tokens, partial_hit_tokens=self.partial_hits,
                    candidates=[dict(alpha1=m.alpha1, alpha2=m.alpha2,
                                     normalized_mse=float(error) / self.energy)
                                for m, error in zip(self.methods, self.errors)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--alpha1", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--alpha2", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--hit-min", type=int, default=1)
    parser.add_argument("--hit-max", type=int)
    args = parser.parse_args()
    collector = OWACalibration(args.alpha1, args.alpha2, args.hit_min, args.hit_max)
    model, processor, state, config, device, metadata = load_calibration(args)
    counts = collect(model, processor, state, config, device, args, collector)
    report = collector.result()
    metadata.update(method="owa", layer_tokens=counts, **report)
    if args.output:
        p = report["parameters"]
        save_artifact(args.output, {"alpha": torch.tensor([p["alpha1"], p["alpha2"]])}, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
