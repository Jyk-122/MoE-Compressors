"""CPU-only cache simulation from complete decode-trace JSONL files."""
from __future__ import annotations

import argparse
from collections import defaultdict
from glob import glob
import json
from pathlib import Path

from prefetch.evaluation.cache.oracle import simulate_layer
from prefetch.utils.logging import configure_logging


COSTS = ("baseline_loads", "prefetch_loads", "demand_loads",
         "baseline_evictions", "prefetch_evictions", "demand_evictions")


def summarize(counts, capacity, tokens, requests, active_requests):
    result = dict(capacity=capacity, requests=requests, requests_with_decode=active_requests, decode_tokens=tokens)
    for prefix in ("", "first_token_"):
        for key in COSTS:
            result[prefix + key] = counts[prefix + key]
        result[prefix + "total_loads"] = counts[prefix + "prefetch_loads"] + counts[prefix + "demand_loads"]
        result[prefix + "total_evictions"] = counts[prefix + "prefetch_evictions"] + counts[prefix + "demand_evictions"]
    for key in (*COSTS, "total_loads", "total_evictions"):
        result[key + "_per_token"] = result[key] / tokens if tokens else None
        result["first_token_" + key + "_per_request"] = result["first_token_" + key] / active_requests if active_requests else None
        result["after_first_" + key + "_per_token"] = ((result[key] - result["first_token_" + key]) /
                                                     (tokens - active_requests) if tokens > active_requests else None)
    result["extra_loads"] = result["total_loads"] - result["baseline_loads"]
    result["extra_loads_per_token"] = result["extra_loads"] / tokens if tokens else None
    base = result["baseline_loads"]
    result["io_amplification"] = result["total_loads"] / base if base else None
    result["demand_load_reduction"] = 1 - result["demand_loads"] / base if base else None
    return result


def validate_request(record, metadata):
    tokens, layers = record["decode_tokens"], metadata["layers"]
    if tokens < 0 or [row["moe"] for row in record["layers"]] != [layer["moe"] for layer in layers]:
        raise ValueError("Trace must contain every MoE layer in metadata order")
    for layer, row in zip(layers, record["layers"]):
        if len(row["truth"]) != tokens or any(len(r) != layer["top_k"] for r in row["truth"]):
            raise ValueError("Incomplete truth trace or wrong native top-k")
        predicted = row["prediction"]
        if layer["source_moe"] is None:
            if predicted is not None:
                raise ValueError("A layer without a prerouter must have prediction=null")
        elif (predicted is None or len(predicted) != tokens or
              any(len(p) != metadata["prediction_k"] for p in predicted)):
            raise ValueError("Incomplete prediction trace or wrong prediction top-k")


def simulate_files(paths, capacities, layers=None):
    capacities = sorted(set(capacities))
    if not capacities or capacities[0] < 1:
        raise ValueError("Specify positive cache capacities")
    shared = None
    provenance, sources = [], {}
    counters = defaultdict(lambda: defaultdict(int))
    seen = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as file:
            metadata = json.loads(next(file))
            if (metadata.get("format") != "moe-cache-trace-v1" or
                    metadata.get("mode") != "same_token" or metadata.get("phase") != "decode"):
                raise ValueError("Use same-token decode JSONL produced by cache.collect")
            identity = {key: metadata.get(key) for key in
                        ("format", "mode", "phase", "prediction_k", "layers", "checkpoint", "model", "lora", "prefetch", "generation")}
            if shared is None:
                shared = identity
                selected = [layer for layer in metadata["layers"] if layers is None or layer["moe"] in layers]
                if not selected or (layers is not None and len(selected) != len(layers)):
                    raise ValueError("Selected layers must be present and unique")
                minimum = max(max(layer["top_k"], metadata["prediction_k"] if layer["source_moe"] is not None else 0)
                              for layer in selected)
                if capacities[0] < minimum:
                    raise ValueError(f"Each cache must hold a whole request set; capacity must be >= {minimum}")
            elif shared != identity:
                raise ValueError("Trace files must use the same model, checkpoint, layers and generation settings")
            provenance.append(dict(path=str(Path(path).resolve()), metadata=metadata))
            source = metadata["source"]
            summary = sources.setdefault(source, dict(requests=0, decode_tokens=0, requests_with_decode=0))
            received, complete = 0, False
            for line in file:
                record = json.loads(line)
                if record["type"] == "complete":
                    if record["requests"] != received or received != metadata["expected_requests"]:
                        raise ValueError(f"Incomplete request count in {path}")
                    if file.read().strip():
                        raise ValueError("Completion marker must be the last trace record")
                    complete = True
                    break
                if record["type"] != "request":
                    raise ValueError("Expected a complete request record")
                key = (source, record["sample_index"])
                if key in seen:
                    raise ValueError(f"Duplicate request across trace files: {key}")
                seen.add(key)
                validate_request(record, metadata)
                received += 1
                summary["requests"] += 1
                summary["decode_tokens"] += record["decode_tokens"]
                summary["requests_with_decode"] += int(record["decode_tokens"] > 0)
                by_layer = {row["moe"]: row for row in record["layers"]}
                for layer in selected:
                    row = by_layer[layer["moe"]]
                    for capacity in capacities:
                        counts = counters[source, layer["moe"], capacity]
                        for event in simulate_layer(row["truth"], row["prediction"], capacity, layer["experts"]):
                            for cost in COSTS:
                                counts[cost] += event[cost]
                                if event["token"] == 0:
                                    counts["first_token_" + cost] += event[cost]
            if not complete:
                raise ValueError(f"Trace collection has not completed: {path}")
    if shared is None:
        raise ValueError("No trace files supplied")

    report = dict(metadata=dict(trace_format=shared["format"], capacities=capacities, layers=selected,
                                phase="decode", initial_cache="empty_at_first_decode_per_request",
                                eviction="farthest_next_true_use; highest_expert_id_breaks_ties",
                                stage_order=["prefetch", "true_demand"], current_stage_set_pinned=True,
                                unit="expert_loads; equal-sized experts; read-only weights",
                                prefetch_completion="assumed_before_true_router", traces=provenance), sources={})
    for source, summary in sources.items():
        def row_for(counts, capacity):
            return summarize(counts, capacity, summary["decode_tokens"], summary["requests"], summary["requests_with_decode"])
        per_layer = {str(layer["moe"]): [row_for(counters[source, layer["moe"], capacity], capacity) for capacity in capacities]
                     for layer in selected}
        global_rows = []
        for capacity in capacities:
            combined = defaultdict(int)
            for layer in selected:
                for key, value in counters[source, layer["moe"], capacity].items():
                    combined[key] += value
            global_rows.append(row_for(combined, capacity))
        report["sources"][source] = dict(**summary, global_curve=global_rows, per_layer=per_layer)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="JSONL paths or glob patterns, including all desired ranks")
    parser.add_argument("--capacities", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--layers", nargs="+", type=int, help="Select MoE ordinals; default: every layer")
    parser.add_argument("--output", required=True, help="Directory for report.json, summary.csv and per-layer figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    configure_logging()
    paths = []
    for pattern in args.traces:
        matched = sorted(glob(pattern))
        if not matched:
            parser.error(f"No trace matches {pattern}")
        paths.extend(matched)
    report = simulate_files(paths, args.capacities, args.layers)
    from prefetch.evaluation.cache.report import save_report
    save_report(report, args.output, plots=not args.no_plots)
    keys = ("capacity", "decode_tokens", "baseline_loads_per_token", "prefetch_loads_per_token",
            "demand_loads_per_token", "total_loads_per_token", "io_amplification", "demand_load_reduction")
    summary = {name: [{key: row[key] for key in keys} for row in data["global_curve"]]
               for name, data in report["sources"].items()}
    print(json.dumps({"output": args.output, "global": summary}), flush=True)


if __name__ == "__main__":
    main()
