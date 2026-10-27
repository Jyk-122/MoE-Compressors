"""python -m skipping_qwen3_5_moe.calibrate --config skipping_qwen3_5_moe/configs/train.yaml"""
from __future__ import annotations

import argparse
import math

import torch

from .artifacts import save_artifact
from .config import read_config, load_model_and_processor
from .data import make_loader, to_device
from .experts import expert_output
from .model import find_moe_layers


@torch.no_grad()
def calibrate_outputs(outputs, epsilon=1e-5):
    """Compute channel means and population standard deviations from [samples, N, H]."""
    if outputs.ndim != 3 or outputs.shape[0] < 2:
        raise ValueError("Calibration needs at least two [sample, expert, channel] observations")
    samples, experts, hidden = outputs.shape
    if experts < 2 or hidden < 1:
        raise ValueError("Expected at least two experts and one hidden channel")
    if not math.isfinite(epsilon) or epsilon <= 0 or not torch.isfinite(outputs).all():
        raise ValueError("Calibration outputs must be finite and epsilon positive")
    variance, mean = torch.var_mean(outputs.cpu().double(), dim=0, unbiased=False)
    return {"mean": mean.float(), "std": variance.sqrt().clamp_min(epsilon).float(),
            "samples": samples, "epsilon": epsilon}


@torch.no_grad()
def calibrate_model(model, loader, *, epsilon=1e-5):
    """Accumulate each expert's output sums and squared sums on all valid tokens."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    layers = find_moe_layers(model)
    if hasattr(model, "skipping_controller"):
        raise ValueError("Calibrate the original model before installing compensation")
    statistics = {}
    for index, layer in layers:
        count, _, hidden = layer.mlp.experts.gate_up_proj.shape
        statistics[index] = {"sum": torch.zeros(count, hidden, dtype=torch.float64, device="cpu"),
                             "square_sum": torch.zeros(count, hidden, dtype=torch.float64, device="cpu"),
                             "samples": 0}
    current_mask = None
    handles = []
    previous_training = model.training
    model.eval()
    try:
        for index, layer in layers:
            def capture(module, args, index=index):
                hidden = args[0].reshape(-1, args[0].shape[-1])
                hidden = hidden[current_mask.reshape(-1).to(hidden.device)]
                if hidden.shape[0] == 0:
                    return
                weight = module.experts.gate_up_proj
                hidden = hidden.to(device=weight.device, dtype=weight.dtype)
                state = statistics[index]
                for expert_id in range(weight.shape[0]):
                    outputs = expert_output(module.experts, hidden, expert_id).cpu().double()
                    state["sum"][expert_id] += outputs.sum(0)
                    state["square_sum"][expert_id] += outputs.square().sum(0)
                state["samples"] += hidden.shape[0]
            handles.append(layer.mlp.register_forward_pre_hook(capture))
        device = model.get_input_embeddings().weight.device
        for batch_index, batch in enumerate(loader, 1):
            inputs = to_device({k: v for k, v in batch.items() if k not in {"labels", "router_mask"}}, device)
            current_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
            model(**inputs, use_cache=False, logits_to_keep=1, output_router_logits=False)
            print(f"Calibrated batch {batch_index}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_training)

    result = {}
    for index, layer in layers:
        state = statistics[index]
        if state["samples"] < 2:
            raise ValueError(f"Layer {index} has fewer than two valid calibration inputs")
        mean = state["sum"] / state["samples"]
        variance = (state["square_sum"] / state["samples"] - mean.square()).clamp_min(0)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError(f"Layer {index} has non-finite calibration moments")
        result[index] = {"mean": mean.float(), "std": variance.sqrt().clamp_min(epsilon).float(),
                         "samples": state["samples"], "epsilon": epsilon, "top_k": layer.mlp.gate.top_k}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    torch.manual_seed(config["seed"])
    model, processor = load_model_and_processor(config, args.device)
    loader = make_loader(processor, config["data"], split="calibration")
    options = {k: v for k, v in config["calibration"].items() if k != "artifact"}
    state = calibrate_model(model, loader, **options)
    path = args.output or config["calibration"]["artifact"]
    save_artifact(state, path, threshold=config["skipping"]["threshold"], base_model=config["model"]["path"])
    print(f"Saved calibration: {path}", flush=True)


if __name__ == "__main__":
    main()
