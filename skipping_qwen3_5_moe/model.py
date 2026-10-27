"""Qwen MoE integration, local reconstruction, and threshold-aware inference."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn

from .artifacts import read_artifact, save_artifact
from .experts import selected_outputs
from .modules import MomentCompensation, CompensateCoefficients
from .routing import top_p_mask, validate_threshold


def find_moe_layers(model):
    backbone = getattr(model, "model", model)
    backbone = getattr(backbone, "language_model", backbone)
    if not hasattr(backbone, "layers"):
        raise ValueError("Expected Qwen text decoder layers")
    layers = []
    for index, layer in enumerate(backbone.layers):
        block = getattr(layer, "mlp", None)
        if isinstance(block, CompensatedMoE):
            layers.append((index, layer))
        elif (hasattr(block, "experts") and hasattr(block.experts, "gate_up_proj")
              and hasattr(block.experts, "down_proj") and hasattr(block, "shared_expert_gate")):
            layers.append((index, layer))
    if not layers:
        raise ValueError("No packed Qwen MoE blocks found")
    return layers


class SkippingController:
    """Request-local execution mode, masks, reconstruction terms, and counts."""

    def __init__(self, threshold):
        self.threshold = validate_threshold(threshold)
        self.mode = "student"
        self.attention_mask = None
        self.loss_mask = None
        self.feature_terms = {}
        self.statistics = {}
        self.generation_depth = 0
        self.blocks = []

    @contextmanager
    def context(self, *, mode=None, threshold=None, attention_mask=None, loss_mask=None, reset=False):
        next_mode = self.mode if mode is None else mode
        if next_mode not in {"teacher", "student", "reconstruction"}:
            raise ValueError("Unknown compensation execution mode")
        next_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        previous = (self.mode, self.threshold, self.attention_mask, self.loss_mask)
        self.mode, self.threshold = next_mode, next_threshold
        self.attention_mask, self.loss_mask = attention_mask, loss_mask
        if reset:
            self.feature_terms, self.statistics = {}, {}
        try:
            yield self
        finally:
            self.mode, self.threshold, self.attention_mask, self.loss_mask = previous

    def summary(self):
        report = {}
        for index, counts in self.statistics.items():
            kept, original, positions, mixed = counts.tolist()
            report[str(index)] = {"positions": int(positions), "kept": int(kept),
                                  "original": int(original), "coefficient_positions": int(mixed),
                                  "pruning_ratio": 1 - kept / original if original else 0.0}
        kept = sum(x["kept"] for x in report.values())
        original = sum(x["original"] for x in report.values())
        return {"layers": report, "kept": kept, "original": original,
                "pruning_ratio": 1 - kept / original if original else 0.0}


class CompensatedMoE(nn.Module):
    """Execute retained experts and reconstruct missing contributions in parallel."""

    def __init__(self, original, state, controller, layer_index):
        super().__init__()
        self.original = original
        std = state["std"] if "std" in state else state["log_std"].exp()
        self.operator = MomentCompensation(state["mean"], std, state["epsilon"])
        self.coefficients = CompensateCoefficients(state["mean"].shape[0])
        if "logits" in state:
            with torch.no_grad():
                self.operator.log_std.copy_(state["log_std"])
                self.coefficients.logits.copy_(state["logits"])
        self.controller = controller
        self.layer_index = layer_index
        self.calibration_samples = int(state["samples"])
        device = original.gate.weight.device
        for module in (self.operator, self.coefficients):
            module.to(device=device)

    def forward(self, hidden_states):
        runtime = self.controller
        if runtime.mode == "teacher":
            return self.original(hidden_states)
        batch, length, hidden_dim = hidden_states.shape
        hidden = hidden_states.reshape(-1, hidden_dim)
        valid = self._mask(runtime.attention_mask, batch, length, hidden.device)
        supervised = valid & self._mask(runtime.loss_mask, batch, length, hidden.device)
        _, weights, ids = self.original.gate(hidden)
        keep = top_p_mask(weights, runtime.threshold)
        source_outputs = selected_outputs(self.original.experts, hidden, ids, keep)
        coefficients = self.coefficients(ids, keep)
        routed = self.operator.aggregate(source_outputs, ids, keep, weights, coefficients)

        if runtime.mode == "reconstruction":
            with torch.no_grad():
                missing = selected_outputs(self.original.experts, hidden.detach(), ids, ~keep)
                target = ((source_outputs.detach().float() + missing.float())
                          * weights.detach().float().unsqueeze(-1)).sum(1)
            error = (routed - target).square().mean(-1)
            runtime.feature_terms[self.layer_index] = ((error * supervised).sum(), supervised.sum())

        shared = self.original.shared_expert(hidden)
        shared = self.original.shared_expert_gate(hidden).sigmoid() * shared
        number = keep.sum(-1)
        counts = torch.stack(((number * valid).sum(), valid.sum() * ids.shape[1],
                              valid.sum(), ((number > 1) & (number < ids.shape[1]) & valid).sum()))

        previous = runtime.statistics.get(self.layer_index)
        runtime.statistics[self.layer_index] = counts.detach() if previous is None else previous + counts.detach()
        return (routed.to(hidden.dtype) + shared).reshape(batch, length, hidden_dim)

    @staticmethod
    def _mask(mask, batch, length, device):
        if mask is None:
            return torch.ones(batch * length, device=device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] < length:
            raise ValueError("Expected a 2D token mask covering the current input")
        return mask[:, -length:].to(device=device, dtype=torch.bool).reshape(-1)

    def compensation_state(self):
        return {"mean": self.operator.mean, "log_std": self.operator.log_std,
                "logits": self.coefficients.logits,
                "samples": self.calibration_samples,
                "epsilon": self.operator.epsilon, "top_k": self.original.gate.top_k}


def patch(model, calibration, threshold=0.5):
    """Install independently replaceable modules and expose threshold on model/generate."""
    if hasattr(model, "skipping_controller"):
        raise ValueError("Compensation is already installed")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Local reconstruction requires backbone gradient checkpointing to be disabled")
    layers = find_moe_layers(model)
    if {index for index, _ in layers} != set(calibration):
        raise ValueError("Calibration layers differ from the model's MoE layers")
    # Validate every layer before changing the model.
    for index, layer in layers:
        state, original = calibration[index], layer.mlp
        count, _, hidden = original.experts.gate_up_proj.shape
        std = state["std"] if "std" in state else state["log_std"].exp()
        if (state["mean"].shape != (count, hidden) or std.shape != (count, hidden)
                or state["top_k"] != original.gate.top_k):
            raise ValueError(f"Layer {index}: calibration geometry differs from the model")
        if not torch.isfinite(state["mean"]).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError(f"Layer {index}: invalid compensation moments")
        if "logits" in state:
            if state["logits"].shape != (count, count) or not torch.isfinite(state["logits"]).all():
                raise ValueError(f"Layer {index}: invalid coefficient matrix")
            if not torch.isfinite(state["log_std"]).all():
                raise ValueError(f"Layer {index}: invalid log standard deviations")
    controller = SkippingController(threshold)
    model.requires_grad_(False)
    for index, layer in layers:
        block = CompensatedMoE(layer.mlp, calibration[index], controller, index)
        layer.mlp = block
        controller.blocks.append(block)
    model.skipping_controller = controller
    model.eval()

    original_forward = model.forward

    @wraps(original_forward)
    def forward(*args, threshold=None, **kwargs):
        with controller.context(threshold=threshold,
                                attention_mask=kwargs.get("attention_mask", controller.attention_mask),
                                loss_mask=controller.loss_mask,
                                reset=controller.generation_depth == 0 and controller.mode == "student"):
            return original_forward(*args, **kwargs)
    model.forward = forward

    if hasattr(model, "generate"):
        original_generate = model.generate

        @wraps(original_generate)
        def generate(*args, threshold=None, **kwargs):
            with controller.context(threshold=threshold, reset=True):
                controller.generation_depth += 1
                try:
                    return original_generate(*args, **kwargs)
                finally:
                    controller.generation_depth -= 1
        model.generate = generate
    return controller


def save_compensation(model, path):
    controller = model.skipping_controller
    state = {block.layer_index: block.compensation_state() for block in controller.blocks}
    save_artifact(state, path, threshold=controller.threshold,
                  base_model=getattr(getattr(model, "config", None), "_name_or_path", ""))


def load_compensation(model, path, threshold=None):
    state, metadata = read_artifact(path)
    return patch(model, state, threshold=metadata["threshold"] if threshold is None else threshold)
