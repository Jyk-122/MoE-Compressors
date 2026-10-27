"""Same-input layer reconstruction and teacher-to-student next-token KL."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def prediction_mask(batch):
    """A logit at position t supervises the label at position t+1."""
    valid = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).bool()
    mask = valid[:, :-1] & valid[:, 1:]
    if "labels" in batch:
        mask = mask & (batch["labels"][:, 1:] != -100)
    return mask


def hidden_kl(student, teacher, lm_head, mask, temperature=1.0):
    """Compute teacher-to-student vocabulary KL over all valid predictions at once."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("KL temperature must be finite and positive")
    source = student[:, :-1][mask.to(student.device)]
    target = teacher[:, :-1][mask.to(teacher.device)]
    if source.shape[0] == 0:
        raise ValueError("No valid next-token predictions for KL")
    head_device = next(lm_head.parameters()).device

    with torch.no_grad():
        log_teacher = (lm_head(target.to(head_device)).float() / temperature).log_softmax(-1)
    log_student = (lm_head(source.to(head_device)).float() / temperature).log_softmax(-1)
    return F.kl_div(log_student, log_teacher, log_target=True, reduction="batchmean") * temperature ** 2


class CompensationTrainingModel(nn.Module):
    """Train coefficient matrices and moment transport through a frozen backbone."""

    def __init__(self, model, threshold=0.5, feature_weight=1.0, kl_weight=1.0,
                 temperature=1.0):
        super().__init__()
        from .routing import validate_threshold
        self.model = model
        self.threshold = validate_threshold(threshold)
        self.feature_weight, self.kl_weight = feature_weight, kl_weight
        self.temperature = temperature
        if (not all(math.isfinite(x) and x >= 0 for x in (feature_weight, kl_weight))
                or feature_weight + kl_weight == 0):
            raise ValueError("Loss weights must be finite, nonnegative, and not both zero")

    @staticmethod
    def _hidden(output):
        return output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k not in {"labels", "router_mask"}}
        inputs.update(use_cache=False, output_router_logits=False, return_dict=True)
        attention = inputs.get("attention_mask")
        runtime = self.model.skipping_controller
        # Backbone access avoids constructing [batch, sequence, vocabulary] logits.
        self.model.eval()
        if self.kl_weight:
            with runtime.context(mode="teacher", attention_mask=attention, reset=True), torch.no_grad():
                teacher = self._hidden(self.model.model(**inputs))
        with runtime.context(mode="reconstruction", threshold=self.threshold,
                             attention_mask=attention, reset=True):
            student = self._hidden(self.model.model(**inputs))
            device = student.device
            terms = list(runtime.feature_terms.values())
            numerator = sum(value.to(device) for value, _ in terms)
            denominator = sum(count.to(device) for _, count in terms)
            if not terms or denominator.item() == 0:
                raise ValueError("No valid local reconstruction positions")
            feature = numerator / denominator
            if self.kl_weight:
                kl = hidden_kl(student, teacher, self.model.lm_head, prediction_mask(batch),
                               self.temperature).to(device)
            else:
                kl = student.new_zeros((), dtype=torch.float32)
            loss = self.feature_weight * feature + self.kl_weight * kl
            # Keep all compensation parameters in the graph for distributed training.
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    loss = loss + parameter.sum().to(device) * 0
            counts = torch.stack([value.to(device) for value in runtime.statistics.values()]).sum(0)
            return {"loss": loss, "feature_loss": feature.detach(), "kl_loss": kl.detach(),
                    "counts": counts.detach(), "feature_positions": denominator.detach(),
                    "prediction_positions": prediction_mask(batch).sum().detach()}
