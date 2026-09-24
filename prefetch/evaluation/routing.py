from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict

import torch

from prefetch.backbone.structure import choice_scores
from prefetch.evaluation.metrics import report_histograms


class RoutingMetrics:
    """Coverage/trace accumulator, updated explicitly from a completed forward."""
    phases = ("teacher_forcing", "decode")

    def __init__(self, state):
        self.config, self.layers = state.config, state.layers
        block = state.blocks[state.pairs[0][1]][1]
        self.counts = torch.zeros(len(self.phases), len(state.pairs), 2, block.gate.weight.shape[0] + 1,
                                  dtype=torch.int64, device=block.gate.weight.device)
        self.trace = []

    def reset(self):
        self.counts.zero_()
        self.trace.clear()

    @torch.no_grad()
    def update(self, state, router_mask=None):
        if state.phase == "prefill":
            return
        mask = state.valid_mask
        if state.phase == "teacher_forcing":
            router_mask = state.router_mask if router_mask is None else router_mask
            if router_mask is None:
                raise ValueError("Teacher-forcing metrics require a response router_mask")
            mask = mask & router_mask[0, state.target_start:].bool()
        for slot, (source, target) in enumerate(state.pairs):
            truth = state.router_indices[target][mask]
            if not truth.numel():
                continue
            scores = choice_scores(state.predictions[target][mask], state.blocks[target][1])
            order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order, torch.arange(1, order.shape[-1] + 1,
                           device=order.device).expand_as(order))
            true_ranks = ranks.gather(1, truth)
            bins = self.counts.shape[-1]
            counts = self.counts[self.phases.index(state.phase), slot]
            counts[0] += torch.bincount(true_ranks.flatten(), minlength=bins)
            counts[1] += torch.bincount(true_ranks.amax(dim=1), minlength=bins)

            remaining = self.config.trace_limit - sum(row["phase"] == state.phase for row in self.trace)
            if remaining <= 0:
                continue
            k = min(max(self.config.ks), scores.shape[-1])
            positions = mask.nonzero().flatten()[:remaining].tolist()
            predicted = order[:remaining, :k].tolist()
            for position, prediction, actual in zip(positions, predicted, truth[:remaining].tolist()):
                token = position + state.target_start + state.token_offset
                self.trace.append(dict(phase=state.phase, forward_index=state.forward_index,
                                       batch_index=0, target_token=token,
                                       source_token=token - int(self.config.mode == "previous_token"),
                                       source_moe=source, target_moe=target,
                                       predictions=prediction, truth=actual))

    def report(self, distributed=False):
        metadata = dict(config=asdict(self.config), layers=self.layers,
                        teacher="native_router_on_current_hidden_states",
                        execution=("prerouter_indices_native_weights" if self.config.prerouter_enabled
                                   else "native_router"),
                        execution_scope="decode_and_teacher_forced_response",
                        metric_scope=dict(teacher_forcing="response_text_inputs", decode="all_forwarded_inputs"),
                        producer_timing="after_source_routing_before_experts")
        counts = self.counts.clone()
        if distributed:
            torch.distributed.all_reduce(counts)
        return report_histograms(dict(zip(self.phases, counts.cpu().tolist())), metadata, self.config.ks)


@contextmanager
def capture_generation(model, state, meter):
    """Attach an evaluation observer for every forward in one generate() request."""
    state.reset(generation=True)
    meter.trace.clear()
    handle = model.register_forward_hook(lambda module, args, output: meter.update(state))
    try:
        yield
    finally:
        handle.remove()
        state.reset()
