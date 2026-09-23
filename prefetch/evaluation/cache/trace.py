"""Observe actual expert calls without applying routing-loss masks."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from prefetch.backbone.structure import choice_scores


class DecodeTrace:
    def __init__(self, state, prediction_k=8):
        if state.config.mode != "same_token":
            raise ValueError("Cache traces currently require same_token prerouters")
        self.state, self.prediction_k = state, prediction_k
        sources = {target: source for source, target in state.pairs}
        self.layers = [dict(moe=index, name=name, experts=block.gate.weight.shape[0],
                            top_k=block.top_k, source_moe=sources.get(index))
                       for index, (name, block) in enumerate(state.blocks)]
        if prediction_k < 1 or any(prediction_k > layer["experts"] for layer in self.layers):
            raise ValueError("prediction_k must fit the expert count")

    def _observer(self, index, block):
        @torch.no_grad()
        def observe(module, args):
            if self.state.phase != "decode":
                return
            # Both native and NF4 expert modules receive the actual indices here.
            truth = args[1]
            if truth.shape != (1, block.top_k):
                raise ValueError("Cache tracing requires single-token, batch-size-one decode")
            entry = self.records[index]
            entry["truth"].append(truth[0].detach().cpu().tolist())
            if entry["prediction"] is not None:
                scores = choice_scores(self.state.predictions[index], block)
                order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
                entry["prediction"].append(order[0, :self.prediction_k].cpu().tolist())
        return observe

    @contextmanager
    def capture(self, model):
        self.records = [dict(moe=layer["moe"], truth=[],
                             prediction=[] if layer["source_moe"] is not None else None)
                        for layer in self.layers]
        self.state.reset(generation=True)
        handles = []
        try:
            for index, (_, block) in enumerate(self.state.blocks):
                handles.append(block.experts.register_forward_pre_hook(self._observer(index, block)))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.state.reset()

    def request(self, **metadata):
        tokens = len(self.records[0]["truth"])
        if any(len(row["truth"]) != tokens or
               (row["prediction"] is not None and len(row["prediction"]) != tokens)
               for row in self.records):
            raise ValueError("Incomplete decode trace: every layer must contain every decode call")
        return dict(type="request", **metadata, decode_tokens=tokens, layers=self.records)
