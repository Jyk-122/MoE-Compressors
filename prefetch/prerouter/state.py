"""Global routing data and layer/token handoff, for batch size 1."""
from __future__ import annotations

import torch

from prefetch.prerouter.configuration import layer_pairs


class PrerouterState:
    def __init__(self, blocks, config):
        self.blocks, self.config = blocks, config
        self.pairs = layer_pairs(len(blocks), config.mode, config.distance, config.targets)
        self.target_of = dict(self.pairs)
        self.targets = {target for _, target in self.pairs}
        self.prerouters = {}  # source MoE -> nn.Module (registered on that MoE)
        self.layers = [dict(source_moe=source, target_moe=target,
                            source_name=blocks[source][0], target_name=blocks[target][0],
                            experts=blocks[target][1].gate.weight.shape[0],
                            top_k=blocks[target][1].top_k) for source, target in self.pairs]
        self.reset()

    def reset(self, generation=False, train_prerouter=False, router_mask=None):
        """Call before each independent training sample or generation request."""
        self.generation, self.train_prerouter = generation, train_prerouter
        self.active = False
        self.phase = "teacher_forcing"
        self.forward_index = -1
        self.token_offset = self.sequence_length = 0
        self.router_mask = router_mask  # [1,S] response-input mask for teacher forcing.
        self.target_start = 0
        self.valid_mask = None
        self.predictions = {}      # target -> [T,E]; T=S-1 for previous-token teacher forcing
        self.router_logits = {}    # target -> [T,E], detached native gate output
        self.router_indices = {}   # target -> [T,K], native selection on current hidden states
        self.next_predictions = {} # target -> [1,E], for the next decode token

    def begin_forward(self, input_ids, attention_mask=None):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Prerouter requires input_ids [1, sequence]; batch size must be 1")
        self.forward_index += 1
        self.phase = ("prefill" if self.forward_index == 0 else "decode") if self.generation else "teacher_forcing"
        if self.phase == "decode" and input_ids.shape[1] != 1:
            raise ValueError("Generation requires cached single-token decode and num_beams=1")
        self.token_offset = self.token_offset + self.sequence_length if self.phase == "decode" else 0
        self.sequence_length = input_ids.shape[1]
        valid = (torch.ones_like(input_ids[0], dtype=torch.bool) if attention_mask is None
                 else attention_mask[0, -self.sequence_length:].bool())
        previous = self.config.mode == "previous_token"
        self.target_start = int(previous and self.phase == "teacher_forcing")
        self.valid_mask = valid[self.target_start:].clone()
        if self.target_start:
            self.valid_mask &= valid[:-1]
        if self.phase == "teacher_forcing":
            for token_id in self.config.excluded_token_ids:
                self.valid_mask &= input_ids[0, self.target_start:] != token_id
            if self.config.prerouter_enabled and self.router_mask is None:
                raise ValueError("Teacher-forcing expert substitution requires state.reset(router_mask=response_mask)")

        self.predictions = self.next_predictions if previous and self.phase == "decode" else {}
        self.next_predictions = {}
        self.router_logits, self.router_indices = {}, {}
        self.active = True

    def publish(self, source, logits):
        """Pair source[:-1] with target[1:], or hand one prediction to the next decode."""
        target = self.target_of[source]
        if self.config.mode == "same_token":
            self.predictions[target] = logits
        else:
            if self.phase == "teacher_forcing":
                self.predictions[target] = logits[:-1]
            if self.generation:
                self.next_predictions[target] = logits[-1:].detach().clone()

    def record(self, target, logits, indices):
        if self.phase != "prefill" and target in self.targets:
            self.router_logits[target] = logits[self.target_start:].detach()
            self.router_indices[target] = indices[self.target_start:].detach()

    def parameters(self):
        # Pair order also defines the optimizer/checkpoint parameter order.
        for head in self.prerouters.values():
            yield from head.parameters()
