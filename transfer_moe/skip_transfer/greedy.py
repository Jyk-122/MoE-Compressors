"""Batched forward greedy selection. All decisions stay on the tensor device."""

import torch


def _greedy_curve(top_ids, weights, error, steps):
    """Record the first `steps` greedy stages using D[source, target]."""
    tokens = top_ids.shape[0]
    costs = error[top_ids.unsqueeze(2), top_ids.unsqueeze(1)]
    rows = torch.arange(tokens, device=top_ids.device)
    selected = torch.zeros_like(top_ids, dtype=torch.bool)

    chosen = (costs * weights.unsqueeze(1)).sum(-1).argmin(-1)
    best_error = costs[rows, chosen]
    assignment = chosen.unsqueeze(1).expand_as(top_ids)
    selected.scatter_(1, chosen.unsqueeze(1), True)
    order = [chosen]
    assignments = [assignment]
    curve = [(weights * best_error).sum(-1)]

    for _ in range(1, steps):
        gains = ((best_error.unsqueeze(1) - costs).clamp_min(0)
                 * weights.unsqueeze(1)).sum(-1)
        chosen = gains.masked_fill(selected, -torch.inf).argmax(-1)
        candidate_error = costs[rows, chosen]
        assignment = torch.where(candidate_error < best_error, chosen.unsqueeze(1), assignment)
        best_error = torch.minimum(best_error, candidate_error)
        selected.scatter_(1, chosen.unsqueeze(1), True)
        order.append(chosen)
        assignments.append(assignment)
        curve.append((weights * best_error).sum(-1))

    return torch.stack(order, 1), torch.stack(assignments, 1), torch.stack(curve, 1)


def greedy_select_topk(top_ids, weights, error, topk):
    """Keep exactly topk experts (1 <= topk <= K), stopping after topk stages.

    Returns retained [tokens, K], assignment [tokens, K], cost [tokens].
    retained and assignment use original Top-K slots, as does the topp API.
    """
    order, assignments, costs = _greedy_curve(top_ids, weights, error, topk)
    retained = torch.zeros_like(top_ids, dtype=torch.bool).scatter_(1, order, True)
    return retained, assignments[:, -1], costs[:, -1]


def greedy_select_topp(top_ids, weights, error, topp_threshold):
    """Choose the largest budget with 1 - J_k / J_1 <= topp_threshold.

    topp_threshold is in [0, 1]; larger values retain more experts. A zero
    J_1 already has zero proxy error, so keep one expert. Returns
    retained [tokens, K], assignment [tokens, K], absolute cost [tokens].
    """
    order, assignments, costs = _greedy_curve(top_ids, weights, error, top_ids.shape[1])
    initial = costs[:, :1]
    # Count admissible additions after the first expert, without dividing by J_1.
    stage = ((costs[:, 1:] >= (1 - topp_threshold) * initial) & (initial > 0)).sum(-1)
    prefix = torch.arange(top_ids.shape[1], device=top_ids.device).unsqueeze(0) <= stage.unsqueeze(1)
    retained = torch.zeros_like(top_ids, dtype=torch.bool).scatter_(1, order, prefix)
    rows = torch.arange(top_ids.shape[0], device=top_ids.device)
    return retained, assignments[rows, stage], costs[rows, stage]
