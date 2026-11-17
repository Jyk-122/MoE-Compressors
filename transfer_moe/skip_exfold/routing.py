"""Per-token ExFold. Tables are indexed [omitted expert, retained expert]."""


def validate_topk(topk, native_topk):
    if type(topk) is not int or not 1 <= topk <= native_topk:
        raise ValueError(f"topk must be an integer in [1, {native_topk}]")


def fold_routes(ids, weights, scale, loss, norm, topk, *,
                loss_threshold=0.99, renormalize=True):
    """Select by w*h, then fold each omitted route into its best retained one.

    Both prefill and decode use this rule. Loss filtering and weight-sum
    normalization follow the Qwen3.5 prefill recipe, applied in both phases.
    Set loss_threshold=None and renormalize=False for plain scalar folding.
    """
    if topk == ids.shape[1]:
        return ids, weights
    order = (weights * norm[ids]).argsort(dim=-1, descending=True, stable=True)
    selected = order[:, :topk]
    kept_ids = ids.gather(1, selected)
    kept_weights = weights.gather(1, selected).clone()
    omitted = order[:, topk:]
    omitted_ids = ids.gather(1, omitted)
    costs = loss[omitted_ids.unsqueeze(2), kept_ids.unsqueeze(1)]
    best_loss, owner = costs.min(-1)
    destination = kept_ids.gather(1, owner)
    contribution = weights.gather(1, omitted) * scale[omitted_ids, destination]
    if loss_threshold is not None:
        contribution = contribution * (best_loss <= loss_threshold)
    kept_weights.scatter_add_(1, owner, contribution)
    if renormalize:
        kept_weights *= weights.sum(-1, keepdim=True) / kept_weights.sum(-1, keepdim=True)
    return kept_ids, kept_weights
