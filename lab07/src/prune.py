import torch

from models import TeacherCNN, weight_layers


def _keep_largest(scores, n_drop):
    mask = torch.ones_like(scores)
    if n_drop > 0:
        mask[torch.topk(scores, n_drop, largest=False).indices] = 0
    return mask


def global_masks(model, sparsity):
    layers = weight_layers(model)
    scores = torch.cat([m.weight.detach().abs().flatten() for _, m in layers])
    flat = _keep_largest(scores, round(sparsity * scores.numel()))
    parts = flat.split([m.weight.numel() for _, m in layers])
    return {n: p.view_as(m.weight) for (n, m), p in zip(layers, parts)}


def layer_masks(model, sparsity):
    out = {}
    for n, m in weight_layers(model):
        scores = m.weight.detach().abs().flatten()
        out[n] = _keep_largest(scores, round(sparsity * scores.numel())).view_as(m.weight)
    return out


MASKS = {"global": global_masks, "layer": layer_masks}


def apply_masks(model, masks):
    with torch.no_grad():
        for n, m in weight_layers(model):
            m.weight.mul_(masks[n])


def sparsity_report(model):
    layers = weight_layers(model)
    per = {n: float((m.weight == 0).sum().item() / m.weight.numel()) for n, m in layers}
    zeros = sum(int((m.weight == 0).sum()) for _, m in layers)
    return zeros / sum(m.weight.numel() for _, m in layers), per


CONVS = ("conv1", "conv2", "conv3", "conv4")


def filters_to_keep(model, fraction):
    keep = {}
    for n, m in weight_layers(model):
        if n in CONVS:
            norms = m.weight.detach().flatten(1).norm(dim=1)
            n_drop = round(fraction * len(norms))
            mask = _keep_largest(norms, n_drop)
            keep[n] = mask.nonzero().flatten()
    return keep


def zero_filters(model, keep):
    with torch.no_grad():
        for n, m in weight_layers(model):
            if n in CONVS:
                drop = torch.ones(m.out_channels, dtype=torch.bool, device=m.weight.device)
                drop[keep[n]] = False
                m.weight[drop] = 0
                m.bias[drop] = 0


def remove_filters(model, keep, dropout):
    new = TeacherCNN([len(keep[c]) for c in CONVS], model.hidden, dropout).to(next(model.parameters()).device)
    old_l, new_l = dict(weight_layers(model)), dict(weight_layers(new))
    with torch.no_grad():
        prev = None
        for c in CONVS:
            w = old_l[c].weight[keep[c]]
            if prev is not None:
                w = w[:, prev]
            new_l[c].weight.copy_(w)
            new_l[c].bias.copy_(old_l[c].bias[keep[c]])
            prev = keep[c]
        cols = (prev[:, None] * 49 + torch.arange(49, device=prev.device)[None]).flatten()
        new_l["fc1"].weight.copy_(old_l["fc1"].weight[:, cols])
        new_l["fc1"].bias.copy_(old_l["fc1"].bias)
        new_l["fc2"].weight.copy_(old_l["fc2"].weight)
        new_l["fc2"].bias.copy_(old_l["fc2"].bias)
    return new
