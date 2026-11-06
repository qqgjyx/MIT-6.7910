"""Weight norms: what linear theory predicts each optimizer keeps small (l2 for GD, linf for Adam,
spectral for Muon), plus the norm-based measures compared by Jiang et al. (2020)."""

import copy
import math

import torch

from ..models import SimpleCNN


def _matrix(p):
    return p.detach().float().reshape(p.shape[0], -1)


@torch.no_grad()
def path_norm(model):
    """Jiang et al. (2020): sum of outputs for an all-ones input, with every parameter squared and
    identity activations. Defined for the SimpleCNN only."""
    if not isinstance(model, SimpleCNN):
        return None
    sq = copy.deepcopy(model).float()
    for p in sq.parameters():
        p.pow_(2)
    sq.act = lambda t: t
    return sq(torch.ones(1, 3, 32, 32, device=next(sq.parameters()).device)).sum().item()


@torch.no_grad()
def weight_metrics(model, init_state=None):
    params = {n: p.detach().float() for n, p in model.named_parameters()}
    flat = torch.cat([p.flatten() for p in params.values()])
    out = dict(
        l0=int((flat.abs() > 1e-8).sum()),
        l1=flat.abs().sum().item(),
        l2=flat.norm().item(),
        l3=flat.abs().pow(3).sum().pow(1 / 3).item(),
        linf=flat.abs().max().item(),
        path_norm=path_norm(model),
    )

    layers, log_prod, ratio_sum = {}, 0.0, 0.0
    for n, p in params.items():
        if p.ndim < 2:
            continue
        W = _matrix(p)
        s = torch.linalg.svdvals(W)
        layers[n] = dict(spectral=s[0].item(), nuclear=s.sum().item(),
                         stable_rank=(s.pow(2).sum() / s[0] ** 2).item())
        if init_state is not None:
            # Bartlett et al. (2017): prod ||W_i||_s * (sum (||(W_i - M_i)^T||_{2,1} / ||W_i||_s)^{2/3})^{3/2}
            D = W - _matrix(init_state[n].to(W.device))
            log_prod += math.log(s[0].item())
            ratio_sum += (D.norm(dim=1).sum().item() / s[0].item()) ** (2 / 3)
    out['layers'] = layers

    if init_state is not None:
        flat0 = torch.cat([init_state[n].to(flat.device).float().flatten() for n in params])
        out['dist_init'] = (flat - flat0).norm().item()
        out['dist_init_rel'] = out['dist_init'] / flat0.norm().item()
        out['log_spectral_complexity'] = log_prod + 1.5 * math.log(ratio_sum) if ratio_sum > 0 else float('-inf')
    return out
