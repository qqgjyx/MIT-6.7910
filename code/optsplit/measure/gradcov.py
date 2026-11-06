"""Covariance of mini-batch gradients K, estimated as in Jastrzebski et al. (2020, App. C) from L
mini-batch gradients through their L x L centered Gram matrix."""

import torch

from .hessian import _grad, flat


def gradcov(model, params, batches, top=5):
    G = torch.stack([flat(_grad(model.loss(x, y), params)) for x, y in batches])
    L = G.shape[0]
    Gc = G - G.mean(0)
    evals, U = torch.linalg.eigh((Gc @ Gc.T / L).double())  # ascending; evals[0] ~ 0 (rank L-1)
    # Eigenvectors of K are Gc^T u / sqrt(L lambda); project each g_i on the top `top` of them.
    k = min(top, L - 1)
    lam, Uk = evals[-k:], U[:, -k:]
    proj = (Uk.T @ (Gc @ G.T).double()) / (L * lam).sqrt()[:, None]
    g_over_gk = (G.norm(dim=1).double() / proj.norm(dim=0)).mean()
    out = {'lambda_K': evals[-1].item(), 'lambda_K_min': evals[1].item(),
           'trace_K': evals.sum().item(), 'g_over_g5': g_over_gk.item()}
    del G, Gc
    return out
