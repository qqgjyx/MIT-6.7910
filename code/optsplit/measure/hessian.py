"""Hessian of the training loss: Lanczos eigenpairs, AdamW-preconditioned sharpness, batch sharpness
and the block anatomy of the top eigenvector. Vectors over parameters are flat float32 tensors in
`model.named_parameters()` order (trainable parameters only)."""

import math

import torch


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def split(v, params):
    out, i = [], 0
    for p in params:
        out.append(v[i:i + p.numel()].view_as(p))
        i += p.numel()
    return out


def _grad(out, params, **kw):
    gs = torch.autograd.grad(out, params, allow_unused=True, **kw)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, gs)]


class HVP:
    """Hessian-vector products of `model.loss(x, y)`, reusing one gradient graph."""

    def __init__(self, model, x, y, params):
        self.params = params
        self.grads = _grad(model.loss(x, y), params, create_graph=True)

    def __call__(self, v):
        dot = sum((g * u).sum() for g, u in zip(self.grads, split(v, self.params)) if g.requires_grad)
        if not torch.is_tensor(dot) or not dot.requires_grad:
            return torch.zeros_like(v)
        return flat(_grad(dot, self.params, retain_graph=True))


def lanczos(matvec, dim, device, iters=20, n_eigs=1, seed=0):
    """Top `n_eigs` (largest algebraic) eigenpairs of a symmetric operator on R^dim, by Lanczos with
    full reorthogonalization. Returns (eigenvalues list, eigenvectors (dim, n_eigs))."""
    k = min(iters, dim)
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(dim, generator=g, device=device)
    q /= q.norm()
    Q = torch.zeros(k, dim, device=device)
    alpha, beta = [], []
    for j in range(k):
        Q[j] = q
        w = matvec(q)
        alpha.append(torch.dot(w, q).item())
        w -= alpha[-1] * q
        if j > 0:
            w -= beta[-1] * Q[j - 1]
        for _ in range(2):
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        b = w.norm().item()
        if j == k - 1 or b < 1e-10:
            break
        beta.append(b)
        q = w / b
    m = len(alpha)
    T = torch.diag(torch.tensor(alpha, dtype=torch.float64))
    if m > 1:
        off = torch.tensor(beta[:m - 1], dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    if not torch.isfinite(T).all():  # the run is diverging; train.py stops it at the next log step
        return [math.nan] * n_eigs, torch.full((dim, n_eigs), math.nan, device=device)
    evals, evecs = torch.linalg.eigh(T)
    idx = torch.arange(m - 1, max(m - 1 - n_eigs, -1), -1)
    vecs = Q[:m].T @ evecs[:, idx].to(Q)
    return evals[idx].tolist(), vecs / vecs.norm(dim=0, keepdim=True)


def anatomy(v, matvec, names, blocks, params):
    """Per block b: its share of the top eigenvector, ||v_b||^2, and its own curvature along it,
    v_b^T H_bb v_b. (v_b^T (Hv)_b would just be lambda ||v_b||^2.) `cross` = v^T H v - sum_b self_b is
    the part of the eigenvalue that comes from interactions between blocks."""
    index = {n: i for i, n in enumerate(names)}
    offsets = [0]
    for p in params:
        offsets.append(offsets[-1] + p.numel())
    out = {'dir': {}, 'self': {}}
    for b, members in blocks.items():
        ids = [index[n] for n in members if n in index]
        if ids:
            vb = torch.zeros_like(v)
            for i in ids:
                vb[offsets[i]:offsets[i + 1]] = v[offsets[i]:offsets[i + 1]]
            out['dir'][b] = vb.pow(2).sum().item()
            out['self'][b] = torch.dot(vb, matvec(vb)).item()
    out['cross'] = torch.dot(v, matvec(v)).item() - sum(out['self'].values())
    return out


def adam_preconditioner(optimizer, params):
    """Flat diagonal P = (1 - beta1^t)(sqrt(v_hat) + eps) of AdamW (Cohen et al. 2022), or None
    before the first step."""
    group_of = {p: g for g in optimizer.param_groups for p in g['params']}
    diag = []
    for p in params:
        state = optimizer.state.get(p, {})
        if 'exp_avg_sq' not in state:
            return None
        g = group_of[p]
        beta1, beta2 = g['betas']
        t = float(state['step'])
        v_hat = state['exp_avg_sq'] / (1 - beta2 ** t)
        diag.append((1 - beta1 ** t) * (v_hat.sqrt() + g['eps']))
    return flat(diag).float()


def batch_sharpness(model, params, batches):
    """E_B[g_B^T H_B g_B] / E_B[||g_B||^2] over mini-batches (Andreyev & Beneventano 2024)."""
    num = den = 0.0
    for x, y in batches:
        grads = _grad(model.loss(x, y), params, create_graph=True)
        gd = [g.detach() for g in grads]
        dot = sum((g * u).sum() for g, u in zip(grads, gd) if g.requires_grad)
        hg = _grad(dot, params)
        num += sum((u * h).sum() for u, h in zip(gd, hg)).item()
        den += sum(u.pow(2).sum() for u in gd).item()
        del grads, dot, hg
    return num / den
