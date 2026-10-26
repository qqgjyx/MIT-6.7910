"""Input-derivative metrics of f(x), the loss at the true label(s) as a function of the input
(image for the CNN, token embeddings of one sequence for the GPT), at fixed test points."""

import torch
from torch.nn import functional as F

from ..models import GPT, double_backward_ok


def jacobian_fro(out_fn, x, exact, n_probes, gen):
    """||d out / d x||_F: exact (one VJP per output) or by Gaussian projections of the output."""
    x = x.detach().requires_grad_(True)
    out = out_fn(x)
    if exact:
        flat = out.flatten()
        sq = sum(torch.autograd.grad(flat[k], x, retain_graph=True)[0].pow(2).sum() for k in range(flat.numel()))
    else:
        sq = sum(torch.autograd.grad(out, x, torch.randn(out.shape, generator=gen, device=x.device, dtype=x.dtype),
                                     retain_graph=True)[0].pow(2).sum() for _ in range(n_probes)) / n_probes
    return sq.sqrt().item()


def tensor_norm(T, rand, n_starts, n_iter):
    """max_{|u|=1} T[u,u,u] (= max |T[u,u,u]|, as T is odd) by shifted symmetric power iteration."""
    best = float('-inf')
    for _ in range(n_starts):
        u = rand()
        u = u / u.norm()
        tu = T(u)
        alpha = tu.norm()
        for _ in range(n_iter):
            w = tu + alpha * u
            u = w / w.norm()
            tu = T(u)
        best = max(best, (tu * u).sum().item())
    return best


def derivatives(f, x):
    """g = grad f(x), and closures u -> H u and u -> T[u,u,.] = grad_x (u^T H(x) u)."""
    x = x.detach().requires_grad_(True)
    g = torch.autograd.grad(f(x), x, create_graph=True)[0]

    def H(u, create_graph=False):
        return torch.autograd.grad(g, x, u, retain_graph=True, create_graph=create_graph)[0]

    def T(u):
        return torch.autograd.grad((H(u, True) * u).sum(), x, retain_graph=True)[0]

    return g, H, T


def point_metrics(f, x, n_probes=8, n_iter=20, n_starts=3, gen=None):
    """Second- and third-order metrics of the scalar f at the point x."""
    g, H, T = derivatives(f, x)

    def rand():
        return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)

    g_norm = g.norm()
    g_hat = g / g_norm

    def P(u):
        return u - g_hat * (g_hat * u).sum()

    probes = [rand() for _ in range(n_probes)]
    hess_fro = torch.stack([H(z).pow(2).sum() for z in probes]).mean().sqrt()
    level = torch.stack([P(H(P(z))).pow(2).sum() for z in probes]).mean().sqrt() / g_norm

    v = rand()
    v = v / v.norm()
    for _ in range(n_iter):
        hv = H(v)
        v = hv / hv.norm()
    lam = (v * H(v)).sum()
    if (v * g).sum() < 0:
        v = -v
    tv = T(v)

    return dict(
        grad_norm=g_norm.item(),
        hess_fro=hess_fro.item(),
        hess_top=lam.item(),
        level_curv=level.item(),
        t_sigma=tensor_norm(T, rand, n_starts, n_iter),
        t_vvv=(tv * v).sum().item(),
        curv_length=(lam.abs() / tv.norm()).item(),
    )


def input_metrics(model, x, y, context=64, n_probes=8, n_iter=20, n_starts=3, seed=0):
    """Mean and std over the test points (rows of x) of the input-derivative metrics."""
    is_gpt = isinstance(model, GPT)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    rows = []
    try:
        with double_backward_ok():
            for i in range(len(x)):
                if is_gpt:
                    xi, yi = model.embed(x[i, :context]).detach(), y[i, :context]
                    def logits(e): return model.forward_embeds(e[None])[0]
                    def f(e): return F.cross_entropy(logits(e), yi)
                else:
                    xi, yi = x[i], y[i:i + 1]
                    def logits(e): return model(e[None])
                    def f(e): return F.cross_entropy(logits(e), yi)
                row = point_metrics(f, xi, n_probes, n_iter, n_starts, gen)
                row['jac_fro'] = jacobian_fro(lambda e: F.softmax(logits(e), -1), xi, not is_gpt, n_probes, gen)
                rows.append(row)
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
    vals = {k: torch.tensor([r[k] for r in rows], dtype=torch.float64) for k in rows[0]}
    return {k: {'mean': v.mean().item(), 'std': v.std().item() if len(v) > 1 else 0.0} for k, v in vals.items()}
