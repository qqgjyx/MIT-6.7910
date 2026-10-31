"""Curvature-balanced learning rates. Block a (a layer: embeddings, each transformer block, final norm + head;
`balance.blocks: sublayer` splits each block into attention and MLP) trains at eta_a(t) = eta(t) g m_a, with eta(t)
the run's schedule. Every `balance.every` steps, the top eigenpairs of Hhat = D^1/2 H D^1/2, D = diag(eta_a)
(AdamW: diag(eta_a) P^-1, P frozen from the current second moments), H on the Hessian batch, are refreshed by a
warm-started block iteration. With v the top eigenvector, s_a = ||v_a||^2 = d log lambda_max / d log m_a and
sum_a s_a = 1; near-ties are handled by averaging the shares of the Ritz vectors within `balance.cluster` of the
top. The shares are smoothed by an EMA, then
  A: log m_a -= alpha (s_a - 1/L), recentred to mean 0 (equal shares maximize prod_a m_a at fixed lambda_max;
     blocks in `balance.fixed` keep m_a = g = 1 and the others equalize their shares among themselves);
  B: A, and log g -= gain log(lambda_max / (target * threshold)), with the linear-stability threshold of Hhat:
     2 + 2 mu for SGD (heavy ball: buf = mu buf + grad, step lr buf), (2 + 2 beta1) / (1 - beta1) for AdamW.
Every g m_a is clipped to `balance.clip`. Before `balance.start` (default: the end of warmup) the multipliers stay
at 1, while the eigenvectors and the EMA are already tracked.

Muon's normalized step is stable when the spectral-norm sharpness S = max <D, H D> over ||D|| <= 1 satisfies
rho = eta S / (2 sum_l c_l ||G_l||_*) <= 1 (Islamov et al. 2026), with ||D|| = max_l ||D_l||_2 / c_l over the matrices
Muon updates and c_l = m_a s_l (s_l Muon's shape factor), so that the step -eta c_l polar(G_l) has norm eta; G is the
gradient on the Hessian batch. For Muon the balancer only rescales the Muon groups (one block per transformer block),
tracks the maximizer D by a warm-started Frank-Wolfe (`balance.fw_iters` steps per refresh) and uses
sigma_a = d log rho / d log m_a = 2 <D_a, (H D)_a> / <D, H D> - c-weighted share of sum_l c_l ||G_l||_* in block a
(sum_a sigma_a = 1) in place of s_a, and rho in place of lambda_max / threshold."""

import math

import torch

from ..measure.hessian import adam_preconditioner, flat, lanczos, split
from ..models import double_backward_ok, layer
from .muon import zeropower_via_newtonschulz5


def threshold(cfg):
    o = cfg.optim[cfg.optimizer]
    if cfg.optimizer == 'sgd':
        assert not o.nesterov, 'the threshold is for heavy ball'
        return 2 + 2 * o.momentum
    if cfg.optimizer == 'adamw':
        return (2 + 2 * o.betas[0]) / (1 - o.betas[0])
    raise ValueError(cfg.optimizer)


def block_iteration(op, X, iters):
    """Top eigenpairs of the symmetric `op` from the block X (dim, k): `iters` times Rayleigh-Ritz on X and its
    residuals (LOBPCG without the conjugate directions), k (iters + 1) products. -> (Ritz values, descending;
    orthonormal Ritz vectors)."""
    X = torch.linalg.qr(X)[0]
    AX = torch.stack([op(x) for x in X.T], 1)
    for _ in range(iters):
        W = AX
        for _ in range(2):
            W = W - X @ (X.T @ W)
        W = torch.linalg.qr(W)[0]
        S, AS = torch.cat([X, W], 1), torch.cat([AX, torch.stack([op(w) for w in W.T], 1)], 1)
        T = (S.T @ AS).double()
        theta, U = torch.linalg.eigh((T + T.T) / 2)
        theta, U = theta.flip(0)[:X.shape[1]], U.flip(1)[:, :X.shape[1]].to(X)
        X, AX = S @ U, AS @ U
    return theta.tolist(), X


def batched(fn, mats):
    """[fn(m) for m in mats], with one call of fn per shape on the stacked matrices of that shape."""
    out, ids = [None] * len(mats), {}
    for i, m in enumerate(mats):
        ids.setdefault(tuple(m.shape), []).append(i)
    for js in ids.values():
        for j, r in zip(js, fn(torch.stack([mats[j] for j in js]))):
            out[j] = r
    return out


def frank_wolfe(hvp, D, c, iters, k0=0):
    """Frank-Wolfe for max <D, H D> over max_l ||D_l||_2 / c_l <= 1 from the matrices D, steps 2 / (k + k0 + 2).
    -> (best <D, H D> at norm 1, its D, H D)."""
    best = (-math.inf, None, None)
    for k in range(iters + 1):
        n = max(torch.stack(batched(lambda x: torch.linalg.matrix_norm(x, 2), D)) / torch.tensor(c).to(D[0])).item()
        D = [d / n for d in D]
        HD = split(hvp(flat(D)), D)
        q = sum((d * h).sum() for d, h in zip(D, HD)).item()
        if q > best[0]:
            best = (q, D, HD)
        if k < iters:
            gam = 2 / (k + k0 + 2)
            P = batched(lambda x: zeropower_via_newtonschulz5(x, 5).float(), HD)
            D = [(1 - gam) * d + gam * cl * p for d, p, cl in zip(D, P, c)]
    return best


def shares(X, theta, bid, n_blocks, cluster):
    """Mean over the Ritz vectors within `cluster` (relative) of the top of each block's share ||x_a||^2."""
    top = [i for i, t in enumerate(theta) if t >= theta[0] - cluster * abs(theta[0])]
    s = torch.zeros(n_blocks, device=X.device).index_add_(0, bid, X[:, top].pow(2).sum(1))
    return (s / s.sum()).double().cpu()


class Balancer:
    def __init__(self, cfg, opt, named, hvp):
        """`named`: the trainable (name, parameter) pairs, in the order of `hvp`'s flat vectors; `hvp()` builds a
        Hessian-vector product at the current parameters. The groups of `opt` name their layer in `block`."""
        b = self.b = cfg.balance
        self.cfg, self.opt, self.hvp = cfg, opt, hvp
        self.params = [p for _, p in named]
        self.blocks = list(dict.fromkeys(layer(n, b.blocks) for n, _ in named))
        dev = self.params[0].device
        self.bid = torch.cat([torch.full((p.numel(),), self.blocks.index(layer(n, b.blocks)), device=dev)
                              for n, p in named])
        self.muon = cfg.optimizer == 'muon'
        self.eta, self.thr = cfg.optim[cfg.optimizer].lr, 1.0 if self.muon else threshold(cfg)
        self.start = cfg.schedule.warmup_steps if b.start is None else b.start
        self.log_m, self.log_g, self.ema = torch.zeros(len(self.blocks), dtype=torch.float64), 0.0, None
        self.free = torch.tensor([n not in b.get('fixed', []) for n in self.blocks])
        gen = torch.Generator(device=dev).manual_seed(0)
        if self.muon:  # one block per matrix; the tracked state is the maximizer D
            self.pid = [self.blocks.index(layer(n, b.blocks)) for n, _ in named]
            self.shape = [max(1, p.shape[0] / p.shape[1]) ** 0.5 for p in self.params]
            self.X = [torch.randn(p.shape, generator=gen, device=dev) for p in self.params]
        else:
            self.X = torch.randn(len(self.bid), b.k, generator=gen, device=dev)

    def rates(self):
        """Each block's multiplier g m_a of the schedule's rate (1 for the blocks in `balance.fixed`)."""
        return torch.where(self.free, (self.log_m + self.log_g).exp().clamp(*self.b.clip), 1.0)

    def apply(self):
        r = self.rates()
        for g in self.opt.param_groups:
            if g.get('use_muon', True):  # Muon's auxiliary AdamW keeps the schedule's rate
                g['lr_mult'] = r[self.blocks.index(g['block'])].item()

    def step(self, step, mult):
        """Before the update of `step`, whose schedule multiplier is `mult`: every `every` steps, refresh the
        shares and the multipliers. -> record to log, or None."""
        b, L = self.b, len(self.blocks)
        if step % b.every:
            return None
        out = self.spectral(step, mult) if self.muon else self.hhat(step, mult)
        if out is None:
            return None
        lam, new, rec = out
        if not (math.isfinite(lam) and lam > 0 and torch.isfinite(new).all()):
            return dict(lam=lam, **rec)  # diverging; train.py stops at the next log step
        self.ema = new if self.ema is None else self.ema + b.ema * (new - self.ema)
        if step >= self.start:
            lo, hi, f = *map(math.log, b.clip), self.free
            self.log_m = torch.where(f, self.log_m - b.alpha * (self.ema - self.ema[f].mean()), 0.0)
            self.log_m = torch.where(f, self.log_m - self.log_m[f].mean(), 0.0).clamp(lo, hi)
            if b.variant == 'B':
                self.log_g = min(hi, max(lo, self.log_g - b.gain * math.log(lam / (b.target * self.thr))))
        self.apply()
        named = lambda t: dict(zip(self.blocks, t.tolist()))  # noqa: E731
        return dict(lam=lam, thr=self.thr, shares=named(new), ema=named(self.ema), m=named(self.log_m.exp()),
                    g=math.exp(self.log_g), rate=named(self.rates()), **rec)

    def hhat(self, step, mult):
        """SGD, AdamW: -> (lambda_max(Hhat), shares, record) or None."""
        b, L = self.b, len(self.blocks)
        d = self.eta * mult * self.rates().float().to(self.bid.device)[self.bid]
        if self.cfg.optimizer == 'adamw':
            P = adam_preconditioner(self.opt, self.params)
            if P is None:
                return None
            d = d / P
        s = d.sqrt()
        rec = {}
        with double_backward_ok(), torch.autocast(s.device.type, enabled=False):
            hvp = self.hvp()

            def op(u):
                return s * hvp(s * u)

            theta, self.X = block_iteration(op, self.X, b.iters)
            if step % b.check_every == 0:
                lam, v = lanczos(op, len(s), s.device, self.cfg.measure.lanczos_iters, 1)
                rec = dict(check=lam[0], check_shares=dict(zip(self.blocks, shares(v, lam, self.bid, L, 0).tolist())))
            del hvp
        return theta[0], shares(self.X, theta, self.bid, L, b.cluster), dict(ritz=theta, **rec)

    def spectral(self, step, mult):
        """Muon: -> (rho, sigma, record)."""
        b, L = self.b, len(self.blocks)
        r = self.rates().tolist()
        c = [r[i] * s for i, s in zip(self.pid, self.shape)]
        eta = self.eta * mult
        rec = {}
        with double_backward_ok(), torch.autocast(self.params[0].device.type, enabled=False):
            hvp = self.hvp()
            nuc = [cl * n.item() for n, cl in zip(batched(lambda x: torch.linalg.matrix_norm(x, 'nuc'),
                                                          [g.detach() for g in hvp.grads]), c)]
            q, self.X, HD = frank_wolfe(hvp, self.X, c, b.fw_iters, k0=2)
            if step % b.check_every == 0:  # from scratch, as optsplit.analysis.spectral_sharpness
                gen = torch.Generator(device=HD[0].device).manual_seed(step)
                qs = [frank_wolfe(hvp, [torch.randn(d.shape, generator=gen, device=d.device) for d in HD], c, 30)[0]
                      for _ in range(2)]
                rec = dict(check=eta * max(qs) / (2 * sum(nuc)))
            del hvp
        row = torch.zeros(L, dtype=torch.float64).index_add_(
            0, torch.tensor(self.pid), torch.tensor([(d * h).sum().item() for d, h in zip(self.X, HD)], dtype=torch.float64))
        dual = torch.zeros(L, dtype=torch.float64).index_add_(0, torch.tensor(self.pid), torch.tensor(nuc, dtype=torch.float64))
        sigma = 2 * row / q - dual / dual.sum()
        return eta * q / (2 * sum(nuc)), sigma, dict(S=q, nuc=sum(nuc), **rec)

    def state_dict(self):
        return dict(log_m=self.log_m, log_g=self.log_g, ema=self.ema, X=self.X)

    def load(self, state):
        self.log_m, self.log_g, self.X = state['log_m'].cpu(), state['log_g'], state['X']  # loaded onto the GPU
        self.ema = None if state['ema'] is None else state['ema'].cpu()
        self.apply()
