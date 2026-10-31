"""Curvature-balanced learning rates. Block a (a layer: embeddings, each transformer block, final norm + head;
`balance.blocks: sublayer` splits each block into attention and MLP) trains at eta_a(t) = eta(t) g m_a, with eta(t)
the run's schedule. Every `balance.every` steps, the top eigenpairs of Hhat = D^1/2 H D^1/2, D = diag(eta_a)
(AdamW: diag(eta_a) P^-1, P frozen from the current second moments), H on the Hessian batch, are refreshed by a
warm-started block iteration. With v the top eigenvector, s_a = ||v_a||^2 = d log lambda_max / d log m_a and
sum_a s_a = 1; near-ties are handled by averaging the shares of the Ritz vectors within `balance.cluster` of the
top. The shares are smoothed by an EMA, then
  A: log m_a -= alpha (s_a - 1/L), recentred to mean 0 (equal shares maximize prod_a m_a at fixed lambda_max);
  B: A, and log g -= gain log(lambda_max / (target * threshold)), with the linear-stability threshold of Hhat:
     2 + 2 mu for SGD (heavy ball: buf = mu buf + grad, step lr buf), (2 + 2 beta1) / (1 - beta1) for AdamW.
Every g m_a is clipped to `balance.clip`. Before `balance.start` (default: the end of warmup) the multipliers stay
at 1, while the eigenvectors and the EMA are already tracked."""

import math

import torch

from ..measure.hessian import adam_preconditioner, lanczos
from ..models import double_backward_ok, layer


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
        self.eta, self.thr = cfg.optim[cfg.optimizer].lr, threshold(cfg)
        self.start = cfg.schedule.warmup_steps if b.start is None else b.start
        self.log_m, self.log_g, self.ema = torch.zeros(len(self.blocks), dtype=torch.float64), 0.0, None
        self.X = torch.randn(len(self.bid), b.k, generator=torch.Generator(device=dev).manual_seed(0), device=dev)

    def rates(self):
        """Each block's multiplier g m_a of the schedule's rate."""
        return (self.log_m + self.log_g).exp().clamp(*self.b.clip)

    def apply(self):
        r = self.rates()
        for g in self.opt.param_groups:
            g['lr_mult'] = r[self.blocks.index(g['block'])].item()

    def step(self, step, mult):
        """Before the update of `step`, whose schedule multiplier is `mult`: every `every` steps, refresh the
        shares and the multipliers. -> record to log, or None."""
        b, L = self.b, len(self.blocks)
        if step % b.every:
            return None
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
        new = shares(self.X, theta, self.bid, L, b.cluster)
        if not (math.isfinite(theta[0]) and theta[0] > 0 and torch.isfinite(new).all()):
            return dict(lam=theta[0], ritz=theta, **rec)  # diverging; train.py stops at the next log step
        self.ema = new if self.ema is None else self.ema + b.ema * (new - self.ema)
        if step >= self.start:
            lo, hi = map(math.log, b.clip)
            self.log_m -= b.alpha * (self.ema - 1 / L)
            self.log_m = (self.log_m - self.log_m.mean()).clamp(lo, hi)
            if b.variant == 'B':
                self.log_g = min(hi, max(lo, self.log_g - b.gain * math.log(theta[0] / (b.target * self.thr))))
        self.apply()
        named = lambda t: dict(zip(self.blocks, t.tolist()))  # noqa: E731
        return dict(lam=theta[0], ritz=theta, thr=self.thr, shares=named(new), ema=named(self.ema),
                    m=named(self.log_m.exp()), g=math.exp(self.log_g), rate=named(self.rates()), **rec)

    def state_dict(self):
        return dict(log_m=self.log_m, log_g=self.log_g, ema=self.ema, X=self.X)

    def load(self, state):
        self.log_m, self.log_g, self.ema, self.X = state['log_m'], state['log_g'], state['ema'], state['X']
        self.apply()
