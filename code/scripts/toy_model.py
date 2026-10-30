"""Toy model of the main results: a two-layer linear bigram model, next-character logits (A B)[:, a] + b for the
previous character a, on character-level Shakespeare, trained from a shared small initialization (b = 0) with SGD
with momentum, AdamW, and Muon (on A and B, with AdamW on b, as Muon leaves embeddings, head and gains to an
AdamW), five data orders each. Everything is exact: the model's function is
the V x V table of next-character distributions, its training loss the cross-entropy against the bigram counts of
the training split, whose minimum over all bigram predictors is the bigram entropy H_2. Reproduces the
loss-matched distance against the data-order floor, the routes past the unigram and bigram models, the swaps, and
the sharpness against each optimizer's stability threshold (theory section of the report).
PYTHONPATH=code python code/scripts/toy_model.py --text input.txt --out DIR"""

import argparse
import json
import math
from itertools import combinations, permutations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from optsplit.analysis.split import episodes, persistent

torch.set_default_dtype(torch.float64)
OPTS = ('sgd', 'adamw', 'muon')
MU, BETAS, MU_MUON, EPS = 0.9, (0.9, 0.99), 0.95, 1e-8
THRESHOLD = dict(sgd=2 + 2 * MU, adamw=(2 + 2 * BETAS[0]) / (1 - BETAS[0]))  # times 1/lr; Muon has none


def load(path):
    text = Path(path).read_text()
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    ids = torch.tensor([stoi[c] for c in text])
    V, n = len(stoi), int(0.9 * len(ids))

    def counts(s):
        N = torch.zeros(V, V)
        N.index_put_((s[1:], s[:-1]), torch.ones(len(s) - 1), accumulate=True)
        return N  # N[y, a]: previous a, next y
    Ntr, Nva = counts(ids[:n]), counts(ids[n:])
    P = Ntr / Ntr.sum()
    q2 = P / P.sum(0, keepdim=True).clamp_min(1e-300)
    q1 = P.sum(1)
    xlogy = lambda a, b: torch.where(a > 0, a * b.clamp_min(1e-300).log(), torch.zeros(()))
    return dict(V=V, x=ids[:n - 1], y=ids[1:n], P=P, Pva=Nva / Nva.sum(), wva=Nva.sum(0) / Nva.sum(),
                logq1=q1.clamp_min(1e-300).log()[:, None].expand(V, V), logq2=q2.clamp_min(1e-300).log(),
                H1=-xlogy(q1, q1).sum().item(), H2=-xlogy(P, q2).sum().item())


def logits(params):
    return params[0] @ params[1] + params[2][:, None]


def loss_of(W, P):
    return -(P * torch.log_softmax(W, 0)).sum()


def js(lp, lq, w):
    """Mean over contexts (weights w) of the JS divergence in bits between the columns of two log-prob tables."""
    p, q = lp.exp(), lq.exp()
    m = (p + q) / 2
    kl = lambda a, la: torch.where(a > 0, a * (la - m.clamp_min(1e-300).log()), torch.zeros(())).sum(0)
    return ((0.5 * (kl(p, lp) + kl(q, lq))) * w).sum().item() / math.log(2)


def msign(u):
    U, _, Vh = torch.linalg.svd(u, full_matrices=False)
    return U @ Vh * max(1, u.shape[0] / u.shape[1]) ** 0.5


def lr_at(t, peak, T, warmup):
    if t < warmup:
        return peak * (t + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - warmup) / max(1, T - warmup))))


class State:
    """Optimizer state of all three optimizers, updated in shadow from the same gradients (warm swaps)."""
    def __init__(self, params):
        z = lambda: [torch.zeros_like(p) for p in params]
        self.t, self.sgd, self.m, self.v, self.muon = 0, z(), z(), z(), z()

    def clone(self):
        s = State([])
        s.t = self.t
        for k in ('sgd', 'm', 'v', 'muon'):
            setattr(s, k, [x.clone() for x in getattr(self, k)])
        return s

    def precond(self, i):
        return (self.v[i] / (1 - BETAS[1] ** self.t)).sqrt() + EPS

    def step(self, params, grads, opt, lr, aux_lr):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.sgd[i].mul_(MU).add_(g)
            self.m[i].mul_(BETAS[0]).add_(g, alpha=1 - BETAS[0])
            self.v[i].mul_(BETAS[1]).addcmul_(g, g, value=1 - BETAS[1])
            self.muon[i].mul_(MU_MUON).add_(g)
            if opt == 'sgd':
                p.sub_(lr * self.sgd[i])
            elif opt == 'adamw' or p.ndim == 1:
                p.sub_((lr if opt == 'adamw' else aux_lr) * self.m[i] / (1 - BETAS[0] ** self.t) / self.precond(i))
            else:
                p.sub_(lr * msign(g + MU_MUON * self.muon[i]))


def top_eig(params, P, scale=None):
    """Largest eigenvalue of the Hessian of the exact training loss (or of S H S with S = diag(scale)), by
    Lanczos."""
    theta = torch.cat([p.reshape(-1) for p in params]).detach().requires_grad_()
    parts = [x.view(p.shape) for x, p in zip(theta.split([p.numel() for p in params]), params)]
    g = torch.autograd.grad(loss_of(logits(parts), P), theta, create_graph=True)[0]
    s = torch.ones(len(theta)) if scale is None else scale

    def mv(v):
        v = torch.from_numpy(np.asarray(v, dtype=np.float64).reshape(-1)) * s
        return (torch.autograd.grad(g, theta, v, retain_graph=True)[0] * s).numpy()
    op = LinearOperator((len(theta),) * 2, matvec=mv, dtype=np.float64)
    return float(eigsh(op, k=1, which='LA', tol=1e-4, return_eigenvectors=False)[0])


def run(D, opt, seed, peak, T, warmup, batch, d, aux=None, init=None, start=0, log=None, keep=(), measure=True):
    """Train from `init` = (params, State) at step `start` (default: the shared initialization) to step T; `aux`
    is the peak rate of Muon's AdamW."""
    if init is None:
        g = torch.Generator().manual_seed(0)
        params = [0.02 * torch.randn(D['V'], d, generator=g), 0.02 * torch.randn(d, D['V'], generator=g),
                  torch.zeros(D['V'])]
        state = State(params)
    else:
        params, state = [p.clone() for p in init[0]], init[1].clone()
    log = sorted(t for t in (log or {T}) if t >= start) or [T]
    out = dict(steps=[], loss=[], val=[], lp=[], lam=[], lam_pre=[], lr=[], kept={})
    for t in range(start, T + 1):
        if t in keep:
            out['kept'][t] = ([p.clone() for p in params], state.clone())
        if t in log:
            W = logits(params)
            out['steps'].append(t)
            out['loss'].append(loss_of(W, D['P']).item())
            out['val'].append(loss_of(W, D['Pva']).item())
            out['lp'].append(torch.log_softmax(W, 0).float())
            out['lr'].append(lr_at(t, peak, T, warmup))
            if measure and math.isfinite(out['loss'][-1]):
                out['lam'].append(top_eig(params, D['P']))
                if opt == 'adamw' and state.t:
                    s = torch.cat([state.precond(i).reshape(-1) for i in range(len(params))]).rsqrt()
                    out['lam_pre'].append(top_eig(params, D['P'], s))
        if t == T or not math.isfinite(out['loss'][-1] if out['loss'] else 0.):
            break
        idx = torch.randint(len(D['x']), (batch,), generator=torch.Generator().manual_seed(seed * 1_000_003 + t))
        for p in params:
            p.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(logits(params)[:, D['x'][idx]].T, D['y'][idx])
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p in params:
                p.requires_grad_(False)
            state.step(params, grads, opt, lr_at(t, peak, T, warmup), lr_at(t, aux or peak, T, warmup))
    out['lp'] = torch.stack(out['lp'])
    out['loss_mono'] = np.minimum.accumulate(np.array(out['loss']))
    out['params'] = params
    return out


def barrier(p, q, P):
    """Loss barrier on the straight line between two parameter vectors (max minus the endpoints' mean)."""
    ls = [loss_of(logits([a * x + (1 - a) * y for x, y in zip(p, q)]), P).item() for a in np.linspace(0, 1, 11)]
    return max(ls) - (ls[0] + ls[-1]) / 2


def weights(params):
    A, B = params[:2]
    th = torch.cat([p.reshape(-1) for p in params])
    G = A.T @ A
    return dict(l2=th.norm().item(), linf_over_l2=(th.abs().max() / th.norm()).item(),
                imbalance=((G - B @ B.T).norm() / G.norm()).item())


def matched(a, b, w, tol=0.02):
    """Loss-matched distance (loss of a, JS), as in scripts/linear_control.py."""
    la, lb = a['loss_mono'], b['loss_mono']
    j = np.abs(lb[None, :] - la[:, None]).argmin(1)
    ok = np.abs(lb[j] - la) <= tol * np.abs(la)
    return la[ok], np.array([js(a['lp'][i], b['lp'][j[i]], w) for i in np.nonzero(ok)[0]])


def on_grid(loss, d, grid):
    loss, idx = np.unique(loss, return_index=True)
    return np.interp(grid, loss, d[idx], left=np.nan, right=np.nan) if len(loss) > 1 else np.full(len(grid), np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True, help="tiny Shakespeare's input.txt")
    p.add_argument('--d', type=int, default=128, help='inner width (>= V: every bigram table is reachable)')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--orders', type=int, default=5)
    p.add_argument('--lr', nargs=3, type=float, metavar=('SGD', 'ADAMW', 'MUON'), help='skip the sweep')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(args.text)
    T, w = args.steps, D['wva']
    print(f"V={D['V']} H1={D['H1']:.4f} H2={D['H2']:.4f} nats")
    kw = dict(T=T, warmup=args.warmup, batch=args.batch, d=args.d)

    if args.lr:
        lrs = dict(zip(OPTS, args.lr))
    else:  # best final validation loss on data order 0, as for the language models
        grid = dict(sgd=[0.3, 1, 3, 10, 30, 100], adamw=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1], muon=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        lrs = {}
        for o, cand in grid.items():
            vals = {lr: run(D, o, 0, lr, aux=lrs.get('adamw'), measure=False, **kw)['val'][-1] for lr in cand}
            vals = {lr: v if math.isfinite(v) else math.inf for lr, v in vals.items()}
            lrs[o] = min(vals, key=vals.get)
            print(o, {lr: round(v, 4) for lr, v in vals.items()}, '->', lrs[o])

    log = set(np.unique(np.geomspace(1, T, 70).astype(int)).tolist()) | set(range(0, T + 1, T // 50))
    fork_steps = [s for s in (10, 30, 60, 100, 200, 400, 800, 1200) if s < T]
    runs = {o: [run(D, o, s, lrs[o], aux=lrs['adamw'], log=log, keep=fork_steps if s == 0 else (), **kw) for s in range(args.orders)]
            for o in OPTS}
    res = dict(args=vars(args), lrs=lrs, H1=D['H1'], H2=D['H2'], pairs={}, route={}, swaps=[], final={},
               stability={})
    floor_of = lambda R, g: np.nanmean([on_grid(*matched(R[i], R[j], w), g) for i, j in combinations(range(len(R)), 2)], 0)
    for a, b in combinations(OPTS, 2):
        R = runs[a] + runs[b]
        lgrid = np.geomspace(min(max(r['loss_mono']) for r in R), max(min(r['loss_mono']) for r in R), 80)
        cross = np.nanmean([on_grid(*matched(x, y, w), lgrid) for x, y in zip(runs[a], runs[b])], 0)
        floor = np.fmax(floor_of(runs[a], lgrid), floor_of(runs[b], lgrid))
        eps = episodes(lgrid, cross, floor, 0.5, 0.0, 3)
        res['pairs'][f'{a}-{b}'] = dict(grid=lgrid.tolist(), cross=cross.tolist(), floor=floor.tolist(), episodes=eps,
                                        persistent=persistent(lgrid, eps))
        print(f'{a}-{b}: episodes {np.round(eps, 3).tolist()} persistent {persistent(lgrid, eps)}')

    # routes: distance to the unigram and bigram models of the training split, against the training loss
    for o, R in runs.items():
        r = R[0]
        res['route'][o] = dict(loss=r['loss'], uni=[js(lp, D['logq1'], w) for lp in r['lp']],
                               bi=[js(lp, D['logq2'], w) for lp in r['lp']])
        u, b = np.array(res['route'][o]['uni']), np.array(res['route'][o]['bi'])
        print(f"{o}: closest to unigram {u.min():.4f} bits at loss {r['loss'][u.argmin()]:.3f}; "
              f"to bigram {b.min():.4f} at loss {r['loss'][b.argmin()]:.3f}; final loss {r['loss'][-1]:.4f}")

    # end of training: same / other data order and floors (the additive decomposition)
    end = lambda r: r['lp'][-1]
    for a, b in combinations(OPTS, 2):
        n = args.orders
        same = np.mean([js(end(runs[a][i]), end(runs[b][i]), w) for i in range(n)])
        other = np.mean([js(end(runs[a][i]), end(runs[b][j]), w) for i in range(n) for j in range(n) if i != j])
        fa, fb = (np.mean([js(end(runs[o][i]), end(runs[o][j]), w) for i, j in combinations(range(n), 2)]) for o in (a, b))
        res['final'][f'{a}-{b}'] = dict(same=same, other=other, floor_a=fa, floor_b=fb)
        print(f'end {a}-{b}: same {same:.5f} other {other:.5f} floors {fa:.5f} {fb:.5f} (mean {(fa + fb) / 2:.5f})')

    # swaps from data order 0 at each fork step, with warm (shadow) optimizer states
    fl = {o: np.mean([js(end(R[i]), end(R[j]), w) for i, j in combinations(range(len(R)), 2)]) for o, R in runs.items()}
    for a, b in permutations(OPTS, 2):
        for s in fork_steps:
            f = run(D, b, 0, lrs[b], aux=lrs['adamw'], init=runs[a][0]['kept'][s], start=s, measure=False, **kw)
            ja, jb = js(end(f), end(runs[a][0]), w), js(end(f), end(runs[b][0]), w)
            res['swaps'].append(dict(a=a, b=b, step=s, r=ja / (ja + jb), to_a=ja / fl[a], to_b=jb / fl[b],
                                     loss=f['loss'][-1], loss_a=runs[a][0]['loss'][-1], loss_b=runs[b][0]['loss'][-1]))
        print(f'{a}->{b}: to b / floor', [round(x['to_b'], 2) for x in res['swaps'] if (x['a'], x['b']) == (a, b)])

    # weight space: barriers between data orders and between optimizers (same data order), norms, balance
    n = args.orders
    res['weights'] = {o: {k: float(np.mean([weights(r['params'])[k] for r in runs[o]])) for k in ('l2', 'linf_over_l2', 'imbalance')}
                      for o in OPTS}
    for o in OPTS:
        res['weights'][o]['barrier_data'] = float(np.mean([barrier(runs[o][i]['params'], runs[o][j]['params'], D['P'])
                                                           for i, j in combinations(range(n), 2)]))
    for a, b in combinations(OPTS, 2):
        res['weights'][f'{a}-{b}'] = dict(barrier=float(np.mean([barrier(runs[a][i]['params'], runs[b][i]['params'], D['P'])
                                                                  for i in range(n)])))
    for k, v in res['weights'].items():
        print('weights', k, {x: round(y, 5) for x, y in v.items()})

    for o in OPTS:
        r = runs[o][0]
        lam = np.array(r['lam'])
        res['stability'][o] = dict(steps=r['steps'][:len(lam)], lam=lam.tolist(), lr=r['lr'][:len(lam)],
                                   lam_pre=r['lam_pre'])
    (out / 'toy_model.json').write_text(json.dumps(res, indent=1))
    plot(res, runs, out / 'toy_model.png')


def plot(res, runs, path):
    H1, H2 = res['H1'], res['H2']
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7))
    ax = ax.ravel()
    for c, o in enumerate(OPTS):
        r = runs[o][0]
        ax[0].plot(r['steps'], r['loss'], color=f'C{c}', label=o)
    for h, n in ((H1, 'unigram entropy $H_1$'), (H2, 'bigram entropy $H_2$')):
        ax[0].axhline(h, color='k', lw=0.6, ls=':')
        ax[0].text(1, h, n, fontsize=7, va='bottom')
    ax[0].set(xscale='symlog', xlabel='step', ylabel='training loss (nats)', title='Training loss (data order 0)')
    ax[0].legend(fontsize=7)
    for c, (k, v) in enumerate(res['pairs'].items()):
        ax[1].plot(v['grid'], np.array(v['cross']) / np.array(v['floor']), color=f'C{c}', label=k.replace('-', ' vs '))
    ax[1].axhline(1.5, color='r', ls='--', lw=0.8)
    ax[1].axhline(1, color='k', lw=0.5)
    ax[1].axvline(H2, color='k', lw=0.6, ls=':')
    ax[1].set(yscale='log', xlabel='training loss', ylabel='distance / data-order floor',
              title='Loss-matched distance over the floor')
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=7)
    for c, o in enumerate(OPTS):
        rt = res['route'][o]
        ax[2].plot(rt['loss'], rt['uni'], color=f'C{c}', ls='--', label=f'{o}: to unigram')
        ax[2].plot(rt['loss'], rt['bi'], color=f'C{c}', label=f'{o}: to bigram')
    ax[2].set(yscale='log', xlabel='training loss', ylabel='JS (bits)', title='Routes past the n-gram models')
    ax[2].invert_xaxis()
    ax[2].legend(fontsize=6, ncol=2)
    # funnel: loss-matched distance between optimizers against the excess over H_2, with the bounds
    for c, (k, v) in enumerate(res['pairs'].items()):
        ax[3].plot(np.array(v['grid']) - H2, v['cross'], color=f'C{c}', label=k.replace('-', ' vs '))
    dl = np.geomspace(1e-3, 2, 50)
    ax[3].plot(dl, 2 * math.sqrt(2) * np.sqrt(dl), 'k--', lw=0.8, label=r'$2\sqrt{2}\sqrt{\Delta L}$ (bound)')
    ax[3].plot(dl, dl / math.log(2), 'k:', lw=0.8, label=r'$\Delta L/\ln 2$ (near $q_2$)')
    ax[3].set(xscale='log', yscale='log', xlabel=r'excess loss $L-H_2$ (nats)', ylabel='JS between optimizers (bits)',
              title='Funnel: equal-loss distance vs excess loss')
    ax[3].legend(fontsize=6)
    for c, o in enumerate(OPTS):
        st = res['stability'][o]
        lam, lr = np.array(st['lam']), np.array(st['lr'])
        if o in THRESHOLD:
            ax[4].plot(st['steps'], lam * lr / THRESHOLD[o], color=f'C{c}', ls='--', label=f'{o}: $\\lambda_{{max}}(H)$')
        if st['lam_pre']:
            pre = np.array(st['lam_pre'])
            ax[4].plot(st['steps'][-len(pre):], pre * lr[-len(pre):] / THRESHOLD[o], color=f'C{c}',
                       label=f'{o}: $\\lambda_{{max}}(P^{{-1}}H)$')
        ax[5].plot(st['steps'], lam, color=f'C{c}', label=o)
    ax[4].axhline(1, color='k', lw=0.5)
    ax[4].set(xscale='symlog', yscale='log', xlabel='step', ylabel='sharpness / threshold',
              title='Edge of stability (1 = threshold)')
    ax[4].legend(fontsize=7)
    ax[5].set(xscale='symlog', yscale='log', xlabel='step', ylabel=r'$\lambda_{max}(H)$', title='Raw sharpness')
    ax[5].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')

    sw = res['swaps']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for c, (a, b) in enumerate(permutations(OPTS, 2)):
        s = [x for x in sw if (x['a'], x['b']) == (a, b)]
        ax[0].plot([x['step'] for x in s], [x['to_b'] for x in s], '-o', ms=3, color=f'C{c}', label=f'{a}->{b}')
        ax[0].plot([x['step'] for x in s], [x['to_a'] for x in s], '--', color=f'C{c}')
        ax[1].scatter([abs(x['loss'] - x['loss_b']) for x in s], [x['to_b'] for x in s], color=f'C{c}', s=12)
    ax[0].axhline(1, color='k', lw=0.5)
    ax[0].set(xscale='log', yscale='log', xlabel='fork step', ylabel='JS to pure run / its floor',
              title='Swaps: solid to the new optimizer, dashed to the old')
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].axhline(1, color='k', lw=0.5)
    ax[1].set(xscale='log', yscale='log', xlabel='|final loss of fork - final loss of pure B|',
              ylabel='JS to pure B / floor', title='Same final loss, same function')
    fig.tight_layout()
    fig.savefig(path.with_name('toy_swaps.png'), dpi=150, bbox_inches='tight')
    print(f"wrote {path.with_name('toy_swaps.png')}")


if __name__ == '__main__':
    main()
