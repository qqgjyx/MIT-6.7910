"""Positive control for the split detection: logistic regression on linearly separable data, where the
limits of training are known to differ by optimizer (gradient descent: the l2 max-margin direction, Soudry
et al. 2018; Adam: the l_inf one, Zhang et al. 2024). Zero init and the same mini-batch order for both
optimizers, five data orders each; the loss-matched JS distance between their test predictions against the
data-order floor goes through the same episode / persistent-split rule as the language models.
python scripts/linear_control.py --out <dir>"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.split import episodes, persistent


def make_data(d, n, n_test, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_star = torch.zeros(d, dtype=torch.float64)
    w_star[:k] = 1
    X = torch.randn(n + n_test, d, generator=g, dtype=torch.float64)
    y = torch.sign(X @ w_star)
    return X[:n], y[:n], X[n:]


def train(opt_name, X, y, X_test, data_seed, steps, batch, log_steps, lr):
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = (torch.optim.SGD([w], lr=lr, momentum=0.9) if opt_name == 'sgd'
           else torch.optim.Adam([w], lr=lr))
    g = torch.Generator().manual_seed(1000 + data_seed)
    out = dict(steps=[], loss=[], p=[], w=[])
    for t in range(steps + 1):
        if t in log_steps:
            with torch.no_grad():
                out['steps'].append(t)
                out['loss'].append(torch.nn.functional.softplus(-y * (X @ w)).mean().item())
                out['p'].append(torch.sigmoid(X_test @ w).clone())
                out['w'].append(w.detach().clone())
        idx = torch.randint(len(X), (batch,), generator=g)
        loss = torch.nn.functional.softplus(-y[idx] * (X[idx] @ w)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    out['loss'] = np.minimum.accumulate(np.array(out['loss']))
    return out


def js_bernoulli(p, q, eps=1e-12):
    P, Q = torch.stack([p, 1 - p], -1).clamp_min(eps), torch.stack([q, 1 - q], -1).clamp_min(eps)
    M = (P + Q) / 2
    kl = lambda A: (A * (A / M).log2()).sum(-1)
    return (0.5 * (kl(P) + kl(Q))).mean().item()


def matched(a, b, tol=0.02):
    """Loss-matched distance (loss of a, JS) as in optsplit.analysis.distance.loss_curve."""
    j = np.abs(b['loss'][None, :] - a['loss'][:, None]).argmin(1)
    ok = np.abs(b['loss'][j] - a['loss']) <= tol * np.abs(a['loss'])
    return a['loss'][ok], np.array([js_bernoulli(a['p'][i], b['p'][j[i]]) for i in np.nonzero(ok)[0]])


def on_grid(loss, d, grid):
    loss, idx = np.unique(loss, return_index=True)
    return np.interp(grid, loss, d[idx], left=np.nan, right=np.nan) if len(loss) > 1 else np.full(len(grid), np.nan)


def margins(w, X, y):
    m = (y * (X @ w)).min().item()
    return m / w.norm(2).item(), m / w.norm(float('inf')).item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, default=50)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--k', type=int, default=3, help='non-zero coordinates of the teacher')
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--orders', type=int, default=5)
    p.add_argument('--lr-sgd', type=float, default=0.1)
    p.add_argument('--lr-adam', type=float, default=0.01)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    X, y, X_test = make_data(args.d, args.n, 2000, args.k)
    log_steps = set(np.unique(np.geomspace(1, args.steps, 80).astype(int)).tolist()) | {0}
    lrs = dict(sgd=args.lr_sgd, adam=args.lr_adam)
    runs = {o: [train(o, X, y, X_test, s, args.steps, args.batch, log_steps, lrs[o]) for s in range(args.orders)]
            for o in ('sgd', 'adam')}
    A, B = runs['sgd'], runs['adam']
    hi = min(r['loss'].max() for r in A + B)
    lo = max(r['loss'].min() for r in A + B)
    grid = np.unique(A[0]['loss'])[::-1]
    grid = grid[(grid <= hi) & (grid >= lo)]
    cross = np.nanmean([on_grid(*matched(a, b), grid) for a, b in zip(A, B)], 0)
    floor = np.fmax(*[np.nanmean([on_grid(*matched(R[i], R[j]), grid) for i, j in combinations(range(len(R)), 2)], 0)
                      for R in (A, B)])
    eps = episodes(grid, cross, floor, 0.5, 0.0, 3)
    res = dict(args=vars(args), grid=grid.tolist(), cross=cross.tolist(), floor=floor.tolist(), episodes=eps,
               persistent=persistent(grid, eps),
               margins={o: [margins(r['w'][-1], X, y) for r in R] for o, R in runs.items()},
               final_loss={o: [float(r['loss'][-1]) for r in R] for o, R in runs.items()})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'linear_control.json').write_text(json.dumps(res, indent=1))
    print('episodes', eps, 'persistent split at loss', res['persistent'])
    for o, m in res['margins'].items():
        print(o, 'final loss', np.round(res['final_loss'][o], 6), 'l2-normalized margin',
              np.round([x[0] for x in m], 4), 'l_inf-normalized margin', np.round([x[1] for x in m], 4))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(grid, cross, color='C0', label='SGD vs Adam (same data order)')
    axes[0].plot(grid, floor, color='0.5', ls='--', label='floor (other data order)')
    axes[1].plot(grid, cross / floor, color='C0')
    axes[1].axhline(1.5, color='r', ls='--', lw=0.8)
    axes[1].axhline(1, color='k', lw=0.5)
    for ax, yl in zip(axes[:2], ('JS (bits)', 'cross / floor')):
        ax.set(xscale='log', yscale='log', xlabel='training loss', ylabel=yl)
        ax.invert_xaxis()
    axes[0].legend(fontsize=7)
    for c, (o, R) in enumerate(runs.items()):
        m = np.array([[margins(w, X, y) for w in r['w'][1:]] for r in R])
        st = R[0]['steps'][1:]
        axes[2].plot(st, m[..., 0].mean(0), color=f'C{c}', label=f'{o}: margin / l2 norm')
        axes[2].plot(st, m[..., 1].mean(0), color=f'C{c}', ls='--', label=f'{o}: margin / l_inf norm')
    axes[2].set(xscale='log', xlabel='step', ylabel='normalized margin', ylim=(0, None))
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'linear_control.png', dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
