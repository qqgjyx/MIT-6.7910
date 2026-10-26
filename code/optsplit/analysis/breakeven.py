"""Figures replicating Jastrzebski et al. (2020) from SGD runs at several learning rates or batch sizes:
curvature and gradient-noise quantities over training (their Figs. 4-5), lambda_H and the one-step loss
change against lambda_K (Fig. 2), and the trajectories of the runs in prediction space (Fig. 1)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.io import load_runs
from optsplit.config import root_dir

CURVES = [('lambda_max', r'$\lambda_H^1$'), ('gradcov.lambda_K', r'$\lambda_K^1$'),
          ('ratio', r'$\lambda_K^*/\lambda_K^1$'), ('gradcov.trace_K', r'Tr$(K)$'),
          ('gradcov.g_over_g5', r'$\|g\|/\|g_5\|$')]


def label(run):
    return f'{run.optimizer} lr={run.lr:g} B={run.cfg.train.batch_size}'


def _series(run, key):
    if key != 'ratio':
        return run.series('measure', key)
    s1, kmin = run.series('measure', 'gradcov.lambda_K_min')
    s2, kmax = run.series('measure', 'gradcov.lambda_K')
    common, i, j = np.intersect1d(s1, s2, return_indices=True)
    return common, kmin[i] / kmax[j]


def curves(runs, out, acc_threshold=None):
    fig, axes = plt.subplots(1, len(CURVES), figsize=(4 * len(CURVES), 3.2))
    for c, run in enumerate(runs):
        for ax, (key, name) in zip(axes, CURVES):
            s, v = _series(run, key)
            ax.plot(s, v, color=f'C{c}', label=label(run))
            ax.set(title=name, xlabel='step', yscale='log' if key != 'ratio' else 'linear')
            if acc_threshold is not None:
                sa, acc = run.series('eval', 'val_acc')
                hit = np.nonzero(acc >= acc_threshold)[0]
                if len(hit):
                    ax.axvline(sa[hit[0]], color=f'C{c}', ls='--', lw=0.8)
    axes[0].legend(fontsize=7)
    _save(fig, out)


def scatter(runs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for c, run in enumerate(runs):
        sk, lk = run.series('measure', 'gradcov.lambda_K')
        sh, lh = run.series('measure', 'lambda_max')
        sd, dl = run.series('measure', 'delta_loss')
        _, i, j = np.intersect1d(sk, sh, return_indices=True)
        a1.scatter(lk[i], lh[j], s=4, color=f'C{c}', label=label(run))
        _, i, j = np.intersect1d(sk, sd, return_indices=True)
        a2.scatter(lk[i], dl[j], s=4, color=f'C{c}')
    a1.set(xscale='log', yscale='log', xlabel=r'$\lambda_K^1$', ylabel=r'$\lambda_H^1$')
    a2.set(xscale='log', xlabel=r'$\lambda_K^1$', ylabel=r'$\Delta L$')
    a2.axhline(0, color='k', lw=0.5)
    a1.legend(fontsize=7)
    _save(fig, out)


def trajectory_embedding(runs, max_rows=128, n_pcs=50, seed=0, use_umap=True):
    """2-D embedding of every run's prediction steps (probabilities on `max_rows` probe rows, PCA then
    UMAP if available). Returns [(run, steps, (n_steps, 2) array)]."""
    blocks, steps = [], []
    for run in runs:
        st = run.pred_steps
        n = run.logp(st[0]).shape[0]
        rows = torch.linspace(0, n - 1, min(max_rows, n)).long()
        blocks.append(torch.stack([run.logp(s)[rows].float().exp().flatten() for s in st]))
        steps.append(st)
    X = torch.cat(blocks)
    X -= X.mean(0)
    evals, evecs = torch.linalg.eigh((X @ X.T).double())
    k = min(n_pcs, len(X) - 1)
    Z = (evecs[:, -k:] * evals[-k:].clamp_min(0).sqrt()).flip(1).numpy()
    if use_umap:
        try:
            import umap
            Z = umap.UMAP(random_state=seed).fit_transform(Z)
        except ImportError:
            pass
    Z = Z[:, :2]
    out, i = [], 0
    for run, st in zip(runs, steps):
        out.append((run, st, Z[i:i + len(st)]))
        i += len(st)
    return out


def trajectories(runs, out, use_umap=True):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for c, (run, st, z) in enumerate(trajectory_embedding(runs, use_umap=use_umap)):
        ax.plot(z[:, 0], z[:, 1], '-o', ms=2, lw=1, color=f'C{c}', label=label(run))
        ax.plot(*z[0], 'k*', ms=8)
    ax.set(xticks=[], yticks=[], title='prediction trajectories')
    ax.legend(fontsize=7)
    _save(fig, out)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='glob patterns of run ids')
    p.add_argument('--out', help='output directory')
    p.add_argument('--acc-threshold', type=float, help='mark the first step with val_acc above this')
    p.add_argument('--no-umap', action='store_true', help='PCA instead of UMAP for the trajectories')
    args = p.parse_args()
    runs = load_runs(args.runs)
    out = Path(args.out or root_dir() / 'analysis' / 'breakeven' / runs[0].id.split('/')[0])
    curves(runs, out / 'curves.png', args.acc_threshold)
    scatter(runs, out / 'scatter.png')
    trajectories(runs, out / 'trajectories.png', not args.no_umap)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
