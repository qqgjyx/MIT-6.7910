"""Where on the vocabulary the optimizers differ: the probe positions grouped by how frequent their target
token is in the training data (Kunstner et al. 2024 tie Adam's advantage over SGD on language models to rare
classes). Per frequency bin: each optimizer's loss, and the JS distance between optimizers (same data order)
over the larger of their data-order floors (orders d and d+1), at the final step and at the lowest training
loss both runs reach (loss-matched).
python -m optsplit.analysis.tokens 'ts30m_main/*' --out analysis/ts30m"""

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.distance import loss_points
from optsplit.analysis.io import load_runs
from optsplit.data import build_data


def js_rows(logp, logq):
    """Jensen-Shannon divergence in bits of each row."""
    logp, logq = logp.float(), logq.float()
    logm = torch.logaddexp(logp, logq) - math.log(2)
    kl = lambda lp: torch.where(lp.exp() > 0, lp.exp() * (lp - logm), 0.).sum(-1)
    return (0.5 * (kl(logp) + kl(logq)) / math.log(2)).numpy()


def token_counts(cfg, chunk=50_000_000):
    data = build_data(cfg, 'cpu')
    train, counts = data.splits['train'], torch.zeros(data.vocab_size, dtype=torch.float64)
    for i in range(0, len(train), chunk):
        counts += torch.bincount(torch.from_numpy(np.asarray(train[i:i + chunk], dtype=np.int64)),
                                 minlength=data.vocab_size)
    return counts.numpy()


def frequency_bins(runs, n_bins):
    """Bin index of every probe position, by quantiles of its target's training frequency."""
    freq = token_counts(runs[0].cfg)[runs[0].targets.reshape(-1).numpy()]
    edges = np.quantile(freq, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(edges, freq, side='right') - 1, 0, n_bins - 1), edges


def by_bin(values, bins, n_bins):
    return [float(values[bins == b].mean()) for b in range(n_bins)]


def step_at(run, level):
    """First prediction step at which the run's (running-minimum) training loss is <= level, else its last."""
    steps, loss = loss_points(run)
    hit = np.nonzero(loss <= level + 1e-9)[0]
    return int(steps[hit[0]] if len(hit) else steps[-1])


def analyze(runs, n_bins=5):
    bins, edges = frequency_bins(runs, n_bins)
    y = runs[0].targets.reshape(-1)
    by_opt = {}
    for r in sorted(runs, key=lambda r: r.data_seed):
        by_opt.setdefault(r.optimizer, []).append(r)
    res = dict(edges=edges.tolist(), loss={}, pairs={})
    for opt, rs in by_opt.items():
        nll = [-r.logp(r.final_pred_step).float()[torch.arange(len(y)), y].numpy() for r in rs]
        res['loss'][opt] = by_bin(np.mean(nll, 0), bins, n_bins)
    for x, z in itertools.combinations(sorted(by_opt), 2):
        X, Z = by_opt[x], by_opt[z]
        n = min(len(X), len(Z))
        d = lambda a, sa, b, sb: by_bin(js_rows(a.logp(sa), b.logp(sb)), bins, n_bins)
        out = {}
        for mode in ('final', 'loss-matched'):
            cross, floor_x, floor_z = [], [], []
            for i in range(n):
                xs, zs = (X[i], X[(i + 1) % n]), (Z[i], Z[(i + 1) % n])
                if mode == 'final':
                    st = lambda r: r.final_pred_step
                else:  # all four runs at the lowest training loss that both X_i and Z_i reach
                    level = max(loss_points(X[i])[1].min(), loss_points(Z[i])[1].min())
                    st = lambda r: step_at(r, level)
                cross.append(d(X[i], st(X[i]), Z[i], st(Z[i])))
                floor_x.append(d(xs[0], st(xs[0]), xs[1], st(xs[1])))
                floor_z.append(d(zs[0], st(zs[0]), zs[1], st(zs[1])))
            cross = np.median(cross, 0)
            floor = np.maximum(np.median(floor_x, 0), np.median(floor_z, 0))
            out[mode] = dict(cross=cross.tolist(), floor=floor.tolist(), ratio=(cross / floor).tolist())
        res['pairs'][f'{x}_{z}'] = out
    return res


def plot(res, path):
    n = len(res['edges']) - 1
    x = np.arange(n)
    labels = [f'{i * 100 // n}-{(i + 1) * 100 // n}%' for i in range(n)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for c, (opt, l) in enumerate(sorted(res['loss'].items())):
        axes[0].plot(x, l, '-o', color=f'C{c}', label=opt)
    axes[0].set(ylabel='final loss', title='loss by target frequency')
    for c, (pair, p) in enumerate(sorted(res['pairs'].items())):
        for ax, mode in zip(axes[1:], ('final', 'loss-matched')):
            ax.plot(x, p[mode]['ratio'], '-o', color=f'C{c + 3}', label=pair.replace('_', ' vs '))
            ax.set(ylabel='cross / floor (JS)', title=f'distance between optimizers, {mode}', yscale='log')
            ax.axhline(1, color='k', lw=0.5)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_xlabel('target-frequency quantile (rare → frequent)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--bins', type=int, default=5)
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    res = analyze(load_runs(args.runs), args.bins)
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'tokens.json').write_text(json.dumps(res))
    print('loss by bin:', {k: [round(v, 3) for v in l] for k, l in res['loss'].items()})
    for pair, p_ in res['pairs'].items():
        for mode, r in p_.items():
            print(pair, mode, 'cross/floor by bin:', [round(v, 2) for v in r['ratio']])
    plot(res, out / 'figures' / 'tokens.png')


if __name__ == '__main__':
    main()
