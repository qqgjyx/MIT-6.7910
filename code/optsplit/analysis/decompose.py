"""How much of the distance between two runs comes from the optimizer and how much from the data order. For
each pair of optimizers (X, Y) with data orders 0..n-1, at equal steps: X_d vs Y_d (other optimizer, same
data: the cross distance of split.py), X_d vs Y_d+1 (other optimizer, other data), and the floors X_d vs
X_d+1 and Y_d vs Y_d+1 (same optimizer, other data), with d+1 taken mod n.
python -m optsplit.analysis.decompose 'ts30m_main/*' --out analysis/ts30m"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.distance import step_curve
from optsplit.analysis.io import load_runs


def decompose(runs, metric='js'):
    by_opt = {}
    for r in sorted(runs, key=lambda r: r.data_seed):
        by_opt.setdefault(r.optimizer, []).append(r)
    res = {}
    for x, y in itertools.combinations(sorted(by_opt), 2):
        X, Y = by_opt[x], by_opt[y]
        n = min(len(X), len(Y))
        kinds = {f'{x}, other data': [(X[d], X[(d + 1) % n]) for d in range(n)],
                 f'{y}, other data': [(Y[d], Y[(d + 1) % n]) for d in range(n)],
                 f'{x} vs {y}, same data': [(X[d], Y[d]) for d in range(n)],
                 f'{x} vs {y}, other data': [(X[d], Y[(d + 1) % n]) for d in range(n)]}
        res[f'{x}_{y}'] = {}
        for kind, pairs in kinds.items():
            curves = [step_curve(a, b) for a, b in pairs]
            steps = sorted(set.intersection(*(set(c['step_a'].tolist()) for c in curves)))
            vals = [[c[metric][list(c['step_a']).index(s)] for s in steps] for c in curves]
            res[f'{x}_{y}'][kind] = dict(steps=steps, values=vals)
    return res


def plot(res, path, metric='js'):
    fig, axes = plt.subplots(1, len(res), figsize=(4.5 * len(res), 3.4), squeeze=False)
    styles = [dict(color='0.35'), dict(color='0.7'), dict(color='C0'), dict(color='C3', ls='--')]
    for ax, (pair, kinds) in zip(axes[0], res.items()):
        for st, (kind, k) in zip(styles, kinds.items()):
            v = np.array(k['values'])
            ax.plot(k['steps'], np.median(v, 0), label=kind, **st)
            ax.fill_between(k['steps'], v.min(0), v.max(0), color=st['color'], alpha=0.15, lw=0)
        ax.set(xscale='symlog', xlabel='step', ylabel=metric)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--metric', default='js', choices=['js', 'disagree'])
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    res = decompose(load_runs(args.runs), args.metric)
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / f'decompose_{args.metric}.json').write_text(json.dumps(res))
    for pair, kinds in res.items():
        print(pair, ', '.join(f'{kind}: {np.median(np.array(k["values"])[:, -1]):.4f}' for kind, k in kinds.items()),
              f'(step {kinds[next(iter(kinds))]["steps"][-1]})')
    plot(res, out / 'figures' / f'decompose_{args.metric}.png', args.metric)


if __name__ == '__main__':
    main()
