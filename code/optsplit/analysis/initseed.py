"""How the distance from changing the initialization compares with changing the data order or the optimizer.
At equal steps, for each optimizer X over the data orders both suites share: X at init 0 vs X at init 1
(same data), X_d vs X_d+1 at either init (same init, other data), and X vs every other optimizer Y at
either init (same init and data).
python -m optsplit.analysis.initseed 'ts10m_main/*' 'ts10m_initseed/*' --out analysis/ts10m"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.distance import step_curve
from optsplit.analysis.io import load_runs


def compare(runs_0, runs_1, metric='js'):
    idx = lambda runs: {(r.optimizer, r.data_seed): r for r in runs}
    R = [idx(runs_0), idx(runs_1)]
    seeds = sorted({d for _, d in R[1]} & {d for _, d in R[0]})
    opts = sorted({o for o, _ in R[1]})
    res = {}
    for x in opts:
        kinds = {'other init': [(R[0][x, d], R[1][x, d]) for d in seeds],
                 'other data': [(R[i][x, seeds[0]], R[i][x, seeds[1]]) for i in (0, 1)]}
        for y in opts:
            if y != x:
                kinds[f'vs {y}'] = [(R[i][x, d], R[i][y, d]) for i in (0, 1) for d in seeds]
        res[x] = {}
        for kind, pairs in kinds.items():
            curves = [step_curve(a, b) for a, b in pairs]
            steps = sorted(set.intersection(*(set(c['step_a'].tolist()) for c in curves)))
            vals = [[c[metric][list(c['step_a']).index(s)] for s in steps] for c in curves]
            res[x][kind] = dict(steps=steps, values=vals)
    return res


def plot(res, path, metric='js'):
    fig, axes = plt.subplots(1, len(res), figsize=(4.5 * len(res), 3.4), squeeze=False)
    colors = {'other init': 'C2', 'other data': '0.5'}
    for ax, (x, kinds) in zip(axes[0], res.items()):
        for c, (kind, k) in enumerate(kinds.items()):
            v = np.array(k['values'])
            color = colors.get(kind, f'C{c + 2}')
            ax.plot(k['steps'], np.median(v, 0), color=color, ls='-' if kind in colors else '--', label=kind)
            ax.fill_between(k['steps'], v.min(0), v.max(0), color=color, alpha=0.15, lw=0)
        ax.set(xscale='symlog', yscale='log', xlabel='step', ylabel=metric, title=x)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs_0', help='glob of the runs at the default init')
    p.add_argument('runs_1', help='glob of the runs at the other init')
    p.add_argument('--metric', default='js', choices=['js', 'disagree'])
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    res = compare(load_runs([args.runs_0]), load_runs([args.runs_1]), args.metric)
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / f'initseed_{args.metric}.json').write_text(json.dumps(res))
    for x, kinds in res.items():
        print(x, ', '.join(f'{kind}: {np.median(np.array(k["values"])[:, -1]):.4f}' for kind, k in kinds.items()),
              f'(step {kinds["other init"]["steps"][-1]})')
    plot(res, out / 'figures' / f'initseed_{args.metric}.png', args.metric)


if __name__ == '__main__':
    main()
