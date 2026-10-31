"""Does the comparison at equal loss depend on the tuned learning rates? Each learning-rate sweep run (data order
0) against the tuned main run of every optimizer with the same data order and initialization, at equal training
loss, in units of the data-order floor of the tuned runs (the larger of the two optimizers' floors, as in
`split`). Summarized by the median ratio over the sweep run's checkpoints below `--below` (past the early,
warmup-time differences) and the ratio at its lowest matched loss.
python -m optsplit.analysis.lr_robust --sweep 'ts10m_lrsweep/*' --main 'ts10m_main/*' --below 3.8 \
    --out analysis/ts10m"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.distance import loss_curve, loss_points
from optsplit.analysis.io import load_runs
from optsplit.analysis.split import _mean, by_seed, on_grid


def compare(run, ref, floors, below, tol):
    grid = np.unique(loss_points(run)[1])[::-1]
    floor = np.fmax.reduce([on_grid(floors[o], grid, 'js') for o in {run.optimizer, ref.optimizer} if o in floors])
    ratio = on_grid(loss_curve(run, ref, tol), grid, 'js') / floor
    ok = np.isfinite(ratio)
    late = ratio[ok & (grid <= below)]
    return dict(run=run.id, optimizer=run.optimizer, lr=run.lr, ref=ref.id, ref_optimizer=ref.optimizer,
                median_late=float(np.median(late)) if len(late) else float('nan'),
                last=float(ratio[ok][-1]) if ok.any() else float('nan'),
                last_loss=float(grid[ok][-1]) if ok.any() else float('nan'))


def heatmap(rows, path, title):
    runs = sorted({(r['optimizer'], r['lr'], r['run']) for r in rows})
    refs = sorted({r['ref_optimizer'] for r in rows})
    m = np.full((len(runs), len(refs)), np.nan)
    for r in rows:
        m[[x[2] for x in runs].index(r['run']), refs.index(r['ref_optimizer'])] = r['median_late']
    fig, ax = plt.subplots(figsize=(3.2, 0.28 * len(runs) + 1.2))
    with np.errstate(divide='ignore'):
        ax.imshow(np.log2(np.where(m > 0, m, np.nan)), cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, 'same run' if v == 0 else '--' if np.isnan(v) else f'{v:.2f}', ha='center', va='center',
                fontsize=7)
    ax.set_xticks(range(len(refs)), [f'tuned {o}' for o in refs], fontsize=7)
    ax.set_yticks(range(len(runs)), [f'{o} {lr:g}' for o, lr, _ in runs], fontsize=7)
    ax.set_title(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep', nargs='+', required=True, help='glob patterns of the learning-rate sweep runs')
    p.add_argument('--main', nargs='+', required=True, help='glob patterns of the tuned runs (several data orders)')
    p.add_argument('--below', type=float, required=True, help='training loss below which to take the median')
    p.add_argument('--tol', type=float, default=0.02)
    p.add_argument('--title', default='')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()

    tuned = {}
    for r in load_runs(args.main):
        if not r.cfg.fork.parent:
            tuned.setdefault(r.optimizer, []).append(r)
    tuned = {o: by_seed(rs) for o, rs in tuned.items()}
    floors = {}
    for o, runs in tuned.items():
        ref = runs[min(runs)]
        grid = np.unique(loss_points(ref)[1])[::-1]
        floors[o] = dict(loss=grid, js=_mean([on_grid(loss_curve(a, b), grid, 'js')
                                              for a, b in combinations(runs.values(), 2)]))
    rows = []
    for run in load_runs(args.sweep):
        for o, runs in sorted(tuned.items()):
            ref = runs.get(run.data_seed)
            if ref is not None and ref.init_seed == run.init_seed:
                rows.append(compare(run, ref, floors, args.below, args.tol))
                r = rows[-1]
                print(f"{run.id} vs {ref.id}: median {r['median_late']:.2f} below {args.below}, "
                      f"{r['last']:.2f} at loss {r['last_loss']:.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'lr_robust.json').write_text(json.dumps(rows, indent=1))
    heatmap(rows, out / 'figures' / 'lr_robust.png', args.title)


if __name__ == '__main__':
    main()
