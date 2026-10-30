"""Break-even point of each optimizer (report, "A model of the results").
(1) AdamW at its edge of stability (`eos.at_eos`, after warmup): its threshold applies to P^-1 H, so the raw
sharpness times the learning rate should scale with the size of the gradient. Fit of log(lambda_max * lr) on
log(gradient norm, mean over the training records within `window` steps), pooled over the runs.
(2) Across learning rates: the median lambda_max over the second half of training against the peak learning rate,
with the slope of the log-log fit per optimizer (-1 for a cap proportional to 1/lr, 0 for none).
python -m optsplit.analysis.stability --eos 'ts10m_main/adamw_*' --sweep 'ts10m_lrsweep_hessian/*' --out analysis/ts10m"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import eos
from optsplit.analysis.io import load_runs


def eos_points(run, window):
    steps = eos.at_eos(run)
    s, lam = run.series('measure', 'lambda_max')
    _, lr = run.series('measure', 'lr')
    ts, g = run.series('train', 'grad_norm')
    lam, lr = dict(zip(s, lam)), dict(zip(s, lr))
    out = []
    for t in steps[steps > run.cfg.schedule.warmup_steps]:
        near = g[np.abs(ts - t) <= window]
        if len(near) and t in lam:
            out.append((float(np.mean(near)), lam[t] * lr[t]))
    return out


def fit(x, y):
    x, y = np.log(x), np.log(y)
    slope, icept = np.polyfit(x, y, 1)
    return float(slope), float(np.corrcoef(x, y)[0, 1])


def sweep_points(run):
    s, lam = run.series('measure', 'lambda_max')
    keep = s >= run.cfg.train.max_steps / 2
    return float(getattr(run.cfg.optim, run.optimizer).lr), float(np.median(lam[keep])) if keep.any() else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eos', nargs='*', default=[], help='AdamW runs')
    p.add_argument('--sweep', nargs='*', default=[], help='runs of the learning-rate sweep with measurements')
    p.add_argument('--window', type=int, default=50)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    res, out = {}, Path(args.out)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    if args.eos:
        pts = [q for r in load_runs(args.eos) for q in eos_points(r, args.window)]
        g, y = np.array(pts).T
        res['eos'] = dict(n=len(pts), slope=fit(g, y)[0], r=fit(g, y)[1], points=pts)
        print(f"AdamW at EoS: {len(pts)} points, log(lambda*lr) on log(grad norm): slope {res['eos']['slope']:.2f}, "
              f"r {res['eos']['r']:.2f}")
        axes[0].scatter(g, y, s=6)
        axes[0].set(xscale='log', yscale='log', xlabel='gradient norm', ylabel=r'$\lambda_{max}\,\eta$',
                    title=f"AdamW at its threshold (slope {res['eos']['slope']:.2f})")
    if args.sweep:
        runs = [r for r in load_runs(args.sweep) if r.series('measure', 'lambda_max')[0].size]
        res['sweep'] = {}
        for c, o in enumerate(('sgd', 'adamw', 'muon')):
            pts = sorted(sweep_points(r) for r in runs if r.optimizer == o)
            pts = [q for q in pts if np.isfinite(q[1])]
            if len(pts) < 2:
                continue
            lr, lam = np.array(pts).T
            res['sweep'][o] = dict(points=pts, slope=fit(lr, lam)[0])
            print(f'{o}: ' + ' '.join(f'lr {a:g} lambda {b:.1f}' for a, b in pts) + f" | slope {res['sweep'][o]['slope']:.2f}")
            axes[1].plot(lr, lam, 'o-', color=f'C{c}', label=f"{o} (slope {res['sweep'][o]['slope']:.2f})")
        axes[1].set(xscale='log', yscale='log', xlabel='peak learning rate',
                    ylabel=r'median $\lambda_{max}$, 2nd half', title='Sharpness across learning rates')
        axes[1].legend(fontsize=7)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'stability.json').write_text(json.dumps(res, indent=1))
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'stability.png', dpi=150)
    print(f"wrote {out / 'figures' / 'stability.png'}")


if __name__ == '__main__':
    main()
