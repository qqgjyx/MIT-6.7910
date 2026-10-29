"""Edge-of-stability onset: the first measured step at which a run's sharpness reaches (1 - delta) of its
stability threshold. SGD with momentum mu: lambda_max against (2 + 2 mu) / lr; AdamW: preconditioned
lambda_max against (2 + 2 beta1) / ((1 - beta1) lr) (Cohen et al. 2022). Muon has no threshold.
python -m optsplit.analysis.eos 'ts10m_main/*' [--delta 0.1] [--split split.json]"""

import argparse
import json

import numpy as np

from optsplit.analysis.io import load_runs


def ratio(run):
    """(steps, sharpness / threshold) at the measured steps, or None if the optimizer has no threshold."""
    steps, lr = run.series('measure', 'lr')
    if run.optimizer == 'sgd':
        key, c = 'lambda_max', 2 + 2 * run.cfg.optim.sgd.momentum
    elif run.optimizer == 'adamw':
        b1 = run.cfg.optim.adamw.betas[0]
        key, c = 'precond_lambda_max', (2 + 2 * b1) / (1 - b1)
    else:
        return None
    s, lam = run.series('measure', key)
    lr = dict(zip(steps, lr))
    keep = [i for i, t in enumerate(s) if lr.get(t, 0) > 0]
    s, lam = s[keep], lam[keep]
    return s, lam * np.array([lr[t] for t in s]) / c


def onset(run, delta=0.1):
    r = ratio(run)
    if r is None:
        return None
    hit = np.nonzero(r[1] >= 1 - delta)[0]
    return int(r[0][hit[0]]) if len(hit) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--delta', type=float, default=0.1)
    p.add_argument('--split', nargs='*', default=[], help='split JSONs whose t* steps to print alongside')
    args = p.parse_args()
    splits = [json.load(open(f)) for f in args.split]
    for run in load_runs(args.runs):
        t = onset(run, args.delta)
        r = ratio(run)
        peak = f'{np.nanmax(r[1]):.2f}' if r is not None and len(r[1]) else '-'
        stars = ' '.join(f"t*({s['optimizer_a']}|{s['optimizer_b']})={s['steps'][run.id]}"
                         for s in splits if run.id in s['steps'])
        print(f'{run.id:<36} warmup {run.cfg.schedule.warmup_steps:>5}  EoS onset {t}  max ratio {peak}  {stars}')


if __name__ == '__main__':
    main()
