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


def at_eos(run, delta=0.1):
    """Measured steps at which the ratio is >= 1 - delta (None without a threshold)."""
    r = ratio(run)
    return None if r is None else r[0][r[1] >= 1 - delta]


def onset(run, delta=0.1):
    s = at_eos(run, delta)
    return int(s[0]) if s is not None and len(s) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--delta', type=float, default=0.1)
    p.add_argument('--split', nargs='*', default=[], help='split JSONs whose t* steps to print alongside')
    args = p.parse_args()
    splits = [json.load(open(f)) for f in args.split]
    for run in load_runs(args.runs):
        r, s = ratio(run), at_eos(run, args.delta)
        if r is None or not len(r[0]):
            eos_txt = 'no threshold'
        else:
            span = f'{s[0]}-{s[-1]}' if len(s) else 'never'
            eos_txt = f'at EoS {span} ({len(s) / len(r[0]):.0%} of measurements), max ratio {np.nanmax(r[1]):.2f}'
        stars = []
        for sp in splits:
            for key, tag in (('steps', 'first'), ('steps_persist', 'persist')):
                t = sp.get(key, {}).get(run.id)
                if t is not None:
                    at = f' ratio {np.interp(t, *r):.2f}' if r is not None and len(r[0]) else ''
                    stars.append(f"{tag}({sp['optimizer_a']}|{sp['optimizer_b']})={t}{at}")
        print(f'{run.id:<32} warmup {run.cfg.schedule.warmup_steps:>4}  {eos_txt}  {" ".join(stars)}')


if __name__ == '__main__':
    main()
