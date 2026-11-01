"""Numbers for the report fixes (checkpoint-wording). python ckw_numbers.py SECTION..."""

import json
import sys
from itertools import combinations

import numpy as np

from optsplit.analysis.distance import loss_curve, loss_points
from optsplit.analysis.io import load_runs
from optsplit.analysis.split import _mean, by_seed, on_grid
from optsplit.config import root_dir

R = root_dir()


def tuned_floors():
    tuned = {'adamw': by_seed(load_runs('ts10m_main/adamw_d*')), 'muon': by_seed(load_runs('ts10m_main/muon_d*')),
             'sgd': by_seed(load_runs('ts10m_main_sgd0.06/sgd_d*'))}
    floors = {}
    for o, runs in tuned.items():
        ref = runs[min(runs)]
        grid = np.unique(loss_points(ref)[1])[::-1]
        floors[o] = dict(loss=grid, js=_mean([on_grid(loss_curve(a, b), grid, 'js')
                                              for a, b in combinations(runs.values(), 2)]))
    return tuned, floors


def around():
    """Each run at another rate (d0) against the tuned run of its optimizer, over the tuned runs' floor."""
    tuned, floors = tuned_floors()
    pats = ['ts10m_lr/*', 'ts10m_main/sgd_d0', 'ts10m_sgdtune/sgd_lr0.045_d0', 'ts10m_sgdtune/sgd_lr0.085_d0',
            'ts10m_lrsweep/sgd_lr0.1']
    for run in load_runs(pats):
        for o in sorted(tuned):
            ref = tuned[o][0]
            grid = np.unique(loss_points(run)[1])[::-1]
            fl = np.fmax.reduce([on_grid(floors[x], grid, 'js') for x in {run.optimizer, o}])
            ratio = on_grid(loss_curve(run, ref), grid, 'js') / fl
            ok = np.isfinite(ratio)
            g, x = grid[ok], ratio[ok]
            out = [f'{run.id:30s} lr {run.lr:<7g} vs {ref.id:26s}']
            for b in (4.0, 3.8, 3.0):
                y = x[g <= b]
                if len(y):
                    out.append(f'<={b}: {y.min():.2f}-{y.max():.2f} med {np.median(y):.2f}')
            bad = g[(x < 0.8) | (x > 1.1)]
            out.append(f'within 0.8-1.1 below {bad.min() if len(bad) else np.inf:.2f}; last {x[-1]:.2f} @ {g[-1]:.2f}')
            print(' | '.join(out))


def stoch():
    """1 - R_pop(exact) = fraction of the first-order decrease of the population loss that a step realizes."""
    from optsplit.analysis.stochastic_eos import load, plateau
    for run in load_runs('ts10m_stoch_eos/*'):
        if not run.records_of('step_sharpness'):
            continue
        T = run.cfg.train.max_steps
        steps, _, q = load(run, 'pop', 'pop')
        if not len(steps):
            continue
        num, den = q['exact']
        num, den = np.asarray(num), np.asarray(den)
        line = [f'{run.id.split("/")[-1]:22s}']
        for f in (0.5, 0.75):
            m = steps >= f * T
            line.append(f'2nd-{f:g}: realized {1 - num[m].sum() / den[m].sum():+.3f}')
        p = plateau(steps, num, den, 0.5 * T)
        line.append(f'plateau R {p[0]:.3f} -> {1 - p[0]:+.3f}')
        ns, ds = q['s']
        ns, ds = np.asarray(ns), np.asarray(ds)
        m = steps >= 0.5 * T
        line.append(f'R_s sums {ns[m].sum() / ds[m].sum():.3f}')
        print(' | '.join(line))


def lam30():
    from optsplit.analysis.eos import ratio
    for run in load_runs('ts30m_main/adamw_d*'):
        s, lam = run.series('measure', 'lambda_max')
        sp, plam = run.series('measure', 'precond_lambda_max')
        sl, lr = run.series('measure', 'lr')
        rs, rr = ratio(run)
        L, P, LR = dict(zip(s, lam)), dict(zip(sp, plam)), dict(zip(sl, lr))
        eos = [t for t, x in zip(rs, rr) if x >= 0.9 and t >= 400]
        after = [t for t in s if t >= 400]
        lv = np.array([L[t] for t in eos if t in L])
        p = np.array([L[t] / P[t] for t in eos if t in L and t in P])
        print(f'{run.id}: {len(eos)} EoS steps after warmup; raw lmax median {np.median(lv):.2f} '
              f'(range {lv.min():.2f}-{lv.max():.2f}); p = lmax/precond median {np.median(p):.2e}; '
              f'lmax*lr/38 median {np.median([L[t] * LR[t] / 38 for t in eos]):.2e}; '
              f'raw lmax after warmup {np.min([L[t] for t in after]):.2f}-{np.max([L[t] for t in after]):.2f}, '
              f'peak lr {run.lr}')


def bal():
    B = R / 'analysis' / 'balanced'
    for name in ('ts10m_sgd', 'ts10m_adamw', 'ts10m_muon', 'ts10m_v2_adamw', 'ts10m_v2_muon', 'ts30m_sgd',
                 'ts30m_adamw', 'ts30m_muon'):
        d = json.loads((B / f'{name}.json').read_text())
        floor = [c for c in d['floor'].values() if len(c['loss']) > 1]
        print(f'== {name}: base {sorted(d["base"])} floors {len(floor)}')
        for b in sorted(d['base'].values(), key=lambda b: b['seed']):
            print(f'   base d{b["seed"]}: final val {b["eval"][-1][2]:.4f} at {b["eval"][-1][0]}')
        for rid, c in sorted(d['cross'].items()):
            if len(c['loss']) < 2:
                continue
            L, J = np.array(c['loss']), np.array(c['js'])
            fl = np.array([np.interp(L, *np.array(sorted(zip(f['loss'], f['js']))).T, left=np.nan, right=np.nan)
                           for f in floor])
            mean, lo, hi = np.nanmean(fl, 0), np.nanmin(fl, 0), np.nanmax(fl, 0)
            ok = np.isfinite(mean)
            r = J[ok] / mean[ok]
            Lk = L[ok]
            i = int(np.argmin(Lk))
            late = r[Lk <= 4]
            r25 = r[Lk <= 2.5]
            print(f'  {rid:40s} loss {Lk.min():.3f}..{Lk.max():.2f}: end {r[i]:.2f} x mean floor '
                  f'(band {J[ok][i] / hi[ok][i]:.2f}-{J[ok][i] / lo[ok][i]:.2f}); <=4: max {late.max() if len(late) else np.nan:.2f} '
                  f'med {np.median(late) if len(late) else np.nan:.2f}; <=2.5 max {r25.max() if len(r25) else np.nan:.2f}; '
                  f'above band at {np.mean(J[ok][Lk <= 4] > hi[ok][Lk <= 4]) if len(late) else np.nan:.2f} of levels <=4')
        for rid, r in sorted(d['runs'].items()):
            if r['eval']:
                print(f'   run {rid}: last {r["eval"][-1][0]} final val {r["eval"][-1][2]:.4f}')
    d = json.loads((B / 'ts30m_muon.json').read_text())
    b = {x['seed']: x for x in d['base'].values()}[0]
    print('ts30m muon base d0 at 6500:', [e for e in b['eval'] if 6000 <= e[0] <= 7000], 'final', b['eval'][-1])


if __name__ == '__main__':
    for sec in sys.argv[1:]:
        print(f'######## {sec}', flush=True)
        globals()[sec]()
