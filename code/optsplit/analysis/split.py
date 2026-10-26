"""Locate the split t* between two optimizers: the first loss level at which their loss-matched
distance exceeds the seed-to-seed noise floor, with a bootstrap over seeds.

Runs of the two optimizers with the same `train.data_seed` share initialization and batch order and form
the cross pairs; runs of one optimizer with different data seeds give its noise floor."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from optsplit.analysis.distance import loss_curve, loss_points
from optsplit.analysis.io import load_runs
from optsplit.config import root_dir


def by_seed(runs):
    out = {}
    for r in runs:
        assert r.data_seed not in out, f'two runs with data_seed {r.data_seed}: {out[r.data_seed]}, {r}'
        out[r.data_seed] = r
    return out


def on_grid(curve, grid, metric):
    """Interpolate a loss-matched curve onto loss levels (NaN outside its range)."""
    loss, idx = np.unique(curve['loss'], return_index=True)
    if len(loss) < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, loss, curve[metric][idx], left=np.nan, right=np.nan)


def loss_grid(runs, reference):
    """Loss levels of `reference`'s checkpoints (high to low) inside the range every run covers."""
    pts = [loss_points(r)[1] for r in runs]
    hi, lo = min(p.max() for p in pts), max(p.min() for p in pts)
    levels = np.unique(loss_points(reference)[1])[::-1]
    return levels[(levels <= hi) & (levels >= lo)]


def step_at_loss(run, level):
    """First prediction step at which the run's running-minimum training loss is <= level."""
    if not np.isfinite(level):
        return None
    steps, loss = loss_points(run)
    hit = np.nonzero(loss <= level)[0]
    return int(steps[hit[0]]) if len(hit) else None


def first_exceed(grid, cross, floor, margin_mult, margin_add, k):
    over = cross > floor * (1 + margin_mult) + margin_add
    count = 0
    for i, o in enumerate(over):
        count = count + 1 if o else 0
        if count == k:
            return float(grid[i - k + 1])
    return float('nan')


def _mean(curves):
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    with np.errstate(all='ignore'):
        return np.nanmean(np.stack(curves), 0)


def locate_split(runs_a, runs_b, metric='js', margin_mult=0.5, margin_add=0.0, k=3, tol=0.02,
                 n_boot=1000, seed=0):
    A, B = by_seed(runs_a), by_seed(runs_b)
    seeds = sorted(A.keys() & B.keys())
    assert seeds, 'no data seed shared by the two groups of runs'
    grid = loss_grid(list(A.values()) + list(B.values()), A[seeds[0]])
    cross = {s: on_grid(loss_curve(A[s], B[s], tol), grid, metric) for s in seeds}
    floors = {(g, s, t): on_grid(loss_curve(G[s], G[t], tol), grid, metric)
              for g, G in (('a', A), ('b', B)) for s, t in combinations(sorted(G), 2)}
    assert floors, 'the noise floor needs two data seeds of at least one optimizer'

    def floor_of(sample):
        sample = set(sample)
        fa = _mean([c for (g, s, t), c in floors.items() if g == 'a' and s in sample and t in sample])
        fb = _mean([c for (g, s, t), c in floors.items() if g == 'b' and s in sample and t in sample])
        if fa is None and fb is None:
            return None
        return fa if fb is None else fb if fa is None else np.fmax(fa, fb)

    def estimate(sample):
        floor = floor_of(sample)
        if floor is None:
            return float('nan')
        return first_exceed(grid, _mean([cross[s] for s in sample]), floor, margin_mult, margin_add, k)

    point = estimate(seeds)
    rng = np.random.default_rng(seed)
    boot = np.array([estimate(rng.choice(seeds, len(seeds))) for _ in range(n_boot)])
    found = boot[np.isfinite(boot)]
    ci = np.percentile(found, [2.5, 97.5]).tolist() if len(found) else [float('nan')] * 2
    all_floors = np.stack(list(floors.values()))
    with np.errstate(all='ignore'):
        floor_lo, floor_hi = np.nanmin(all_floors, 0), np.nanmax(all_floors, 0)
    runs = list(A.values()) + list(B.values())
    return dict(
        metric=metric, margin_mult=margin_mult, margin_add=margin_add, k=k, tol=tol,
        optimizer_a=A[seeds[0]].optimizer, optimizer_b=B[seeds[0]].optimizer, seeds=seeds,
        runs_a=[r.id for r in A.values()], runs_b=[r.id for r in B.values()],
        t_star_loss=point, ci_loss=ci, boot_found=float(len(found) / max(n_boot, 1)),
        steps={r.id: step_at_loss(r, point) for r in runs},
        step_ci={r.id: [step_at_loss(r, ci[1]), step_at_loss(r, ci[0])] for r in runs},
        grid=grid.tolist(), cross=_mean(list(cross.values())).tolist(), floor=floor_of(seeds).tolist(),
        floor_lo=floor_lo.tolist(), floor_hi=floor_hi.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-a', nargs='+', required=True, help='glob patterns of run ids')
    p.add_argument('--runs-b', nargs='+', required=True)
    p.add_argument('--metric', default='js', choices=['js', 'disagree'])
    p.add_argument('--margin-mult', type=float, default=0.5)
    p.add_argument('--margin-add', type=float, default=0.0)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--tol', type=float, default=0.02)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--out')
    args = p.parse_args()
    res = locate_split(load_runs(args.runs_a), load_runs(args.runs_b), args.metric, args.margin_mult,
                       args.margin_add, args.k, args.tol, args.n_boot)
    lo, hi = res['ci_loss']
    print(f"{res['optimizer_a']} vs {res['optimizer_b']}: t* at loss {res['t_star_loss']:.4f} "
          f"(95% CI {lo:.4f}-{hi:.4f}, split found in {res['boot_found']:.0%} of bootstraps)")
    for rid, s in res['steps'].items():
        print(f'  {rid}: step {s} (CI {res["step_ci"][rid]})')
    suite = res['runs_a'][0].split('/')[0]
    out = Path(args.out or root_dir() / 'analysis' / 'split' / f"{suite}_{res['optimizer_a']}_vs_{res['optimizer_b']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
