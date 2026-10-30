"""Where swaps land: for each fork A->B, the distance of its final predictions to pure A (the parent,
which is also the A-continued control) and pure B (same init and data seed), and the shift of the
function metrics as a fraction of the pure A to pure B gap. Each distance also comes divided by its
noise floor: the median final distance between the pure run and the same optimizer's other data orders."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optsplit.analysis.distance import distance
from optsplit.analysis.fm_table import final_metrics
from optsplit.analysis.io import load_runs
from optsplit.config import root_dir


def landing(fork, pure_a, pure_b):
    sf, sa, sb = fork.final_pred_step, pure_a.final_pred_step, pure_b.final_pred_step
    js_a, ag_a = distance(fork, sf, pure_a, sa)
    js_b, ag_b = distance(fork, sf, pure_b, sb)
    js_ab, ag_ab = distance(pure_a, sa, pure_b, sb)
    row = dict(fork=fork.id, pure_a=pure_a.id, pure_b=pure_b.id, fork_step=fork.fork_step,
               opt_a=pure_a.optimizer, opt_b=fork.optimizer, js_to_a=js_a, js_to_b=js_b, js_ab=js_ab,
               agree_to_a=ag_a, agree_to_b=ag_b, agree_ab=ag_ab,
               r_js=js_a / (js_a + js_b) if js_a + js_b > 0 else float('nan'),
               r_disagree=(1 - ag_a) / (2 - ag_a - ag_b) if ag_a + ag_b < 2 else float('nan'))
    fm_f, fm_a, fm_b = (final_metrics(r) or {} for r in (fork, pure_a, pure_b))
    for k in fm_f.keys() & fm_a.keys() & fm_b.keys():
        gap = fm_b[k] - fm_a[k]
        row[f'fm.{k}'] = (fm_f[k] - fm_a[k]) / gap if gap != 0 else float('nan')
    return row


def pure_b_of(fork, pures):
    match = [r for r in pures if r.optimizer == fork.optimizer and r.data_seed == fork.data_seed
             and r.init_seed == fork.parent.init_seed and not r.cfg.fork.parent]
    assert len(match) == 1, f'{fork}: expected one pure {fork.optimizer} run, found {match}'
    return match[0]


def final_floor(run, pures):
    """Median final JS between `run` and the pure runs of its optimizer with other data orders."""
    others = [r for r in pures if r.optimizer == run.optimizer and r.init_seed == run.init_seed
              and r.data_seed != run.data_seed and not r.cfg.fork.parent]
    return float(np.median([distance(run, run.final_pred_step, r, r.final_pred_step)[0] for r in others]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forks', nargs='+', required=True, help='glob patterns of fork run ids')
    p.add_argument('--pure', nargs='+', required=True, help='glob patterns of the pure (from init) runs')
    p.add_argument('--out', help='output path without extension (writes .json and .csv)')
    args = p.parse_args()

    pures = load_runs(args.pure)
    forks = [f for f in load_runs(args.forks) if f.cfg.fork.parent and f.optimizer != f.parent.optimizer and f.done]
    rows = [landing(f, f.parent, pure_b_of(f, pures)) for f in sorted(forks, key=lambda f: (f.id, f.fork_step))]
    floors = {}
    for r in rows:
        for end in ('a', 'b'):
            run = next(x for x in pures if x.id == r[f'pure_{end}'])
            floors.setdefault(run.id, final_floor(run, pures))
            r[f'floor_{end}'] = floors[run.id]
            r[f'js_to_{end}_over_floor'] = r[f'js_to_{end}'] / floors[run.id]
    for r in rows:
        print(f"{r['opt_a']}->{r['opt_b']} at step {r['fork_step']}: r_js={r['r_js']:.3f} "
              f"(js to A {r['js_to_a']:.4f} = {r['js_to_a_over_floor']:.2f} floors, "
              f"to B {r['js_to_b']:.4f} = {r['js_to_b_over_floor']:.2f} floors)")
    out = Path(args.out or root_dir() / 'analysis' / 'swap' / forks[0].id.split('/')[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f'{out}.json').write_text(json.dumps(rows, indent=1))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f'{out}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}.json/.csv')


if __name__ == '__main__':
    main()
