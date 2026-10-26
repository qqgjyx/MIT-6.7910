"""Where swaps land: for each fork A->B, the distance of its final predictions to pure A (the parent,
which is also the A-continued control) and pure B (same init and data seed), and the shift of the
function metrics as a fraction of the pure A to pure B gap."""

import argparse
import csv
import json
from pathlib import Path

from optsplit.analysis.distance import distance
from optsplit.analysis.io import flatten, load_runs
from optsplit.config import root_dir


def final_function_metrics(run):
    fm = run.function_metrics
    return flatten(fm[max(fm)]) if fm else {}


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
    fm_f, fm_a, fm_b = (final_function_metrics(r) for r in (fork, pure_a, pure_b))
    for k in fm_f.keys() & fm_a.keys() & fm_b.keys():
        gap = fm_b[k] - fm_a[k]
        row[f'fm.{k}'] = (fm_f[k] - fm_a[k]) / gap if gap != 0 else float('nan')
    return row


def pure_b_of(fork, pures):
    match = [r for r in pures if r.optimizer == fork.optimizer and r.data_seed == fork.data_seed
             and r.init_seed == fork.parent.init_seed and not r.cfg.fork.parent]
    assert len(match) == 1, f'{fork}: expected one pure {fork.optimizer} run, found {match}'
    return match[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forks', nargs='+', required=True, help='glob patterns of fork run ids')
    p.add_argument('--pure', nargs='+', required=True, help='glob patterns of the pure (from init) runs')
    p.add_argument('--out', help='output path without extension (writes .json and .csv)')
    args = p.parse_args()

    pures = load_runs(args.pure)
    forks = [f for f in load_runs(args.forks) if f.cfg.fork.parent and f.optimizer != f.parent.optimizer]
    rows = [landing(f, f.parent, pure_b_of(f, pures)) for f in sorted(forks, key=lambda f: (f.id, f.fork_step))]
    for r in rows:
        print(f"{r['opt_a']}->{r['opt_b']} at step {r['fork_step']}: r_js={r['r_js']:.3f} "
              f"(js to A {r['js_to_a']:.4f}, to B {r['js_to_b']:.4f})")
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
