"""Final function metrics per optimizer: mean and standard deviation over data orders at each run's last
function-metrics step, including the largest spectral norm of the hidden (transformer block) matrices, the norm
Muon is predicted to keep small, next to l2 (SGD) and l_inf (Adam).
python -m optsplit.analysis.fm_table 'ts10m_main/*' --out analysis/ts10m"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from optsplit.analysis.io import flatten, load_runs

KEYS = ['weights.l2', 'weights.linf', 'spectral_hidden_max', 'weights.dist_init_rel',
        'weights.log_spectral_complexity', 'input.jac_fro.mean', 'input.hess_top.mean', 'robustness.sensitivity',
        'robustness.acc']


def final_metrics(run):
    fm = run.function_metrics
    if not fm:
        return None
    m = flatten(fm[max(fm)])
    hidden = [v for k, v in m.items() if re.match(r'weights\.layers\.transformer\.h\..*\.spectral$', k)]
    m['spectral_hidden_max'] = max(hidden) if hidden else float('nan')
    return m


def table(runs):
    by_opt = {}
    for r in runs:
        m = final_metrics(r)
        if m is not None:
            by_opt.setdefault(r.optimizer, []).append(m)
    return {opt: {k: (float(np.mean([m.get(k, np.nan) for m in ms])), float(np.std([m.get(k, np.nan) for m in ms])),
                      len(ms)) for k in KEYS}
            for opt, ms in sorted(by_opt.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    res = table(load_runs(args.runs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fm_table.json').write_text(json.dumps(res, indent=1))
    print('| metric | ' + ' | '.join(f'{o} (n={next(iter(v.values()))[2]})' for o, v in res.items()) + ' |')
    print('|:--|' + '--:|' * len(res))
    for k in KEYS:
        print(f'| {k} | ' + ' | '.join(f'{v[k][0]:.4g} ± {v[k][1]:.2g}' for v in res.values()) + ' |')


if __name__ == '__main__':
    main()
