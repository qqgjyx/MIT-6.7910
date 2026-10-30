"""Which layers carry the sharp direction at the edge of stability, by optimizer. For each pure run, the share of
the top eigenvector in each layer (embeddings, the blocks, the gains of all LayerNorms, head; `plots._group`)
and the layer's own curvature along it as a fraction of the eigenvalue, averaged over the measured steps at the
edge of stability (sharpness at least 0.9 of its threshold, `eos.at_eos`) and over those below it. The operator
is H, and for AdamW also its preconditioned P^{-1/2}HP^{-1/2}, the one its threshold applies to; for AdamW both
are split by the preconditioned EoS. Muon has no threshold, so its steps are all in one group.
python -m optsplit.analysis.anatomy_layers 'ts10m_main/*' --out analysis/ts10m"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import eos
from optsplit.analysis.io import load_runs
from optsplit.analysis.plots import _group

OPS = {'anatomy': 'H', 'anatomy_precond': 'P^-1/2 H P^-1/2'}


def order(names):
    layers = sorted((n for n in names if n[1:].isdigit()), key=lambda n: int(n[1:]))
    return [n for n in ['emb'] + layers + ['norms', 'head'] if n in names]


def rows(run, delta=0.1):
    at = eos.at_eos(run, delta)
    out = []
    for key, op in OPS.items():
        recs = [r for r in run.records_of('measure') if r.get(key) and r.get('lr', 0) > 0]
        if not recs:
            continue
        phase = (lambda s: 'no threshold') if at is None else (lambda s: 'at EoS' if s in set(at) else 'below EoS')
        for ph in sorted({phase(r['step']) for r in recs}):
            rs = [r for r in recs if phase(r['step']) == ph]
            share = [_group(r[key]['dir'], 'layer') for r in rs]
            lam = [sum(r[key]['self'].values()) + r[key]['cross'] for r in rs]
            own = [{k: v / l for k, v in _group(r[key]['self'], 'layer').items()} for r, l in zip(rs, lam)]
            names = order(share[0])
            out.append(dict(run=run.id, optimizer=run.optimizer, op=op, phase=ph, n=len(rs),
                            steps=[r['step'] for r in rs],
                            share={k: float(np.mean([d.get(k, 0) for d in share])) for k in names},
                            own={k: float(np.mean([d.get(k, 0) for d in own])) for k in names},
                            cross=float(np.mean([r[key]['cross'] / l for r, l in zip(rs, lam)]))))
    return out


def summary(rs):
    """Mean and std over data orders of each (optimizer, operator, phase)."""
    out = {}
    for k in sorted({(r['optimizer'], r['op'], r['phase']) for r in rs}):
        g = [r for r in rs if (r['optimizer'], r['op'], r['phase']) == k]
        names = list(g[0]['share'])
        out[' | '.join(k)] = dict(runs=len(g), steps=sum(r['n'] for r in g), **{
            f: {n: [float(np.mean([r[f][n] for r in g])), float(np.std([r[f][n] for r in g]))] for n in names}
            for f in ('share', 'own')}, cross=[float(np.mean([r['cross'] for r in g])), float(np.std([r['cross'] for r in g]))])
    return out


SHOW = ['sgd | H | at EoS', 'adamw | P^-1/2 H P^-1/2 | at EoS', 'adamw | H | at EoS', 'muon | H | no threshold']


def plot(panels, path):
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.3 * len(panels)), squeeze=False)
    for ax, (title, s) in zip(axes[:, 0], panels.items()):
        keys = [k for k in SHOW if k in s]
        names = list(s[keys[0]]['share'])
        w = 0.8 / len(keys)
        for i, k in enumerate(keys):
            m, sd = np.array([s[k]['share'][n] for n in names]).T
            opt, op, ph = k.split(' | ')
            label = f"{opt}, {'precond. ' if op != 'H' else ''}{ph if ph != 'no threshold' else 'all steps'}"
            ax.bar(np.arange(len(names)) + i * w, m, w, yerr=sd, label=f"{label} ({s[k]['steps']})", color=f'C{i}')
        ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2, names)
        ax.set(ylabel='share of the direction', ylim=(0, 1))
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', metavar='TITLE=JSON', help='only plot earlier anatomy_layers.json files')
    p.add_argument('--delta', type=float, default=0.1)
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    if args.plot:
        panels = {t: json.loads(Path(f).read_text())['summary'] for t, f in (a.split('=', 1) for a in args.plot)}
        return plot(panels, Path(args.out) / 'anatomy_layers.png')
    rs = [r for run in load_runs(args.runs) if not run.cfg.fork.parent for r in rows(run, args.delta)]
    s = summary(rs)
    for k, d in s.items():
        print(f"{k:40s} runs {d['runs']} steps {d['steps']:3d} | " +
              ' '.join(f'{n} {m:.2f}' for n, (m, _) in d['share'].items()) +
              ' | own ' + ' '.join(f'{n} {m:.2f}' for n, (m, _) in d['own'].items() if m > 0.02) +
              f" | cross {d['cross'][0]:.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'anatomy_layers.json').write_text(json.dumps(dict(summary=s, rows=rs), indent=1))
    plot({out.name: s}, out / 'figures' / 'anatomy_layers.png')


if __name__ == '__main__':
    main()
