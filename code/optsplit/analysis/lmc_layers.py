"""Layer-wise linear mode connectivity (Adilova et al., 2024): the loss barrier when only one group of parameters
(the embeddings, one transformer block, or the final norm and head) moves along the straight line from run A's
final weights to run B's, the rest staying at A's. Localizes where two solutions are not linearly connected.
python -m optsplit.analysis.lmc_layers ts10m_main/adamw_d0:ts10m_main/muon_d0 ... --out analysis/ts10m"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.io import Run
from optsplit.analysis.lmc import barrier, interpolation, load_state, setup


def group(key):
    parts = key.split('.')
    if parts[:2] == ['transformer', 'h']:
        return f'block {parts[2]}'
    return 'embeddings' if parts[1] in ('wte', 'wpe') else 'head'


def layerwise(model, sd_a, sd_b, batches, n_alphas):
    out = {'all': barrier(*interpolation(model, sd_a, sd_b, batches, n_alphas))}
    for g in dict.fromkeys(map(group, sd_a)):
        moved = {k: sd_b[k] if group(k) == g else v for k, v in sd_a.items()}
        out[g] = barrier(*interpolation(model, sd_a, moved, batches, n_alphas))
    return out


def kind(row):
    a, b = (x.split('/')[-1].rsplit('_', 1)[0] for x in (row['a'], row['b']))
    return f'{a}, data orders' if a == b else f'{a}–{b}'


def plot(panels, path):
    """One row of bars per {title: rows}: the mean barrier of each kind of pair (two optimizers on the same data
    order, or two data orders of one optimizer) in each parameter group, with the standard deviation over pairs."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), squeeze=False)
    for ax, (title, rows) in zip(axes[:, 0], panels.items()):
        groups = list(rows[0]['barriers'])
        kinds = list(dict.fromkeys(map(kind, rows)))
        width = 0.8 / len(kinds)
        for i, k in enumerate(kinds):
            b = np.array([[r['barriers'][g] for g in groups] for r in rows if kind(r) == k])
            ax.bar(np.arange(len(groups)) + i * width, b.mean(0), width, yerr=b.std(0) if len(b) > 1 else None,
                   error_kw=dict(lw=0.6), label=f'{k} ({len(b)})')
        ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2, groups, fontsize=7)
        ax.set(ylabel='loss barrier (nats)', title=title)
    axes[0, 0].legend(fontsize=6, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pairs', nargs='*', help="'<run a>:<run b>' run ids")
    p.add_argument('--plot', nargs='+', metavar='TITLE=JSON[,JSON...]', help='only plot earlier lmc_layers.json '
                   'files, several per panel if comma-separated')
    p.add_argument('--n', type=int, default=64, help='training examples for the loss')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--alphas', type=int, default=11)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    if args.plot:
        panels = {t: [r for f in fs.split(',') for r in json.loads(Path(f).read_text())]
                  for t, fs in (a.split('=', 1) for a in args.plot)}
        return plot(panels, Path(args.out) / 'lmc_layers.png')

    pairs = [tuple(map(Run, s.split(':'))) for s in args.pairs]
    model, batches = setup(pairs[0][0].cfg, args.n, args.chunk, 0, args.device)
    rows = []
    for a, b in pairs:
        bars = layerwise(model, load_state(a, device=args.device), load_state(b, device=args.device), batches,
                         args.alphas)
        rows.append(dict(a=a.id, b=b.id, barriers=bars))
        print(f'{a.id} vs {b.id}: ' + ', '.join(f'{g} {v:.3f}' for g, v in bars.items()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'lmc_layers.json').write_text(json.dumps(rows, indent=1))
    plot({out.name: rows}, out / 'figures' / 'lmc_layers.png')


if __name__ == '__main__':
    main()
