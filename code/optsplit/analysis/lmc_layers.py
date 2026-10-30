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


def plot(rows, path):
    groups = list(rows[0]['barriers'])
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(groups), 3.2))
    width = 0.8 / len(rows)
    for i, r in enumerate(rows):
        label = ' vs '.join(x.split('/')[-1] for x in (r['a'], r['b']))
        ax.bar(np.arange(len(groups)) + i * width, [r['barriers'][g] for g in groups], width, label=label)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2, groups, rotation=30, fontsize=7)
    ax.set(ylabel='loss barrier (nats)')
    ax.legend(fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pairs', nargs='+', help="'<run a>:<run b>' run ids")
    p.add_argument('--n', type=int, default=64, help='training examples for the loss')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--alphas', type=int, default=11)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()

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
    plot(rows, out / 'figures' / 'lmc_layers.png')


if __name__ == '__main__':
    main()
