"""Representation similarity: linear CKA (Kornblith et al., 2019) between the final hidden states of pairs of
pure runs on a fixed batch of training sequences, after each transformer block and after the final LayerNorm.
Pairs are grouped into other optimizer (same init and data order), same optimizer with another data order, and
same optimizer from another init, to ask whether the optimizer leaves a mark in the representations as it does
in the weights.
python -m optsplit.analysis.cka 'ts10m_main/*' 'ts10m_initseed/*' --out analysis/ts10m"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.io import load_runs
from optsplit.analysis.lmc import load_state, setup


@torch.no_grad()
def features(model, batches):
    """Hidden states after each block and after the final LayerNorm, as (tokens, width) matrices."""
    layers = [*model.transformer.h, model.transformer.ln_f]
    feats = []
    hooks = [m.register_forward_hook(lambda _, __, out: feats.append(out.flatten(0, 1).float())) for m in layers]
    for x, _ in batches:
        model(x)
    for h in hooks:
        h.remove()
    return [torch.cat(feats[i::len(layers)]) for i in range(len(layers))]


def cka(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(0), y - y.mean(0)
    return float((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm()))


def group(a, b):
    if a.optimizer != b.optimizer:
        same = a.data_seed == b.data_seed and a.init_seed == b.init_seed
        return '-'.join(sorted((a.optimizer, b.optimizer))) if same else None
    return f'{a.optimizer}, other init' if a.init_seed != b.init_seed else f'{a.optimizer}, other data'


def plot(panels, path):
    """One panel per {title: summary}, one line per group: solid for other optimizer, dashed for other data,
    dotted for other init; each optimizer keeps its color."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False, sharey=True)
    style = lambda g: ':' if 'init' in g else '--' if 'data' in g else '-'
    for ax, (title, summary) in zip(axes[0], panels.items()):
        colors = {}
        for g in sorted(summary, key=lambda g: ('-' in g, g)):
            color = colors.setdefault(g.split(',')[0], f'C{len(colors)}')
            ax.plot(range(len(summary[g])), summary[g], style(g), marker='o', ms=3, color=color, label=g)
        ax.set(xlabel='block (last: final LayerNorm)', title=title)
    axes[0, 0].set_ylabel('linear CKA')
    handles = {l: h for ax in axes[0] for h, l in zip(*ax.get_legend_handles_labels())}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5, fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', metavar='TITLE=JSON', help='only plot earlier cka.json files side by side')
    p.add_argument('--n', type=int, default=64, help='training sequences')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    if args.plot:
        panels = {t: json.loads(Path(f).read_text())['summary'] for t, f in (a.split('=', 1) for a in args.plot)}
        return plot(panels, Path(args.out) / 'cka.png')

    runs = [r for r in load_runs(args.runs) if not r.cfg.fork.parent and r.done]
    model, batches = setup(runs[0].cfg, args.n, args.chunk, 0, args.device)
    feats = {}
    for r in runs:
        model.load_state_dict(load_state(r, device=args.device))
        feats[r.id] = features(model, batches)
    rows = [dict(a=a.id, b=b.id, group=group(a, b), cka=[cka(x, y) for x, y in zip(feats[a.id], feats[b.id])])
            for a, b in combinations(runs, 2) if group(a, b)]
    summary = {g: np.mean([r['cka'] for r in rows if r['group'] == g], axis=0).tolist()
               for g in sorted({r['group'] for r in rows})}
    for g, ys in summary.items():
        print(f'{g:22s} ' + ' '.join(f'{y:.3f}' for y in ys))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cka.json').write_text(json.dumps(dict(summary=summary, pairs=rows), indent=1))
    plot({out.name: summary}, out / 'figures' / 'cka.png')


if __name__ == '__main__':
    main()
