"""Do the optimizers differ off the data distribution? The final functions of the pure runs compared by the JS
divergence of their next-token predictions, as in the main analysis, on validation sequences, on the same
sequences with their tokens shuffled within each sequence, and on uniformly random tokens. Pairs are grouped as in
`cka`; each pair of optimizers (same data order and init) is reported in units of the floor, the larger of the two
optimizers' mean distance between data orders.
python -m optsplit.analysis.ood 'ts10m_main/*' 'ts10m_initseed/*' --out analysis/ts10m"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.cka import group
from optsplit.analysis.distance import js_divergence
from optsplit.analysis.io import load_runs
from optsplit.analysis.lmc import load_state, setup


def inputs(x, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    shuffled = torch.stack([row[torch.randperm(len(row), generator=g)] for row in x.cpu()])
    return dict(validation=x, shuffled=shuffled.to(x.device),
                random=torch.randint(vocab, x.shape, generator=g).to(x.device))


@torch.no_grad()
def predictions(model, x, chunk):
    return torch.cat([torch.log_softmax(model(x[i:i + chunk]).float(), -1).half() for i in range(0, len(x), chunk)])


def ratios(summary):
    """Each optimizer pair's mean distance over the larger of its two optimizers' data-order distances."""
    return {g: {k: v / max(summary[f'{o}, other data'][k] for o in g.split('-')) for k, v in d.items()}
            for g, d in summary.items() if '-' in g}


def plot(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8), squeeze=False, sharey=True)
    for ax, (title, summary) in zip(axes[0], panels.items()):
        r = ratios(summary)
        kinds = list(next(iter(r.values())))
        width = 0.8 / len(r)
        for i, (g, d) in enumerate(sorted(r.items())):
            ax.bar(np.arange(len(kinds)) + i * width, [d[k] for k in kinds], width, label=g.replace('-', '–'))
        ax.axhline(1, color='k', lw=0.8, ls='--')
        ax.set_xticks(np.arange(len(kinds)) + 0.4 - width / 2, kinds)
        ax.set_title(title, fontsize=9)
    axes[0, 0].set_ylabel('JS / data-order JS')
    axes[0, 0].legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', metavar='TITLE=JSON', help='only plot earlier ood.json files side by side')
    p.add_argument('--n', type=int, default=32, help='validation sequences')
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='output directory')
    args = p.parse_args()
    if args.plot:
        panels = {t: json.loads(Path(f).read_text())['summary'] for t, f in (a.split('=', 1) for a in args.plot)}
        return plot(panels, Path(args.out) / 'ood.png')

    runs = [r for r in load_runs(args.runs) if not r.cfg.fork.parent and r.done]
    model, batches = setup(runs[0].cfg, args.n, args.chunk, 0, args.device, split='val')
    xs = inputs(torch.cat([x for x, _ in batches]), model.lm_head.out_features)
    preds = {}
    for r in runs:
        model.load_state_dict(load_state(r, device=args.device))
        preds[r.id] = {k: predictions(model, x, args.chunk) for k, x in xs.items()}
    rows = [dict(a=a.id, b=b.id, group=group(a, b), js={k: js_divergence(preds[a.id][k], preds[b.id][k]) for k in xs})
            for a, b in combinations(runs, 2) if group(a, b)]
    summary = {g: {k: float(np.mean([r['js'][k] for r in rows if r['group'] == g])) for k in xs}
               for g in sorted({r['group'] for r in rows})}
    for g, d in summary.items():
        print(f'{g:22s} ' + ' '.join(f'{k} {v:.4f}' for k, v in d.items()))
    for g, d in ratios(summary).items():
        print(f'{g:22s} ' + ' '.join(f'{k} {v:.2f}' for k, v in d.items()) + ' floors')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ood.json').write_text(json.dumps(dict(summary=summary, pairs=rows), indent=1))
    plot({out.name: summary}, out / 'figures' / 'ood.png')


if __name__ == '__main__':
    main()
