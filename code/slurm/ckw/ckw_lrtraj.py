"""Fig. 'traj-lr' with SGD centred on its re-tuned rate 0.06: densMAP of the SGD runs at 1/8..2 x 0.06 (d0), the AdamW
and Muon embeddings reused from analysis/ts10m_lr. CPU version of ~/optsplit-umap/lr_traj.py.
python ckw_lrtraj.py OUTDIR"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from optsplit.analysis.distance import loss_points
from optsplit.analysis.io import load_runs
from optsplit.analysis.trajectories import embed, js_matrix
from optsplit.config import root_dir

SRC = root_dir() / 'analysis' / 'ts10m_lr'
OPTS = {'sgd': 'SGD', 'adamw': 'AdamW', 'muon': 'Muon'}
SGD_RUNS = ['ts10m_main_sgd0.06/sgd_d0', 'ts10m_main/sgd_d0', 'ts10m_lr/sgd_lr0.0075', 'ts10m_lr/sgd_lr0.015',
            'ts10m_lr/sgd_lr0.12']
LOSSES = [7, 5, 3, 1.8]
MARKERS = 'osD^'
COLORS = ['#86b6ef', '#4a90e2', '#256abf', '#184f95', '#0a2a55']


def compute_sgd(out, rows=512):
    runs = load_runs(SGD_RUNS)
    n = runs[0].logp(runs[0].pred_steps[0]).shape[0]
    rows = torch.linspace(0, n - 1, min(rows, n)).long()
    index, P = [], []
    for r in runs:
        steps, loss = loss_points(r)
        index += [(r, s, l) for s, l in zip(steps, loss)]
        P += [r.logp(int(s))[rows].float().exp() for s in steps]
    torch.set_num_threads(8)
    D = np.sqrt(js_matrix(torch.stack(P)).numpy())
    res = {}
    for (r, s, l), z in zip(index, embed(D, 'densmap')):
        d = res.setdefault(r.id, dict(optimizer='sgd', lr=r.lr, steps=[], loss=[], xy=[]))
        d['steps'].append(int(s))
        d['loss'].append(float(l))
        d['xy'].append([float(v) for v in z])
    (out / 'trajectories_densmap_sgd.json').write_text(json.dumps(res))
    print({k: (v['lr'], len(v['steps'])) for k, v in res.items()})


def plot(out):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, (opt, title) in zip(axes, OPTS.items()):
        d = json.loads(((out if opt == 'sgd' else SRC) / f'trajectories_densmap_{opt}.json').read_text())
        lrs = sorted({r['lr'] for r in d.values()})
        for r in sorted(d.values(), key=lambda r: r['lr']):
            c = COLORS[lrs.index(r['lr'])]
            xy = np.array(r['xy'])
            ax.plot(*xy.T, '-', lw=1.1, color=c, label=f'{r["lr"]:g}', zorder=2)
            ax.annotate(f'{r["lr"]:g}', xy[-1], xytext=(3, 0), textcoords='offset points', fontsize=6.5,
                        color='0.25', va='center')
            for m, L in zip(MARKERS, LOSSES):
                i = next((i for i, l in enumerate(r['loss']) if l <= L), None)
                if i is not None:
                    ax.plot(*xy[i], m, ms=4.5, mfc=c, mec='k', mew=0.4, zorder=3)
        ax.plot(*np.array(next(iter(d.values()))['xy'][0]), '*', ms=12, color='k', mec='w', mew=0.6, zorder=4)
        ax.set(xticks=[], yticks=[], title=title)
        leg = ax.legend(title='learning rate', fontsize=6.5, title_fontsize=6.5, frameon=False)
        for h in leg.get_lines():
            h.set_linewidth(2.5)
        for s in ax.spines.values():
            s.set_color('0.7')
    keys = [plt.Line2D([], [], ls='', marker='*', ms=10, color='k', label='initialization')]
    keys += [plt.Line2D([], [], ls='', marker=m, ms=5, mfc='w', mec='k', label=f'loss {L:g}')
             for m, L in zip(MARKERS, LOSSES)]
    fig.legend(handles=keys, loc='lower center', ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / 'figures' / 'trajectories_lr_sgd0.06.pdf')
    print('wrote', out / 'figures' / 'trajectories_lr_sgd0.06.pdf')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    compute_sgd(out)
    plot(out)
