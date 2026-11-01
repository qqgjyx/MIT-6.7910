"""Compact AdamW/Muon figures of the checkpoint: whole-model loss barriers between pure runs (from lmc_layers.json),
the metric ranking and the metrics under swaps (from rank_metrics.py's metrics.json).

    python -m optsplit.analysis.ckpt_figs barriers --lmc ts10m=<lmc_layers.json> ts30m=... shakespeare=... --out f.pdf
    python -m optsplit.analysis.ckpt_figs forks --dir report/figures/ts10m --out f.pdf
    python -m optsplit.analysis.ckpt_figs ranking --metrics metrics.json --keys k1 k2 ... --out f.pdf
    python -m optsplit.analysis.ckpt_figs swaps --metrics metrics.json --swap-json swap.json --keys k1 ... --out f.pdf
"""

import argparse
import json
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.rank_metrics import COLORS, METRICS, NAMES, loss_axis

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6})
TITLES = {'ts10m': 'TinyStories-10M', 'ts30m': 'TinyStories-30M', 'shakespeare': 'Shakespeare'}
PAIRS = {('adamw', 'muon'): ('AdamW–Muon, same data', 'C3'), ('adamw', 'adamw'): ('AdamW, two data orders', 'C0'),
         ('muon', 'muon'): ('Muon, two data orders', 'C1')}


def barriers(lmc, out):
    """Whole-model loss barriers between final weights, one point per pair, sized to sit next to the 3.3 in tall
    panels of trajectories.py's divergence figure (same font sizes)."""
    with plt.rc_context({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9}):
        fig, ax = plt.subplots(figsize=(3.4, 3.3))
        for i, (name, path) in enumerate(lmc):
            rows = json.load(open(path))
            for j, (pair, (label, c)) in enumerate(PAIRS.items()):
                v = [r['barriers']['all'] for r in rows
                     if tuple(x.split('/')[1].split('_')[0] for x in (r['a'], r['b'])) == pair]
                x = i + (j - 1) * 0.25 + np.linspace(-0.06, 0.06, len(v))
                ax.scatter(x, v, s=14, color=c, label=label if i == 0 else None, lw=0)
        ax.set_xticks(range(len(lmc)))
        ax.set_xticklabels([TITLES.get(n, n).replace('TinyStories', 'TS') for n, _ in lmc])
        ax.set_ylabel('loss barrier (nats)')
        ax.set_ylim(0, 9.5)
        ax.legend(frameon=False, loc='upper left', handletextpad=0.1, markerscale=1.5)
        fig.tight_layout(pad=0.3)
        fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def ranking(metrics, keys, curves, out):
    m = json.load(open(metrics))
    levels = np.array(m['levels'])
    ncurves = len(curves)
    fig = plt.figure(figsize=(7.1, 1.65))
    gs = fig.add_gridspec(1, 1 + ncurves, width_ratios=[1.5] + [1] * ncurves, wspace=0.35)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(keys))
    for i, (p, c) in enumerate((('adamw-muon', 'k'), ('adamw-sgd', '0.55'), ('muon-sgd', '0.8'))):
        ax.barh(y + (i - 1) * 0.27, [m['summary'][k]['ratio'][p] or np.nan for k in keys], height=0.27, color=c,
                label=' – '.join(NAMES[o] for o in p.split('-')))
    ax.set_yticks(y)
    ax.set_yticklabels([METRICS[k][1] for k in keys])
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.axvline(1, color='k', lw=0.6)
    ax.set_xlabel('optimizer gap / data-order gap')
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1), ncol=3, handlelength=1, columnspacing=0.8)
    for j, k in enumerate(curves):
        a = fig.add_subplot(gs[1 + j])
        for rid, pts in m['metrics'][k]['points'].items():
            opt, seed = rid.split('/')[1].split('_d')
            p = np.array(pts)
            loss = np.minimum.accumulate(p[:, 0])
            keep = (loss <= levels[0]) & (loss >= levels[-1]) & np.r_[True, np.diff(loss) < 0]
            a.plot(loss[keep], p[keep, 1], color=COLORS[opt], lw=0.8, ls=['-', '--', ':'][int(seed) % 3],
                   label=NAMES[opt] if seed == '0' else None)
        loss_axis(a)
        a.set_xticks([4, 3, 2.5, 2, 1.7])
        a.set_xticklabels(['4', '3', '2.5', '2', '1.7'])
        a.set_xlim(levels[0], levels[-1])
        a.set_title(METRICS[k][1])
        a.set_xlabel('training loss')
        if j == 0:
            a.legend(frameon=False)
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def swaps(metrics, swap_json, keys, out):
    m = json.load(open(metrics))
    rows = m['swaps']['rows']
    fn = json.load(open(swap_json))
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(7.1, 1.55), sharey=True)
    dirs = ((('adamw', 'muon'), 'C1'), (('muon', 'adamw'), 'C0'))

    def draw(ax, pts, c, label):
        s, v = zip(*sorted(pts))
        ax.plot(s, v, 'o-', color=c, ms=2.5, lw=0.9, label=label)

    for (a, b), c in dirs:
        draw(axes[0], [(r['fork_step'], r['r_js']) for r in fn if (r['opt_a'], r['opt_b']) == (a, b)], c,
             f'{NAMES[a]} → {NAMES[b]}')
        for ax, k in zip(axes[1:], keys):
            draw(ax, [(r['fork_step'], r['metrics'][k]['shift']) for r in rows if (r['opt_a'], r['opt_b']) == (a, b)],
                 c, None)
    axes[0].set_title('function (JS position)')
    axes[0].set_ylabel('0 = parent, 1 = new optimizer')
    axes[0].legend(frameon=False, loc='lower left')
    for ax, k in zip(axes[1:], keys):
        g = rows[0]['metrics'][k]['gap_over_floor']
        ax.set_title(f'{METRICS[k][1]}')
    for ax in axes:
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(1, color='k', lw=0.5, ls='--')
        ax.set_xscale('log')
        ax.set_xticks([100, 1000])
        ax.set_xlabel('fork step')
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def forks(d, out, metrics=None, keys=()):
    """AdamW<->Muon swaps from the existing analyses in directory d (swap.json, lmc_{adamw,muon}.json,
    lmc_into_{adamw,muon}.json, lmc_layers.json): function, barrier to the parent, barrier to the pure run of the new
    optimizer, and the final weight metrics as a fraction of the way from the parent's to the new optimizer's (from
    swap.json, or the given keys of rank_metrics.py's metrics.json)."""
    load = lambda n: json.load(open(f'{d}/{n}.json'))
    rows = json.load(open(metrics))['swaps']['rows'] if metrics else []
    styles = ['-', '--', ':', '-.']
    swap = [r for r in load('swap') if {r['opt_a'], r['opt_b']} == {'adamw', 'muon'}]
    dirs = (('adamw', 'muon', 'C1'), ('muon', 'adamw', 'C0'))
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.6))

    def line(ax, pts, c, ls='-', label=None):
        s, v = zip(*sorted(pts))
        ax.plot(s, v, marker='o', ms=2, lw=0.9, color=c, ls=ls, label=label)

    for a, b, c in dirs:
        rr = [r for r in swap if (r['opt_a'], r['opt_b']) == (a, b)]
        line(axes[0], [(r['fork_step'], r['js_to_b_over_floor']) for r in rr], c, '-')
        line(axes[0], [(r['fork_step'], r['js_to_a_over_floor']) for r in rr], c, '--')
        par = load(f'lmc_{a}')['forks']
        line(axes[1], [(f['fork_step'], f['barrier']) for f in par if f['fork'].split('/')[1].startswith(f'{a}_to_{b}_')], c)
        line(axes[1], [(f['fork_step'], f['barrier']) for f in par
                       if f['fork'].split('/')[1].startswith(f'{a}_to_{a}_d1')], c, ':')
        into = load(f'lmc_into_{b}')['forks']
        line(axes[2], [(f['fork_step'], f['barrier']) for f in into if f['fork'].split('/')[1].startswith(f'{a}_to_{b}_')],
             c, '-', f'{NAMES[a]} → {NAMES[b]}')
        own = [r['barriers']['all'] for r in load('lmc_layers')
               if all(x.split('/')[1].startswith(f'{b}_') for x in (r['a'], r['b']))]
        axes[2].axhspan(min(own), max(own), color=COLORS[b], alpha=0.2, lw=0)
        if rows:
            for k, ls in zip(keys, styles):
                line(axes[3], [(r['fork_step'], r['metrics'][k]['shift']) for r in rows
                               if (r['opt_a'], r['opt_b']) == (a, b)], c, ls)
        else:
            line(axes[3], [(r['fork_step'], r['fm.stable_rank_hidden_mean']) for r in rr], c, '-')
            line(axes[3], [(r['fork_step'], r['fm.weights.dist_init_rel']) for r in rr], c, '--')
    style = lambda ax, labels, loc: ax.legend(
        [plt.Line2D([], [], color='k', lw=0.9, ls=ls) for ls, _ in labels], [t for _, t in labels], frameon=False,
        loc=loc, handlelength=1.6)
    axes[0].axhline(1, color='k', lw=0.5)
    axes[0].set(title='function: distance / floor', ylim=(0, 1.15))
    style(axes[0], [('-', "to the new optimizer's run"), ('--', 'to the parent')], 'lower left')
    axes[1].set_yscale('symlog', linthresh=0.01)
    axes[1].set(title='barrier to the parent (nats)')
    style(axes[1], [('-', 'swap'), (':', 'same optimizer, data order 1')], 'lower left')
    axes[2].set(title="barrier to the new optimizer's run", ylim=(0, 7))
    axes[2].legend(frameon=False, loc='upper right')
    axes[3].axhline(1, color='k', lw=0.5)
    axes[3].set(title='weights: 0 = parent, 1 = new', ylim=(0, 1.15))
    style(axes[3], list(zip(styles, [METRICS[k][1] for k in keys])) if rows else
          [('-', 'stable rank'), ('--', 'distance from init')], 'lower left')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('fork step')
    fig.tight_layout(pad=0.2, w_pad=0.5)
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='fig', required=True)
    s = sub.add_parser('forks')
    s.add_argument('--dir', required=True, help='directory with swap.json, lmc_*.json and lmc_layers.json')
    s.add_argument('--metrics', help="rank_metrics.py's metrics.json with swaps, for the last panel")
    s.add_argument('--keys', nargs='*', default=[])
    s.add_argument('--out', required=True)
    s = sub.add_parser('barriers')
    s.add_argument('--lmc', nargs='+', required=True, help='name=path of lmc_layers.json')
    s.add_argument('--out', required=True)
    s = sub.add_parser('ranking')
    s.add_argument('--metrics', required=True)
    s.add_argument('--keys', nargs='+', required=True)
    s.add_argument('--curves', nargs='+', default=[])
    s.add_argument('--out', required=True)
    s = sub.add_parser('swaps')
    s.add_argument('--metrics', required=True)
    s.add_argument('--swap-json', required=True, help='swap.json of analysis/swap.py, for the function')
    s.add_argument('--keys', nargs='+', required=True)
    s.add_argument('--out', required=True)
    args = p.parse_args()
    if args.fig == 'forks':
        forks(args.dir, args.out, args.metrics, args.keys)
    elif args.fig == 'barriers':
        barriers([tuple(x.split('=', 1)) for x in args.lmc], args.out)
    elif args.fig == 'ranking':
        ranking(args.metrics, args.keys, args.curves, args.out)
    else:
        swaps(args.metrics, args.swap_json, args.keys, args.out)


if __name__ == '__main__':
    main()
