"""Overview: the loss-matched distance over the data-order floor against training loss, for every pair, one panel
per group of split JSONs (from optsplit.analysis.split or scripts/linear_control.py); JSON=LABEL overrides the label.
python scripts/overview_figure.py --panel Shakespeare a/split_adamw_muon.json ... --panel Linear l.json --out o.png"""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def label(res):
    if 'optimizer_a' not in res:
        return 'SGD vs Adam'
    if res['optimizer_a'] != res['optimizer_b']:
        return f"{res['optimizer_a']} vs {res['optimizer_b']}"
    name = lambda ids: ids[0].split('/')[-1].rsplit('_d', 1)[0].split('_', 1)[-1]
    return f"{name(res['runs_a'])} vs {name(res['runs_b'])}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', nargs='+', action='append', required=True, metavar=('TITLE', 'JSON'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    fig, axes = plt.subplots(1, len(args.panel), figsize=(3.3 * len(args.panel), 3), squeeze=False)
    for ax, (title, *files) in zip(axes[0], args.panel):
        lo, hi = np.inf, 0
        for f in files:
            f, _, lab = f.partition('=')
            res = json.loads(open(f).read())
            grid, ratio = np.array(res['grid']), np.array(res['cross']) / np.array(res['floor'])
            ok = np.isfinite(ratio)
            ax.plot(grid[ok], ratio[ok], lw=1.2, label=lab or label(res))
            lo, hi = min(lo, grid[ok].min()), max(hi, grid[ok].max())
        ax.axhline(1, color='k', lw=0.5)
        ax.axhline(1.5, color='r', ls='--', lw=0.6)
        ax.set(xscale='log' if hi / lo > 10 else 'linear', yscale='log', xlabel='training loss', title=title)
        ax.invert_xaxis()
        ax.legend(fontsize=6)
    axes[0, 0].set_ylabel('distance / data-order floor')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
