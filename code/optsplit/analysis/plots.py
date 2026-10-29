"""Main figures. Either `python -m optsplit.analysis.plots <figure> [args]` or
`python -m optsplit.analysis.plots spec.yaml`, where the spec is a list of {fig: <figure>, **args}."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import eos
from optsplit.analysis.distance import step_curve
from optsplit.analysis.io import flatten, load_runs
from optsplit.analysis.split import by_seed, locate_split
from optsplit.config import load_yaml, root_dir


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def _t_star_step(split, run):
    """The run's step at the persistent split, or at the first exceedance if there is none."""
    split = split or {}
    s = split.get('steps_persist', {}).get(run.id)
    return s if s is not None else split.get('steps', {}).get(run.id)


def divergence(runs_a, runs_b, split, path):
    """Left: distance by step for the cross pairs (same data seed) and the noise-floor pairs.
    Right: loss-matched distance with the floor band, the episodes above the floor (shaded), the first
    exceedance t* (dotted) and the persistent split (dashed, with its CI). Third: cross distance over the
    floor, with the split threshold 1 + margin_mult."""
    metric = split['metric']
    A, B = by_seed(runs_a), by_seed(runs_b)
    name = lambda r: r.id.split('/')[-1].rsplit('_d', 1)[0]  # the run name without its data order
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, s in enumerate(sorted(A.keys() & B.keys())):
        c = step_curve(A[s], B[s])
        a1.plot(c['step_a'], c[metric], color='C0', label=f'{name(A[s])} vs {name(B[s])}' if i == 0 else None)
    for G, color in ((A, '0.35'), (B, '0.7')):
        for i, (s, t) in enumerate(combinations(sorted(G), 2)):
            c = step_curve(G[s], G[t])
            a1.plot(c['step_a'], c[metric], color=color, lw=0.8,
                    label=f'{name(G[s])}, other data order' if i == 0 else None)
    steps = [_t_star_step(split, r) for r in A.values()]
    steps = [s for s in steps if s is not None]
    if steps:
        a1.axvline(np.median(steps), color='r', ls='--', label='t*')
    a1.set(xlabel='step', ylabel=metric, xscale='symlog', xlim=(0, None))
    a1.legend(fontsize=7)
    grid = np.array(split['grid'])
    a2.fill_between(grid, split['floor_lo'], split['floor_hi'], color='0.85', label='noise floor (range)')
    a2.plot(grid, split['floor'], color='0.4', ls='--', lw=0.8)
    a2.plot(grid, split['cross'], color='C0')
    for lo, hi in split.get('episodes', []):
        a2.axvspan(lo, hi, color='C0', alpha=0.08)
    if np.isfinite(split['t_star_loss']):
        a2.axvline(split['t_star_loss'], color='r', ls=':', label='first exceedance')
    if np.isfinite(split.get('persist_loss', np.nan)):
        a2.axvline(split['persist_loss'], color='r', ls='--', label='persistent split')
        a2.axvspan(*split['ci_persist'], color='r', alpha=0.1)
    a2.set(xlabel='training loss', ylabel=metric)
    a2.invert_xaxis()
    a2.legend(fontsize=7)
    with np.errstate(all='ignore'):
        a3.plot(grid, np.array(split['cross']) / np.array(split['floor']), color='C0')
    a3.axhline(1 + split['margin_mult'], color='r', lw=0.8, ls='--')
    a3.axhline(1, color='k', lw=0.5)
    a3.set(xlabel='training loss', ylabel='cross / floor', yscale='log')
    a3.invert_xaxis()
    _save(fig, path)


def sharpness(runs, split, path):
    """Sharpness against its stability threshold, per run, with the end of warmup and t*."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        sl, lr = run.series('measure', 'lr')
        sl, lr = sl[lr > 0], lr[lr > 0]
        ax.plot(*run.series('measure', 'lambda_max'), 'C0', label=r'$\lambda_{max}$')
        ax.plot(*run.series('measure', 'batch_sharpness'), 'C1', lw=0.8, label='batch sharpness')
        if run.optimizer == 'sgd':
            ax.plot(sl, (2 + 2 * run.cfg.optim.sgd.momentum) / lr, 'C0--', label=r'$(2+2\mu)/\eta$')
        if run.optimizer == 'adamw':
            b1 = run.cfg.optim.adamw.betas[0]
            ax.plot(*run.series('measure', 'precond_lambda_max'), 'C2', label=r'precond. $\lambda_{max}$')
            ax.plot(sl, (2 + 2 * b1) / ((1 - b1) * lr), 'C2--', label=r'$(2+2\beta_1)/((1-\beta_1)\eta)$')
        ax.axvline(run.cfg.schedule.warmup_steps, color='k', ls=':', label='end of warmup')
        if _t_star_step(split, run) is not None:
            ax.axvline(_t_star_step(split, run), color='r', ls='--', label='t*')
        ax.set(yscale='log', xlabel='step', title=run.id)
        ax.legend(fontsize=6)
    _save(fig, path)


def eos_ratio(runs, split, path):
    """Sharpness over its stability threshold (optimizers that have one); 1 is the edge of stability.
    Dashed: t* of each run."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for c, run in enumerate(runs):
        r = eos.ratio(run)
        if r is None:
            continue
        ax.plot(*r, color=f'C{c}', label=run.id)
        if _t_star_step(split, run) is not None:
            ax.axvline(_t_star_step(split, run), color=f'C{c}', ls='--', lw=0.8)
    ax.axhline(1, color='k', lw=0.5)
    ax.set(xlabel='step', ylabel='sharpness / threshold', xscale='symlog', yscale='log', xlim=(0, None))
    ax.legend(fontsize=7)
    _save(fig, path)


def _group(shares, group):
    if group == 'block':
        return shares
    out = {}
    for b, v in shares.items():
        if b.startswith('L') and '.' in b:
            layer, kind = b.split('.', 1)
            b = kind if group == 'type' else layer
        out[b] = out.get(b, 0.) + v
    return out


def anatomy(runs, path, precond=False, group='type'):
    """Top row: shares of the top Hessian eigenvector by block. Bottom row: each block's own curvature
    along it, v_b^T H_bb v_b, and the cross-block remainder, as fractions of the eigenvalue."""
    key = 'anatomy_precond' if precond else 'anatomy'
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 6), squeeze=False)
    for c, run in enumerate(runs):
        recs = [r for r in run.records_of('measure') if r.get(key)]
        if not recs:
            continue
        steps = [r['step'] for r in recs]
        shares = [_group(r[key]['dir'], group) for r in recs]
        names = list(dict.fromkeys(k for d in shares for k in d))
        colors = plt.cm.tab20(np.arange(len(names)) % 20)
        M = np.clip([[d.get(n, 0.) for n in names] for d in shares], 0, None)
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-30)
        axes[0, c].stackplot(steps, M.T, labels=names, colors=colors)
        axes[0, c].set(title=run.id, xlabel='step', ylabel='share of the direction', ylim=(0, 1))
        lam = np.array([sum(r[key]['self'].values()) + r[key]['cross'] for r in recs])
        selfs = [_group(r[key]['self'], group) for r in recs]
        for n, col in zip(names, colors):
            axes[1, c].plot(steps, [d.get(n, 0.) for d in selfs] / lam, color=col, lw=1)
        axes[1, c].plot(steps, [r[key]['cross'] for r in recs] / lam, 'k--', lw=1, label='cross')
        axes[1, c].set(xlabel='step', ylabel='own curvature / eigenvalue')
    axes[0, -1].legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1, -1].legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
    _save(fig, path)


def swap_landing(rows, path, split=None):
    """Where each fork lands (0 = pure A, 1 = pure B) against the fork step."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for c, pair in enumerate(sorted({(r['opt_a'], r['opt_b']) for r in rows})):
        rs = sorted((r for r in rows if (r['opt_a'], r['opt_b']) == pair), key=lambda r: r['fork_step'])
        ax.plot([r['fork_step'] for r in rs], [r['r_js'] for r in rs], '-o', color=f'C{c}', label='->'.join(pair))
        t = (split or {}).get('steps', {}).get(rs[0]['pure_a'])
        if t is not None:
            ax.axvline(t, color=f'C{c}', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(1, color='k', lw=0.5)
    ax.set(xlabel='fork step', ylabel='landing position (JS)', xscale='symlog', xlim=(0, None))
    ax.legend(fontsize=7)
    _save(fig, path)


def lmc(results, path):
    """Loss barrier between the parent's and each fork's final weights against the fork step."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for res in results:
        for opt in sorted({f['optimizer'] for f in res['forks']}):
            fs = sorted((f for f in res['forks'] if f['optimizer'] == opt), key=lambda f: f['fork_step'])
            ax.plot([f['fork_step'] for f in fs], [f['barrier'] for f in fs], '-o', label=f"{res['parent']} -> {opt}")
    ax.set(xlabel='fork step', ylabel='loss barrier', xscale='symlog', xlim=(0, None))
    ax.legend(fontsize=7)
    _save(fig, path)


def function_metrics(runs, keys, path):
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2), squeeze=False)
    for c, run in enumerate(runs):
        fm = run.function_metrics
        steps = sorted(fm)
        for ax, key in zip(axes[0], keys):
            ax.plot(steps, [flatten(fm[s]).get(key, np.nan) for s in steps], '-o', ms=3, color=f'C{c}', label=run.id)
            ax.set(title=key, xlabel='step', xscale='symlog', xlim=(0, None))
    axes[0, 0].legend(fontsize=6)
    _save(fig, path)


def losses(runs, path):
    """Probe training and validation loss against step: one color per optimizer, data order 0 thick."""
    opts = sorted({r.optimizer for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for run in runs:
        for ax, key in zip(axes, ('train_loss', 'val_loss')):
            ax.plot(*run.series('eval', key), color=f'C{opts.index(run.optimizer)}',
                    lw=1.2 if run.data_seed == 0 else 0.5, label=run.optimizer if run.data_seed == 0 else None)
    for ax, key in zip(axes, ('train loss', 'validation loss')):
        ax.set(xlabel='step', ylabel=key, xscale='symlog', xlim=(0, None))
    axes[0].legend(fontsize=7)
    _save(fig, path)


def sweep(runs, path, select='final'):
    """One panel per optimizer with the validation loss of each learning rate against step, and a last panel
    with the final (or minimum) validation loss against the learning rate; x marks runs that diverged."""
    opts = sorted({r.optimizer for r in runs})
    fig, axes = plt.subplots(1, len(opts) + 1, figsize=(4 * (len(opts) + 1), 3.2), squeeze=False)
    axes = axes[0]
    curves = {r.id: r.series('eval', 'val_loss') for r in runs}
    best = min(v.min() for _, v in curves.values() if len(v))
    ylim = (0.97 * best, 2 * best)
    for c, opt in enumerate(opts):
        rs = sorted((r for r in runs if r.optimizer == opt), key=lambda r: r.lr)
        pts = []
        for r, col in zip(rs, plt.cm.viridis(np.linspace(0, 0.9, len(rs)))):
            s, v = curves[r.id]
            axes[c].plot(s, v, color=col, label=f'lr {r.lr:g}')
            diverged = s[-1] < r.cfg.train.max_steps
            pts.append((r.lr, ylim[1] if diverged else v.min() if select == 'min' else v[-1], diverged))
        axes[c].set(title=opt, xlabel='step', ylabel='validation loss', ylim=ylim)
        axes[c].legend(fontsize=6)
        lr, val, div = map(np.array, zip(*pts))
        axes[-1].plot(lr[~div], val[~div], '-o', ms=4, color=f'C{c}', label=opt)
        axes[-1].plot(lr[div], val[div], 'x', color=f'C{c}', clip_on=False)
    axes[-1].set(xscale='log', xlabel='learning rate', ylabel=f'{select} validation loss', ylim=ylim)
    axes[-1].legend(fontsize=7)
    _save(fig, path)


def _path(out, name):
    return Path(out or root_dir() / 'analysis' / 'figures') / f'{name}.png'


def _json(path):
    return json.loads(Path(path).read_text()) if path else None


def fig_divergence(runs_a, runs_b, split=None, metric='js', out=None, name='divergence'):
    runs_a, runs_b = load_runs(runs_a), load_runs(runs_b)
    divergence(runs_a, runs_b, _json(split) or locate_split(runs_a, runs_b, metric=metric, n_boot=200),
               _path(out, name))


def fig_sharpness(runs, split=None, out=None, name='sharpness'):
    sharpness(load_runs(runs), _json(split), _path(out, name))


def fig_eos(runs, split=None, out=None, name='eos'):
    eos_ratio(load_runs(runs), _json(split), _path(out, name))


def fig_anatomy(runs, precond=False, group='type', out=None, name='anatomy'):
    anatomy(load_runs(runs), _path(out, name), precond, group)


def fig_swap(swap_json, split=None, out=None, name='swap'):
    swap_landing(_json(swap_json), _path(out, name), _json(split))


def fig_lmc(lmc_json, out=None, name='lmc'):
    lmc([_json(p) for p in lmc_json], _path(out, name))


def fig_function(runs, keys, out=None, name='function_metrics'):
    function_metrics(load_runs(runs), keys, _path(out, name))


def fig_losses(runs, out=None, name='losses'):
    losses(load_runs(runs), _path(out, name))


def fig_sweep(runs, select='final', out=None, name='sweep'):
    sweep(load_runs(runs), _path(out, name), select)


FIGS = dict(divergence=fig_divergence, sharpness=fig_sharpness, eos=fig_eos, anatomy=fig_anatomy, swap=fig_swap,
            lmc=fig_lmc, function=fig_function, sweep=fig_sweep, losses=fig_losses)


def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(('.yaml', '.yml')):
        for spec in load_yaml(sys.argv[1]):
            FIGS[spec.pop('fig')](**spec)
        return
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--out', help='output directory')
    common.add_argument('--name', help='file name (without .png)')
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='fig', required=True)
    s = sub.add_parser('divergence', parents=[common])
    s.add_argument('--runs-a', nargs='+', required=True)
    s.add_argument('--runs-b', nargs='+', required=True)
    s.add_argument('--split', help='JSON from optsplit.analysis.split (computed if omitted)')
    s.add_argument('--metric', default='js', choices=['js', 'disagree'])
    s = sub.add_parser('sharpness', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s.add_argument('--split')
    s = sub.add_parser('eos', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s.add_argument('--split')
    s = sub.add_parser('anatomy', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s.add_argument('--precond', action='store_true')
    s.add_argument('--group', default='type', choices=['block', 'type', 'layer'])
    s = sub.add_parser('swap', parents=[common])
    s.add_argument('--swap-json', required=True)
    s.add_argument('--split')
    s = sub.add_parser('lmc', parents=[common])
    s.add_argument('--lmc-json', nargs='+', required=True)
    s = sub.add_parser('function', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s.add_argument('--keys', nargs='+', required=True, help='dotted keys of function_metrics/*.json')
    s = sub.add_parser('losses', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s = sub.add_parser('sweep', parents=[common])
    s.add_argument('--runs', nargs='+', required=True)
    s.add_argument('--select', default='final', choices=['final', 'min'],
                   help='validation loss to compare learning rates by (min for runs that overfit)')
    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    FIGS[args.pop('fig')](**args)


if __name__ == '__main__':
    main()
