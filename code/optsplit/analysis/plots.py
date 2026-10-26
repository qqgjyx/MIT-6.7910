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

from optsplit.analysis.distance import step_curve
from optsplit.analysis.io import flatten, load_runs
from optsplit.analysis.split import by_seed, locate_split
from optsplit.config import load_yaml, root_dir


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}')


def _t_star_step(split, run):
    return (split or {}).get('steps', {}).get(run.id)


def divergence(runs_a, runs_b, split, path):
    """Left: distance by step for the cross pairs (same data seed) and the noise-floor pairs.
    Right: loss-matched distance with the floor band and t* (with its CI)."""
    metric = split['metric']
    A, B = by_seed(runs_a), by_seed(runs_b)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, s in enumerate(sorted(A.keys() & B.keys())):
        c = step_curve(A[s], B[s])
        a1.plot(c['step_a'], c[metric], color='C0', label=f"{split['optimizer_a']} vs {split['optimizer_b']}" if i == 0 else None)
    pairs = [(G[s], G[t]) for G in (A, B) for s, t in combinations(sorted(G), 2)]
    for i, (r, q) in enumerate(pairs):
        c = step_curve(r, q)
        a1.plot(c['step_a'], c[metric], color='0.6', lw=0.8, label='same optimizer, other data order' if i == 0 else None)
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
    if np.isfinite(split['t_star_loss']):
        a2.axvline(split['t_star_loss'], color='r', ls='--')
        a2.axvspan(*split['ci_loss'], color='r', alpha=0.1)
    a2.set(xlabel='training loss', ylabel=metric)
    a2.invert_xaxis()
    a2.legend(fontsize=7)
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
        axes[0, c].set(title=f'{run.id}: direction share', xlabel='step', ylim=(0, 1))
        lam = np.array([sum(r[key]['self'].values()) + r[key]['cross'] for r in recs])
        selfs = [_group(r[key]['self'], group) for r in recs]
        for n, col in zip(names, colors):
            axes[1, c].plot(steps, [d.get(n, 0.) for d in selfs] / lam, color=col, lw=1)
        axes[1, c].plot(steps, [r[key]['cross'] for r in recs] / lam, 'k--', lw=1, label='cross')
        axes[1, c].set(title=f'{run.id}: own curvature / eigenvalue', xlabel='step')
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


def fig_anatomy(runs, precond=False, group='type', out=None, name='anatomy'):
    anatomy(load_runs(runs), _path(out, name), precond, group)


def fig_swap(swap_json, split=None, out=None, name='swap'):
    swap_landing(_json(swap_json), _path(out, name), _json(split))


def fig_lmc(lmc_json, out=None, name='lmc'):
    lmc([_json(p) for p in lmc_json], _path(out, name))


def fig_function(runs, keys, out=None, name='function_metrics'):
    function_metrics(load_runs(runs), keys, _path(out, name))


FIGS = dict(divergence=fig_divergence, sharpness=fig_sharpness, anatomy=fig_anatomy, swap=fig_swap,
            lmc=fig_lmc, function=fig_function)


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
    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    FIGS[args.pop('fig')](**args)


if __name__ == '__main__':
    main()
