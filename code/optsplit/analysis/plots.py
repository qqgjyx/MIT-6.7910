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

from optsplit.analysis import eos, trajectories
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


def divergence(runs_a, runs_b, split, path, traj=None):
    """Left: distance by step for the cross pairs (same data seed) and the noise-floor pairs.
    Right: loss-matched distance with the floor band, the episodes above the floor (shaded), the first
    exceedance t* (dotted) and the persistent split (dashed, with its CI). Third: cross distance over the
    floor, with the split threshold 1 + margin_mult. With `traj` (trajectories and losses to mark, from
    `trajectories.parse`), the UMAP of the prediction trajectories comes first instead of the third panel."""
    metric = split['metric']
    A, B = by_seed(runs_a), by_seed(runs_b)
    name = lambda r: r.id.split('/')[-1].rsplit('_d', 1)[0]  # the run name without its data order
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    if traj:
        trajectories.draw(axes[0], *traj)
        a1, a2, a3 = *axes[1:], None
    else:
        a1, a2, a3 = axes
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
    if a3 is not None:
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
    """Left: where each fork lands (0 = pure A, 1 = pure B) against the fork step. Right (if the rows have
    floors): its distance to pure B (solid) and to pure A (dashed) in units of their noise floors."""
    floors = 'js_to_b_over_floor' in rows[0]
    fig, axes = plt.subplots(1, 1 + floors, figsize=(5 * (1 + floors), 3.4), squeeze=False)
    ax = axes[0, 0]
    for c, pair in enumerate(sorted({(r['opt_a'], r['opt_b']) for r in rows})):
        rs = sorted((r for r in rows if (r['opt_a'], r['opt_b']) == pair), key=lambda r: r['fork_step'])
        steps = [r['fork_step'] for r in rs]
        ax.plot(steps, [r['r_js'] for r in rs], '-o', color=f'C{c}', label='->'.join(pair))
        t = (split or {}).get('steps', {}).get(rs[0]['pure_a'])
        if t is not None:
            ax.axvline(t, color=f'C{c}', ls='--', lw=0.8)
        if floors:
            axes[0, 1].plot(steps, [r['js_to_b_over_floor'] for r in rs], '-o', ms=3, color=f'C{c}', label='->'.join(pair))
            axes[0, 1].plot(steps, [r['js_to_a_over_floor'] for r in rs], '--', color=f'C{c}')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(1, color='k', lw=0.5)
    ax.set(xlabel='fork step', ylabel='landing position (JS)', xscale='log')
    ax.legend(fontsize=7)
    if floors:
        axes[0, 1].axhline(1, color='k', lw=0.5)
        axes[0, 1].set(xlabel='fork step', ylabel='JS to pure run / its noise floor', xscale='log', yscale='log',
                       title='solid: to the new optimizer, dashed: to the old')
        axes[0, 1].title.set_fontsize(8)
        axes[0, 1].legend(fontsize=6, ncol=2)
    _save(fig, path)


SWAP_METRICS = {'weights.dist_init_rel': 'distance moved from init', 'linf_over_l2': 'l_inf / l2 of the weights',
                'stable_rank_hidden_mean': 'stable rank, hidden matrices', 'input.jac_fro.mean': 'input Jacobian'}


def swap_metrics(rows, path, keys=SWAP_METRICS, min_snr=3):
    """Final function metrics of each fork as a fraction of the pure A to pure B gap (0 = like pure A, 1 = like
    pure B) against the fork step, one panel per metric, one line per pair; clipped to [-0.5, 1.5]. Pairs whose pure
    A to pure B gap is under `min_snr` data-order standard deviations are left out."""
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2), squeeze=False, sharey=True)
    pairs = sorted({(r['opt_a'], r['opt_b']) for r in rows})
    for ax, (k, title) in zip(axes[0], keys.items()):
        for c, pair in enumerate(pairs):
            rs = sorted((r for r in rows if (r['opt_a'], r['opt_b']) == pair), key=lambda r: r['fork_step'])
            y = [r.get(f'fm.{k}', np.nan) if r.get(f'fm_snr.{k}', np.inf) >= min_snr else np.nan
                 for r in rs]
            ax.plot([r['fork_step'] for r in rs], y, '-o', ms=3, color=f'C{c}', label='->'.join(pair))
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(1, color='k', lw=0.5)
        ax.set(xlabel='fork step', title=title, xscale='log', ylim=(-0.5, 1.5))
        ax.title.set_fontsize(9)
    axes[0, 0].set_ylabel('0 = old optimizer, 1 = new')
    axes[0, 0].legend(fontsize=6)
    _save(fig, path)


def lmc(results, path):
    """Loss barrier between the parent's and each fork's final weights against the fork step, one panel per
    parent, one line per (fork's source optimizer, fork's optimizer); dotted lines: the barrier to the other pure
    runs (`others`), which share nothing but the init."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 3.4), squeeze=False, sharey=True)
    src = lambda f: f['fork'].split('/')[-1].split('_to_')[0]
    opts = sorted({o for res in results for f in res['forks'] for o in (src(f), f['optimizer'])})
    color = {o: f'C{i}' for i, o in enumerate(opts)}
    for ax, res in zip(axes[0], results):
        for key in sorted({(src(f), f['optimizer']) for f in res['forks']}):
            fs = sorted((f for f in res['forks'] if (src(f), f['optimizer']) == key), key=lambda f: f['fork_step'])
            other = key[1] if key[0] == res['optimizer'] else key[0]
            tag = ' (other data)' if key[0] == key[1] else ''
            ax.plot([f['fork_step'] for f in fs], [f['barrier'] for f in fs], '-o', ms=3, color=color[other],
                    label=f'{key[0]} -> {key[1]}{tag}')
        for o in res.get('others', []):
            if o['run'] != res['parent']:
                ax.axhline(o['barrier'], color=color.get(o['optimizer'], 'k'), ls=':', lw=1)
        ax.set(xlabel='fork step', xscale='log', title=f"parent {res['parent']}")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel('loss barrier')
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


def fig_divergence(runs_a, runs_b, split=None, metric='js', traj=None, out=None, name='divergence'):
    runs_a, runs_b = load_runs(runs_a), load_runs(runs_b)
    divergence(runs_a, runs_b, _json(split) or locate_split(runs_a, runs_b, metric=metric, n_boot=200),
               _path(out, name), traj and trajectories.parse(traj))


def fig_sharpness(runs, split=None, out=None, name='sharpness'):
    sharpness(load_runs(runs), _json(split), _path(out, name))


def fig_eos(runs, split=None, out=None, name='eos'):
    eos_ratio(load_runs(runs), _json(split), _path(out, name))


def fig_anatomy(runs, precond=False, group='type', out=None, name='anatomy'):
    anatomy(load_runs(runs), _path(out, name), precond, group)


def fig_swap(swap_json, split=None, out=None, name='swap'):
    rows = _json(swap_json)
    swap_landing(rows, _path(out, name), _json(split))
    if any(k.startswith('fm.') for k in rows[0]):
        swap_metrics(rows, _path(out, f'{name}_metrics'))


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
    s.add_argument('--traj', metavar='JSON[@L1,L2,...]',
                   help='trajectories.json from optsplit.analysis.trajectories: its UMAP replaces the ratio panel')
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
