"""Which whole-model metrics separate the optimizers, ranked as the functions are: at equal training loss, the
difference between two optimizers on the same data order over the difference between two data orders of one
optimizer. Metrics come from the checkpoints (metrics/trajectory.py), the stored predictions and the measurements
logged during training. With --forks, also where the forks of a swap end in each metric, between pure A and pure B.

    python -m optsplit.analysis.rank_metrics 'ts10m_traj/*' [--forks 'ts10m_forks_ckpt/*'] --out <dir>

Writes <dir>/metrics.json (curves, ratios, ranking), metrics.txt (ranked table) and figures/metrics.pdf.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from optsplit.analysis.distance import js_divergence
from optsplit.analysis.io import load_runs
from optsplit.analysis.ngram import ngram_logp

# key: (group, label); the order is that of the figure pages
METRICS = {
    'l2': ('weights', r'$\|\theta\|_2$'),
    'dist_init_rel': ('weights', r'$\|\theta-\theta_0\|/\|\theta_0\|$'),
    'linf_over_l2': ('weights', r'$\|\theta\|_\infty/\|\theta\|_2$'),
    'stable_rank': ('weights', 'stable rank of W'),
    'eff_rank': ('weights', 'effective rank of W'),
    'spectral': ('weights', r'spectral norm of W'),
    'sv_spread': ('weights', r'$\sigma_1/\sigma_{\rm median}$ of W'),
    'update_rel': ('updates', r'$\|\Delta\theta\|/\|\theta\|$'),
    'update_stable_rank': ('updates', r'stable rank of $\Delta W$'),
    'update_eff_rank': ('updates', r'effective rank of $\Delta W$'),
    'update_align': ('updates', r'$\cos(\Delta\theta_t,\Delta\theta_{t+1})$'),
    'update_grad_cos': ('updates', r'$\cos(\Delta\theta_t,-g_t)$'),
    'lambda_max': ('curvature', r'$\lambda_{\max}(H)$'),
    'batch_sharpness': ('curvature', 'batch sharpness'),
    'hess_trace': ('curvature', r'tr $H$'),
    'trace_over_lmax': ('curvature', r'tr $H/\lambda_{\max}$'),
    'edge_fraction': ('curvature', 'sharpness / threshold'),
    'entropy': ('function', 'prediction entropy'),
    'confidence': ('function', 'mean top-1 probability'),
    'ece': ('function', 'calibration error (ECE)'),
    'acc': ('function', 'top-1 accuracy'),
    'loss_freq': ('function', 'loss, frequent targets'),
    'loss_rare': ('function', 'loss, rare targets'),
    'rare_minus_freq': ('function', 'loss rare - frequent'),
    'js_unigram': ('function', 'JS to unigram'),
    'js_bigram': ('function', 'JS to bigram'),
    'gen_gap': ('function', 'val - train loss'),
    'attn_entropy': ('internals', 'attention entropy'),
    'attn_sink': ('internals', 'attention on token 1'),
    'mlp_active': ('internals', 'active MLP units'),
    'resid_rms': ('internals', 'residual RMS'),
    'feature_erank': ('internals', 'eff. rank of features'),
}
COLORS = {'adamw': 'C0', 'muon': 'C1', 'sgd': 'C2'}
NAMES = {'adamw': 'AdamW', 'muon': 'Muon', 'sgd': 'SGD'}
CKPT_SKIP = {'step', 'train_loss'}


def ece(conf, correct, bins=15):
    idx = (conf * bins).long().clamp(max=bins - 1)
    out = 0.
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.float().mean().item() * abs(correct[m].float().mean().item() - conf[m].mean().item())
    return out


class Refs:
    """n-gram references and the frequent / rare split of the probe targets (terciles of training frequency)."""

    def __init__(self, run):
        self.lu, self.lb = ngram_logp(run.cfg)
        f = self.lu[0][run.targets]
        lo, hi = f.quantile(1 / 3), f.quantile(2 / 3)
        self.rare, self.freq = f <= lo, f >= hi


def pred_metrics(logp, y, refs):
    logp = logp.float()
    p = logp.exp()
    nll = -logp.gather(1, y[:, None])[:, 0]
    conf, pred = p.max(1)
    return dict(entropy=-(p * logp).sum(1).mean().item(), confidence=conf.mean().item(),
                ece=ece(conf, pred == y), acc=(pred == y).float().mean().item(),
                loss_freq=nll[refs.freq].mean().item(), loss_rare=nll[refs.rare].mean().item(),
                rare_minus_freq=(nll[refs.rare].mean() - nll[refs.freq].mean()).item(),
                js_unigram=js_divergence(logp, refs.lu), js_bigram=js_divergence(logp, refs.lb))


def run_points(run, refs, cache):
    """{metric: (loss, value) arrays}, training loss on the training probe."""
    pts = {}

    def add(key, loss, v):
        if v is not None and np.isfinite(v):
            pts.setdefault(key, []).append((loss, v))

    ev = {r['step']: r for r in run.records_of('eval')}
    for r in ev.values():
        add('gen_gap', r['train_loss'], r['val_loss'] - r['train_loss'])
    lr_beta = run.cfg.optim.adamw.betas[0] if run.optimizer == 'adamw' else None
    for r in run.records_of('measure'):
        if r['step'] not in ev:
            continue
        loss = ev[r['step']]['train_loss']
        add('lambda_max', loss, r.get('lambda_max'))
        add('batch_sharpness', loss, r.get('batch_sharpness'))
        if run.optimizer == 'sgd' and r.get('lambda_max') is not None:
            add('edge_fraction', loss, r['lambda_max'] * r['lr'] / (2 + 2 * run.cfg.optim.sgd.momentum))
        if run.optimizer == 'adamw' and r.get('precond_lambda_max') is not None:
            add('edge_fraction', loss, r['precond_lambda_max'] * r['lr'] * (1 - lr_beta) / (2 + 2 * lr_beta))
    key = f'{run.id}:{run.path.joinpath("preds", "targets.pt").stat().st_mtime}'
    if key not in cache:
        cache[key] = {s: pred_metrics(run.logp(s), run.targets, refs) for s in run.pred_steps if s in ev}
    for s, m in cache[key].items():
        for k, v in m.items():
            add(k, ev[s]['train_loss'], v)
    for f in sorted((run.path / 'traj_metrics').glob('*.json')):
        m = json.loads(f.read_text())
        for k, v in m.items():  # lambda_max: the denser measurement during training is ranked
            if k not in CKPT_SKIP:
                add('lambda_max_ckpt' if k == 'lambda_max' else k, m['train_loss'], v)
        add('trace_over_lmax', m['train_loss'], m['hess_trace'] / m['lambda_max'])
    return {k: np.array(v) for k, v in pts.items()}


def monotone(pts):
    """Points in training order with the loss made monotone by a running minimum; the earliest point of each loss."""
    loss = np.minimum.accumulate(pts[:, 0])
    keep = np.r_[True, np.diff(loss) < 0]
    return loss[keep], pts[keep, 1]


def at_loss(pts, levels):
    if pts is None or len(pts) < 2:
        return np.full(len(levels), np.nan)
    loss, v = monotone(pts)
    out = np.interp(levels, loss[::-1], v[::-1], left=np.nan, right=np.nan)
    return out


def mean_abs_pairs(vals):
    """Mean |a - b| over pairs of rows (data orders), per column (loss level)."""
    return np.nanmean([np.abs(a - b) for a, b in combinations(vals, 2)], axis=0)


def rank(runs, refs, levels, cache):
    opts = [o for o in ('adamw', 'muon', 'sgd') if any(r.optimizer == o for r in runs)]
    seeds = sorted({r.data_seed for r in runs})
    by = {(r.optimizer, r.data_seed): r for r in runs}
    pts = {r.id: run_points(r, refs, cache) for r in runs}
    res = {}
    for k in METRICS:
        v = {(o, d): at_loss(pts[by[o, d].id].get(k), levels) for o in opts for d in seeds if (o, d) in by}
        floor = {o: mean_abs_pairs([v[o, d] for d in seeds]) for o in opts}
        mean = {o: np.nanmean([v[o, d] for d in seeds], axis=0) for o in opts}
        pairs = {}
        for a, b in combinations(opts, 2):
            cross = np.nanmean([np.abs(v[a, d] - v[b, d]) for d in seeds], axis=0)
            pairs[f'{a}-{b}'] = dict(cross=cross, floor=np.fmax(floor[a], floor[b]),
                                     ratio=cross / np.fmax(floor[a], floor[b]))
        res[k] = dict(values=v, mean=mean, floor=floor, pairs=pairs)
    return res, pts


def summarize(res, levels, lo, hi):
    """Median ratio over the loss levels in [lo, hi] per pair; per optimizer, the signed deviation from the other two
    (difference of means over the largest floor), median over the same levels."""
    sel = (levels >= lo) & (levels <= hi)
    out = {}
    for k, r in res.items():
        s = {p: float(np.nanmedian(d['ratio'][sel])) if np.isfinite(d['ratio'][sel]).any() else math.nan
             for p, d in r['pairs'].items()}
        fl = np.nanmax([r['floor'][o] for o in r['floor']], axis=0)
        dev = {}
        for o in r['mean']:
            others = [r['mean'][q] for q in r['mean'] if q != o]
            d = (r['mean'][o] - np.nanmean(others, axis=0)) / fl
            dev[o] = float(np.nanmedian(d[sel])) if np.isfinite(d[sel]).any() else math.nan
        out[k] = dict(ratio=s, deviation=dev)
    return out


def loss_axis(ax):
    ax.set_xscale('function', functions=(lambda x: np.log(np.maximum(x - 1, 1e-9)), lambda y: np.exp(y) + 1))
    ax.set_xticks([4, 3, 2.5, 2, 1.8, 1.6, 1.4])
    ax.set_xticklabels(['4', '3', '2.5', '2', '1.8', '1.6', '1.4'])
    ax.invert_xaxis()


def plot_curves(pdf, runs, pts, keys, max_loss):
    n = len(keys)
    cols = 4
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(13, 2.6 * math.ceil(n / cols)), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        for r in runs:
            p = pts[r.id].get(k)
            if p is None:
                continue
            loss, v = monotone(p)
            m = loss <= max_loss
            ax.plot(loss[m], v[m], color=COLORS[r.optimizer], lw=0.9,
                    ls=['-', '--', ':'][r.data_seed % 3], label=f'{NAMES[r.optimizer]} d{r.data_seed}')
        loss_axis(ax)
        ax.set_title(METRICS[k][1], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=6, ncol=3)
    fig.supxlabel('training loss (log scale in L - 1)', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_ranking(pdf, summary, order, title):
    pairs = list(next(iter(summary.values()))['ratio'])
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(order) + 1))
    y = np.arange(len(order))
    for i, p in enumerate(pairs):
        ax.barh(y + (i - 1) * 0.27, [summary[k]['ratio'][p] for k in order], height=0.27,
                label=' - '.join(NAMES[o] for o in p.split('-')))
    ax.set_yticks(y)
    ax.set_yticklabels([METRICS[k][1] for k in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.axvline(1, color='k', lw=0.6)
    ax.set_xlabel('between-optimizer difference / data-order difference (median over loss levels)')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def final_value(pts_run, k):
    """Value of metric k at the end of a run: last checkpoint metric (for the updates, that of the triple 9998-10000)
    or last prediction. lambda_max from the final checkpoint, as the forks were rerun without measurements."""
    p = pts_run.get('lambda_max_ckpt' if k == 'lambda_max' else k)
    return float(p[-1, 1]) if p is not None and len(p) else math.nan


def swaps(forks, pures, refs, cache, keys):
    """For each AdamW<->Muon fork, final metric shift (F - A) / (B - A) and distances to A and B in floors."""
    pts = {r.id: run_points(r, refs, cache) for r in pures + forks}
    finals = {r.id: {k: final_value(pts[r.id], k) for k in keys} for r in pures + forks}
    floor = {}
    for o in {r.optimizer for r in pures}:
        rs = [r for r in pures if r.optimizer == o]
        floor[o] = {k: float(np.nanmean([abs(finals[a.id][k] - finals[b.id][k]) for a, b in combinations(rs, 2)]))
                    for k in keys}
    pure0 = {r.optimizer: r for r in pures if r.data_seed == 0}
    rows = []
    for f in sorted(forks, key=lambda f: (f.cfg.fork.parent, f.fork_step)):
        a, b = f.cfg.fork.parent.split('/')[-1].split('_')[0], f.optimizer
        row = dict(fork=f.id, opt_a=a, opt_b=b, fork_step=f.fork_step, metrics={})
        for k in keys:
            va, vb, vf = finals[pure0[a].id][k], finals[pure0[b].id][k], finals[f.id][k]
            fl = max(floor[a][k], floor[b][k])
            row['metrics'][k] = dict(value=vf, pure_a=va, pure_b=vb, shift=(vf - va) / (vb - va),
                                     gap_over_floor=abs(vb - va) / fl, to_a=abs(vf - va) / floor[a][k],
                                     to_b=abs(vf - vb) / floor[b][k])
        rows.append(row)
    return rows, floor


def plot_swaps(pdf, rows, keys):
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(keys) / cols), cols, figsize=(13, 2.5 * math.ceil(len(keys) / cols)),
                             squeeze=False)
    for ax, k in zip(axes.flat, keys):
        for (a, b), c in ((('adamw', 'muon'), 'C1'), (('muon', 'adamw'), 'C0')):
            rr = [r for r in rows if r['opt_a'] == a and r['opt_b'] == b]
            y = np.array([r['metrics'][k]['shift'] for r in rr], dtype=float)
            ax.plot([r['fork_step'] for r in rr], np.where(np.isfinite(y), np.clip(y, -1, 2), np.nan), 'o-', color=c,
                    ms=3, label=f'{NAMES[a]} to {NAMES[b]}')
        g = rows[0]['metrics'][k]['gap_over_floor']
        ax.set_title(f'{METRICS[k][1]} (gap {g:.1f} floors)', fontsize=8)
        ax.set_ylim(-1.1, 2.1)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(1, color='k', lw=0.5, ls='--')
        ax.set_xscale('log')
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(keys):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=6)
    fig.supxlabel('fork step', fontsize=9)
    fig.supylabel('shift: 0 = parent (pure A), 1 = new optimizer (pure B)', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--forks', nargs='*', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--levels', type=int, default=40, help='loss levels, evenly spaced in log(L - 1)')
    p.add_argument('--max-loss', type=float, default=4.0)
    p.add_argument('--lo-hi', type=float, nargs=2, default=None, help='loss range of the summary (default: from '
                   'the lowest loss all runs reach to 3)')
    p.add_argument('--report', type=float, nargs='+', default=[3.0, 2.2, 1.8], help='loss levels for the table')
    args = p.parse_args()
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    refs = Refs(runs[0])
    cache_file = out / 'pred_metrics_cache.pt'
    cache = torch.load(cache_file) if cache_file.exists() else {}
    floor_loss = max(r.series('eval', 'train_loss')[1].min() for r in runs)
    levels = 1 + np.exp(np.linspace(np.log(args.max_loss - 1), np.log(floor_loss - 1 + 1e-3), args.levels))
    res, pts = rank(runs, refs, levels, cache)
    lo, hi = args.lo_hi or (floor_loss, 3.0)
    summary = summarize(res, levels, lo, hi)
    order = sorted(summary, key=lambda k: -np.nan_to_num(summary[k]['ratio'].get('adamw-muon', np.nan), nan=-1))
    rep = {L: int(np.argmin(np.abs(levels - L))) for L in args.report}

    lines = [f'loss levels {levels[0]:.2f}..{levels[-1]:.2f}; summary = median ratio over losses {lo:.2f}..{hi:.2f}',
             'metric | ratio A-M, A-S, M-S | ' + ' | '.join(f'L={levels[i]:.2f}: AdamW / Muon / SGD (A-M ratio)'
                                                            for i in rep.values())]
    for k in order:
        r, s = res[k], summary[k]['ratio']
        cells = []
        for i in rep.values():
            m = [r['mean'][o][i] if o in r['mean'] else np.nan for o in ('adamw', 'muon', 'sgd')]
            cells.append(' / '.join(f'{x:.4g}' for x in m) + f" ({r['pairs']['adamw-muon']['ratio'][i]:.2f})")
        lines.append(f"{k} | {s.get('adamw-muon', np.nan):.2f}, {s.get('adamw-sgd', np.nan):.2f}, "
                     f"{s.get('muon-sgd', np.nan):.2f} | " + ' | '.join(cells))
    lines.append('')
    for o in ('adamw', 'muon', 'sgd'):
        dev = sorted(((summary[k]['deviation'].get(o, np.nan), k) for k in summary), key=lambda t: -np.nan_to_num(t[0]))
        dev = [(d, k) for d, k in dev if np.isfinite(d)]
        lines.append(f"{o} up: " + ', '.join(f'{k} {d:+.1f}' for d, k in dev[:5]))
        lines.append(f"{o} down: " + ', '.join(f'{k} {d:+.1f}' for d, k in dev[::-1][:5]))
    (out / 'metrics.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    js = dict(levels=levels.tolist(), lo=lo, hi=hi, summary=summary, order=order,
              metrics={k: dict(group=METRICS[k][0], label=METRICS[k][1],
                               mean={o: v.tolist() for o, v in r['mean'].items()},
                               floor={o: v.tolist() for o, v in r['floor'].items()},
                               pairs={p: {q: x.tolist() for q, x in d.items()} for p, d in r['pairs'].items()},
                               points={rid: pts[rid][k].tolist() for rid in pts if k in pts[rid]})
                       for k, r in res.items()})
    with PdfPages(out / 'figures' / 'metrics.pdf') as pdf:
        plot_ranking(pdf, summary, order, f'TinyStories-10M, losses {lo:.2f}-{hi:.2f}')
        for g in dict.fromkeys(v[0] for v in METRICS.values()):
            plot_curves(pdf, runs, pts, [k for k in METRICS if METRICS[k][0] == g], args.max_loss)
        if args.forks:
            forks = load_runs(args.forks)
            pures = [r for r in runs if r.optimizer in ('adamw', 'muon')]
            rows, floor = swaps(forks, pures, refs, cache, list(METRICS))
            js['swaps'] = dict(rows=rows, floor=floor)
            plot_swaps(pdf, rows, [k for k in order if k != 'edge_fraction'])
    (out / 'metrics.json').write_text(json.dumps(js, default=lambda x: None if x is None else float(x)))
    torch.save(cache, cache_file)
    print(f'wrote {out}/metrics.json, metrics.txt, figures/metrics.pdf')


if __name__ == '__main__':
    main()
