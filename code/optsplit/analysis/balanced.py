"""Curvature-balanced runs (optsplit/optim/balance.py) against their baseline: losses, steps to reach the baseline's
final validation loss, Jensen-Shannon divergence at equal training loss to the baseline run on the same data order
(against the baseline's data-order floor), and the balancer's trace (per-layer rates, shares, tracked lambda_max of
the balanced operator against its target, Lanczos checks). `summarize` reads the runs (on the cluster), `plot` writes
one multi-page PDF from several summaries, one section each, after a header page.
python -m optsplit.analysis.balanced summarize 'ts10m_balanced/sgd_*' --base 'ts10m_main/sgd_*' --out analysis/balanced/ts10m_sgd.json
python -m optsplit.analysis.balanced plot 'SGD, TinyStories-10M=ts10m_sgd.json' ... --out balanced.pdf"""

import argparse
import json
import textwrap
from itertools import combinations
from pathlib import Path

import numpy as np

HEADER = """Curvature-balanced learning rates

Each layer a (embeddings; transformer blocks 1-n; final norm + head) trains at its own rate eta_a(t) = eta(t) g m_a,
with eta(t) the tuned schedule of the baseline (warmup, then cosine to 10%). The balanced operator is
Hhat = D^1/2 H D^1/2, D = diag(eta_a) for SGD and diag(eta_a) P^-1 for AdamW (P frozen from the current second
moments), with H the Hessian on 32 fixed training sequences. Its linear-stability threshold is 2 + 2 mu = 3.8 for
SGD (heavy ball) and (2 + 2 beta1)/(1 - beta1) = 38 for AdamW. With v the top eigenvector of Hhat, the share of layer a
is s_a = ||v_a||^2 = d log lambda_max / d log m_a (sum_a s_a = 1); equal shares maximize prod_a m_a at fixed
lambda_max. Every 20 steps a warm-started block iteration (3 vectors, 2 Rayleigh-Ritz steps on the vectors and their
residuals) refreshes the top eigenpairs; shares are averaged over the Ritz vectors within 5% of the top, then
smoothed (EMA weight 0.2). A 20-step Lanczos check of lambda_max(Hhat) runs every 500 steps. Multipliers stay at 1
during warmup.

Muon (orthogonalized updates) is balanced on its spectral-norm sharpness instead (Islamov et al.): with c_l = m_a times
Muon's shape scale for each hidden matrix l, S = max <D, H D> over updates with every ||D_l||_2 <= c_l, and the
stability ratio is rho = eta S / (2 sum_l c_l ||grad_l||_*) (edge at 1). S is tracked by 5 warm-started Frank-Wolfe
steps per refresh (checked from scratch every 500 steps); the share of layer a is d log rho / d log m_a =
2 <D_a, (HD)_a> / <D, HD> minus layer a's share of the c-weighted nuclear norm (sum 1). Embeddings and head (AdamW in
Muon) keep the schedule's rate.

Variants
  A  (redistribute):  log m_a -= alpha (s_a - 1/L), recentred to geometric mean 1; g = 1; m_a in [0.1, 10].
  B  (A + scale):     also log g -= gain log(lambda_max(Hhat) / (target x threshold)); every g m_a in [0.1, 10].
                      B<t> targets t x threshold (Muon: rho = t).
  -fe                 embedding kept at the schedule's rate (still in the operator): the 32 sequences see only part of
                      the vocabulary, and SGD B0.9 blew up with the embedding at 10x.
  -fresh              Hessian on 32 fresh training sequences at every refresh instead of the fixed ones.
  plain-lr<x>         unbalanced control at base rate x.
Defaults: alpha = 0.3, gain = 0.2 per refresh. AdamW A-fe tracks 4 vectors with 3 iterations. Batch sharpness is not
measured in these runs (memory).

Comparisons
  Baseline: the main runs (same initialization, same rate, same data orders). Speedup = the baseline's step at its
  best validation loss / the first evaluated step at which the balanced run reaches it (same data order); the best
  validation loss is the final one except on Shakespeare, where the runs overfit.
  JS: Jensen-Shannon divergence of next-token distributions on the fixed validation contexts, at equal training
  loss, between a balanced run and the baseline on the same data order; the grey band is the baseline's floor
  (min to max over pairs of its data orders)."""


def summarize(runs, base):
    from optsplit.analysis.distance import loss_curve
    from optsplit.analysis.split import by_seed

    def curve(a, b):
        c = loss_curve(a, b)
        return dict(loss=c['loss'].tolist(), js=c['js'].tolist())

    def info(r, trace):
        keys = ('step', 'lr', 'lambda_max', 'precond_lambda_max')
        return dict(optimizer=r.optimizer, seed=r.data_seed, cfg=dict(balance=r.cfg.get('balance'),
                    optim=r.cfg.optim[r.optimizer], max_steps=r.cfg.train.max_steps), done=r.done,
                    train=[[x['step'], x['loss']] for x in r.records_of('train')],
                    eval=[[x['step'], x['train_loss'], x['val_loss']] for x in r.records_of('eval')],
                    measure=[{k: x.get(k) for k in keys} for x in r.records_of('measure')],
                    balance=r.records_of('balance') if trace else [])

    out = dict(runs={r.id: info(r, True) for r in runs}, base={r.id: info(r, False) for r in base},
               cross={}, floor={}, own={})
    B = by_seed(base)
    for r in runs:
        if r.data_seed in B:
            out['cross'][r.id] = curve(r, B[r.data_seed])
    for s, t in combinations(sorted(B), 2):
        out['floor'][f'{B[s].id}|{B[t].id}'] = curve(B[s], B[t])
    variants = {}
    for r in runs:
        variants.setdefault(r.id.rsplit('_d', 1)[0], []).append(r)
    for rs in variants.values():
        for a, b in combinations(sorted(rs, key=lambda r: r.data_seed), 2):
            out['own'][f'{a.id}|{b.id}'] = curve(a, b)
    return out


def variant(rid):
    return rid.split('/')[-1].rsplit('_d', 1)[0].split('_', 1)[1]


def first_step(ev, level):
    hit = [s for s, _, v in ev if v <= level]
    return hit[0] if hit else None


def table(d):
    """Rows: (run, last step, final val, its difference to the baseline's, best val, its difference, first step
    reaching the baseline's best val, speedup = the baseline's step at its best / that step). The best (minimum)
    validation loss is the final one on TinyStories; on Shakespeare the runs overfit."""
    base = {r['seed']: r for r in d['base'].values()}
    rows = []
    for rid, r in sorted(d['runs'].items(), key=lambda kv: (kv[1]['seed'], kv[0])):
        b = base.get(r['seed'])
        if not r['eval'] or b is None:
            continue
        fin, best = r['eval'][-1][2], min(v for _, _, v in r['eval'])
        bstep, _, bbest = min(b['eval'], key=lambda e: e[2])
        s = first_step(r['eval'], bbest)
        rows.append((rid.split('/')[-1], r['eval'][-1][0], fin, fin - b['eval'][-1][2], best, best - bbest, s,
                     bstep / s if s else None))
    return rows


def _smooth(train, w=20):
    """(steps, mean of the last `w` logged losses) of [[step, loss], ...]."""
    s, y = np.array(train, float).T
    w = min(w, len(y))
    return s[w - 1:], np.convolve(y, np.ones(w) / w, mode='valid')


def page_losses(pdf, title, d, plt):
    vs = sorted({variant(r) for r in d['runs']})
    col = {v: f'C{i}' for i, v in enumerate(vs)}
    seeds = sorted({r['seed'] for r in d['runs'].values()})
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f'{title}: losses', fontsize=12)
    ax = [fig.add_axes([0.06 + 0.32 * i, 0.47, 0.26, 0.42]) for i in range(3)]
    for rid, b in d['base'].items():
        s, _, v = zip(*b['eval'])
        lw = 1.0 if b['seed'] in seeds else 0.5
        ax[0].plot(s, v, color='k' if b['seed'] in seeds else '0.7', lw=lw, ls='--')
        ax[1].plot(*_smooth(b['train']), color='k' if b['seed'] in seeds else '0.7', lw=lw, ls='--')
    base = {b['seed']: b for b in d['base'].values()}
    for rid, r in d['runs'].items():
        if not r['eval']:
            continue
        v = variant(rid)
        s, _, val = zip(*r['eval'])
        ls = ['-', ':', '-.', (0, (5, 1)), (0, (1, 1))][seeds.index(r['seed']) % 5]
        lab = v if r['seed'] == seeds[0] else None
        ax[0].plot(s, val, color=col[v], lw=1, ls=ls, label=lab)
        ax[1].plot(*_smooth(r['train']), color=col[v], lw=0.8, ls=ls)
        b = base.get(r['seed'])
        if b:
            bs, _, bv = zip(*b['eval'])
            common = sorted(set(s) & set(bs))
            dv = dict(zip(s, val))
            db = dict(zip(bs, bv))
            ax[2].plot(common, [dv[c] - db[c] for c in common], color=col[v], lw=1, ls=ls)
    ax[0].plot([], [], 'k--', label='baseline (same orders)')
    ax[0].plot([], [], '--', color='0.7', label='baseline (other orders)')
    for a, t in zip(ax, ['validation loss', 'training loss (mini-batch, 20-log-step mean)',
                         'validation loss minus baseline (same data order)']):
        a.set(xlabel='step', title=t, xscale='symlog', xlim=(100, None))
    ax[2].axhline(0, color='k', lw=0.6)
    lo = min(min(v for _, _, v in r['eval']) for r in list(d['runs'].values()) + list(d['base'].values()) if r['eval'])
    ax[0].set_ylim(lo - 0.05, lo + 2)
    ax[1].set_ylim(lo - 0.1, lo + 2)
    dv = [abs(x) for a in ax[2].get_lines() for x in a.get_ydata()[len(a.get_ydata()) // 3:]]
    if dv:
        ax[2].set_ylim(-min(0.6, 1.2 * max(dv)), min(0.6, 1.2 * max(dv)))  # blow-ups go off the panel
    ax[0].legend(fontsize=7, ncol=2)
    if len(seeds) > 1:
        ax[1].legend([plt.Line2D([], [], color='k', ls=['-', ':', '-.'][i % 3]) for i in range(len(seeds))],
                     [f'data order {s}' for s in seeds], fontsize=7)
    rows = table(d)
    tab = fig.add_axes([0.06, 0.03, 0.88, 0.34])
    tab.axis('off')
    cells = [[n, str(st), f'{f:.4f}', f'{df:+.4f}', f'{b:.4f}', f'{db:+.4f}', str(s) if s else '-',
              f'{sp:.2f}x' if sp else '-'] for n, st, f, df, b, db, s, sp in rows]
    if cells:
        t = tab.table(cellText=cells, colLabels=['run', 'last step', 'final val', 'minus baseline', 'best val',
                                                 'minus baseline', 'reaches baseline best', 'speedup'],
                      loc='upper center')
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.15)
    pdf.savefig(fig)
    plt.close(fig)


def page_js(pdf, title, d, plt, max_loss=5):
    vs = sorted({variant(r) for r in d['runs']})
    col = {v: f'C{i}' for i, v in enumerate(vs)}
    seeds = sorted({r['seed'] for r in d['runs'].values()})
    fig, axes = plt.subplots(1, len(seeds), figsize=(11, 4.6), squeeze=False, sharey=True)
    fig.suptitle(f'{title}: Jensen-Shannon divergence at equal training loss', fontsize=12)
    floor = [c for c in d['floor'].values() if len(c['loss']) > 1]
    for ax, s in zip(axes[0], seeds):
        if floor:
            grid = np.linspace(max(min(c['loss']) for c in floor), min(max(c['loss']) for c in floor), 200)
            ys = np.array([np.interp(grid, *np.array(sorted(zip(c['loss'], c['js']))).T) for c in floor])
            ax.fill_between(grid, ys.min(0), ys.max(0), color='0.85', label='baseline: two data orders (range)')
            ax.plot(grid, np.median(ys, 0), color='0.5', lw=0.8, ls='--')
        for rid, c in d['cross'].items():
            if d['runs'][rid]['seed'] == s and c['loss']:
                ax.plot(c['loss'], c['js'], color=col[variant(rid)], lw=1, label=f'{variant(rid)} vs baseline')
        for key, c in d['own'].items():
            a = key.split('|')[0]
            if d['runs'][a]['seed'] == s and c['loss']:
                ax.plot(c['loss'], c['js'], color=col[variant(a)], lw=0.8, ls=':',
                        label=f'{variant(a)}: two data orders')
        ax.set(xlabel='training loss (fixed training probe)', title=f'data order {s}', yscale='log', ylim=(1e-3, 1))
        curves = [c['loss'] for c in [*floor, *d['cross'].values()] if c['loss']]
        lo, hi = min(map(min, curves), default=0), max(map(max, curves), default=max_loss)
        ax.set_xlim(min(max_loss, hi + 0.05), lo - 0.05)  # high to low loss
        ax.legend(fontsize=6)
    axes[0, 0].set_ylabel('JS divergence (bits)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_trace(pdf, title, d, plt, per_page=4):
    runs = [(rid, r) for rid, r in sorted(d['runs'].items(), key=lambda kv: (kv[1]['seed'], kv[0])) if r['balance']]
    base = {b['seed']: b for b in d['base'].values()}
    for p in range(0, len(runs), per_page):
        chunk = runs[p:p + per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(11, 8.5), squeeze=False)
        fig.suptitle(f'{title}: balancer trace' + (f' ({p // per_page + 1})' if len(runs) > per_page else ''),
                     fontsize=12)
        for row, (rid, r) in zip(axes, chunk):
            rec = r['balance']
            st = [x['step'] for x in rec if 'rate' in x]
            names = list(rec[-1]['rate']) if 'rate' in rec[-1] else list(rec[0]['rate'])
            cmap = plt.get_cmap('viridis')
            for i, n in enumerate(names):
                c = cmap(i / max(1, len(names) - 1))
                row[0].plot(st, [x['rate'][n] for x in rec if 'rate' in x], color=c, lw=1, label=n)
                row[1].plot(st, [x['ema'][n] for x in rec if 'rate' in x], color=c, lw=1, label=n)
            row[0].set(yscale='log', ylabel=f'{rid.split("/")[-1]}\nrate / schedule (g m_a)')
            row[0].axhline(1, color='k', lw=0.5)
            row[1].axhline(1 / len(names), color='k', lw=0.5, ls='--')
            row[1].set(ylabel='share s_a (EMA)', ylim=(0, None))
            thr = r['cfg']['balance'] and next(x['thr'] for x in rec if 'thr' in x)
            ls, lam = zip(*[(x['step'], x['lam']) for x in rec])
            row[2].plot(ls, np.array(lam) / thr, color='C0', lw=0.6, label=r'tracked $\lambda_{max}(\hat H)$')
            ck = [(x['step'], x['check']) for x in rec if 'check' in x]
            if ck:
                row[2].plot(*zip(*[(s, v / thr) for s, v in ck]), 'o', color='C3', ms=3, label='Lanczos check')
            b = base.get(r['seed'])
            if b:
                key = 'precond_lambda_max' if r['optimizer'] == 'adamw' else 'lambda_max'
                pts = [(x['step'], x[key] * x['lr'] / thr) for x in b['measure'] if x.get(key) and x.get('lr')]
                if pts:
                    row[2].plot(*zip(*pts), color='0.6', lw=0.8, label='baseline (same data order)')
            if r['cfg']['balance']['variant'] == 'B':
                row[2].axhline(r['cfg']['balance']['target'], color='C2', ls='--', lw=1, label='target')
            row[2].axhline(1, color='k', lw=0.6)
            row[2].set(yscale='log', ylabel=r'$\lambda_{max}(\hat H)$ / threshold')
            gs = [x['g'] for x in rec if 'g' in x]
            if r['cfg']['balance']['variant'] == 'B' and gs:
                tw = row[0].twinx()
                tw.plot(st, gs, color='k', lw=1, ls=':')
                tw.set(yscale='log', ylabel='g (dotted)')
            for a in row:
                a.set_xscale('symlog', linthresh=100)
                a.set_xlim(0, None)
        axes[0, 0].legend(fontsize=6, ncol=2)
        axes[0, 2].legend(fontsize=6)
        for a in axes[-1]:
            a.set_xlabel('step')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot(sections, path, header=HEADER):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        text = '\n'.join(textwrap.fill(line, 130, subsequent_indent=' ' * (len(line) - len(line.lstrip()) + 4))
                         if len(line) > 130 else line for line in header.splitlines())
        fig.text(0.05, 0.95, text, va='top', ha='left', family='monospace', fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)
        for title, d in sections:
            page_losses(pdf, title, d, plt)
            page_js(pdf, title, d, plt)
            page_trace(pdf, title, d, plt)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('runs', nargs='+', help='glob patterns of the balanced runs (and of unbalanced controls)')
    s.add_argument('--base', nargs='+', required=True, help='glob patterns of the baseline runs')
    s.add_argument('--out', required=True)
    q = sub.add_parser('plot')
    q.add_argument('sections', nargs='+', metavar='TITLE=JSON')
    q.add_argument('--header', help='text file replacing the header page')
    q.add_argument('--out', required=True)
    args = p.parse_args()
    if args.cmd == 'summarize':
        from optsplit.analysis.io import load_runs
        runs = load_runs(args.runs)  # balanced runs and controls (unbalanced, e.g. another rate)
        base = [r for r in load_runs(args.base) if not r.cfg.fork.parent]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summarize(runs, base)))
        print(f'wrote {out}')
    else:
        sections = [(t, json.loads(Path(f).read_text())) for t, f in (a.split('=', 1) for a in args.sections)]
        plot(sections, args.out, Path(args.header).read_text() if args.header else HEADER)


if __name__ == '__main__':
    main()
