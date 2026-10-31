"""Stochastic edge of stability: ratios R = d^T H d / (-2 g^T d) from the `step_sharpness` records of
optsplit/measure/step_sharpness.py, a table of their plateaus and the figures (one multi-page PDF).

    python -m optsplit.analysis.stochastic_eos 'ts10m_stoch_eos/*' [--out stochastic_eos.pdf]
"""

import argparse

import numpy as np

from optsplit.analysis.io import load_runs


def _quad(M, names):
    """d^T H d for d = sum of the directions `names`, from the upper-triangle record `M`."""
    tot = 0.0
    for i, a in enumerate(names):
        for b in names[i:]:
            v = M.get(f'{a},{b}', M.get(f'{b},{a}'))
            tot += v if a == b else 2 * v
    return tot


def terms(rec, H='B', g='B'):
    """{quantity: (numerator d^T H d, denominator -2 g^T d)}, with the Hessian of source `H` and the gradient of
    source `g` ('B': the step's batch, 'pop': fresh sequences). R = numerator / denominator; the step lowers
    the loss to second order iff R < 1. Quantities: `s` (the step without weight decay), `s.<part>`, `s+wd`,
    `nm` (the step without momentum), `nm.<part>`, `g` (along the batch gradient, as d^T H d / -2 g^T d with
    d = -g: multiply by lr to get Andreyev's lr * batch sharpness / 2)."""
    M, gd = rec[H]['dHd'], rec[g]['gd']
    parts = [k[2:] for k in rec['d2'] if k.startswith('s.')]
    out = {}

    def add(name, dirs, sign=1.0):
        out[name] = (_quad(M, dirs), -2 * sign * sum(gd[d] for d in dirs))

    add('s', [f's.{q}' for q in parts])
    add('nm', [f'nm.{q}' for q in parts])
    if len(parts) > 1:
        for q in parts:
            add(f's.{q}', [f's.{q}'])
            add(f'nm.{q}', [f'nm.{q}'])
    if 'wd' in gd:
        add('s+wd', [f's.{q}' for q in parts] + ['wd'])
    add('g', ['g'], sign=-1.0)
    return out


def ratios(rec, H='B', g='B'):
    return {k: n / d for k, (n, d) in terms(rec, H, g).items()}


def load(run, H='B', g='B'):
    """(steps, lr of the main part, {quantity: (numerators, denominators)}) over the records with H and g.
    With H == g, also `exact`: the same ratio from the measured loss change dL of the full step S = s + wd,
    2 (dL - g^T S) / (-2 g^T S), which equals S^T H S / (-2 g^T S) for a quadratic loss."""
    recs = [r for r in run.records_of('step_sharpness') if H in r and g in r]
    steps = np.array([r['step'] for r in recs])
    lr = np.array([r['lr'].get('muon', r['lr'].get('all')) for r in recs])
    t = [terms(r, H, g) for r in recs]
    out = {k: (np.array([x[k][0] for x in t]), np.array([x[k][1] for x in t])) for k in (t[0] if t else {})}
    if 'g' in out:  # along -lr g: lr g^T H g / 2 ||g||^2 (Andreyev's lr * batch sharpness / 2)
        out['g'] = (out['g'][0] * lr, out['g'][1])
    if H == g and recs:
        gS = np.array([sum(v for k, v in r[g]['gd'].items() if k.startswith('s.') or k == 'wd') for r in recs])
        dL = np.array([r[g]['loss1'] - r[g]['loss0'] for r in recs])
        out['exact'] = (2 * (dL - gS), -2 * gS)
    return steps, lr, out


def smooth(steps, num, den, half):
    """Ratio of the sums of numerators and denominators within +-`half` steps."""
    return np.array([num[m].sum() / den[m].sum() for m in (np.abs(steps - t) <= half for t in steps)])


def plateau(steps, num, den, start, blocks=6):
    """Median over `blocks` consecutive blocks of steps >= start of the ratio of sums in each block, and half
    their interquartile range (robust to a single loss spike)."""
    idx = np.where(steps >= start)[0]
    vals = [num[b].sum() / den[b].sum() for b in np.array_split(idx, blocks)]
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    return float(q2), float((q3 - q1) / 2)


def lam_ratio(run):
    """(steps, eta lambda_max / threshold) on the 32 fixed sequences (AdamW: preconditioned), or None (Muon)."""
    from optsplit.analysis.eos import ratio
    return ratio(run)


def summarize(run, start_frac=0.5):
    """Plateaus over steps >= start_frac * max_steps, as `plateau` (median over blocks, half IQR); `rho` and
    `kappa` split the own-batch ratio as B.s = pop.s * kappa * rho."""
    start = start_frac * run.cfg.train.max_steps
    out = {}
    for src, (H, g) in (('B', ('B', 'B')), ('pop', ('pop', 'pop')), ('mix', ('B', 'pop'))):
        steps, _, q = load(run, H, g)
        if (steps >= start).sum() >= 5:
            out.update({f'{src}.{k}': plateau(steps, n, d, start) for k, (n, d) in q.items()})
    bs, _, b = load(run, 'B', 'B')
    ps, _, p = load(run, 'pop', 'pop')
    common = np.intersect1d(bs[bs >= start], ps)
    if len(common) >= 6:
        i, j = np.isin(bs, common), np.isin(ps, common)
        out['rho'] = plateau(common, p['s'][1][j], b['s'][1][i], start)  # g_pop^T s / g_B^T s
        out['kappa'] = plateau(common, b['s'][0][i], p['s'][0][j], start)  # s^T H_B s / s^T H_pop s
    lam = lam_ratio(run)
    if lam is not None and (lam[0] >= start).sum():
        v = lam[1][lam[0] >= start]
        out['lam'] = (float(np.median(v)), float(np.std(v)))
    return out


def table(runs, start_frac=0.5):
    cols = ['B.s', 'B.exact', 'B.nm', 'B.s.muon', 'B.s.adam', 'B.s+wd', 'pop.s', 'pop.exact', 'pop.nm', 'mix.s',
            'rho', 'kappa', 'lam']
    lines = [f'{"run":20s}' + ''.join(f'{c:>12s}' for c in cols)]
    for run in runs:
        s = summarize(run, start_frac)
        cells = ['' if s.get(c) is None else f'{s[c][0]:.3f}±{s[c][1]:.2f}' if isinstance(s[c], tuple)
                 else f'{s[c]:.3f}' for c in cols]
        lines.append(f'{run.id.split("/")[-1]:20s}' + ''.join(f'{x:>12s}' for x in cells))
    return '\n'.join(lines)


COLORS = {'s': '#2a78d6', 'nm': '#eb6834', 'pop.s': '#1baf7a', 'pop.nm': '#eda100', 'lam': '#008300',
          'part1': '#6250d6', 'part2': '#e87ba4'}
NAMES = {'sgd': 'SGD (momentum 0.9)', 'adamw': 'AdamW', 'muon': 'Muon'}
NM = {'sgd': r'$-\eta g_B$ ($\eta\,$batch sharpness$/2$)', 'adamw': r'$-\eta D^{-1} g_B$',
      'muon': r'$-\eta\,\mathrm{polar}(g_B)$ (+ AdamW part)'}
YLIM = (0.01, 30)


def _line(ax, x, y, **kw):
    ax.plot(x, np.where(y > 0, y, np.nan), **kw)


def run_page(pdf, run, half=100):
    import matplotlib.pyplot as plt
    o = run.optimizer
    cosine = run.cfg.schedule.decay == 'cosine'
    fig, axes = plt.subplots(3 if cosine else 2, 1, figsize=(8.5, 9.5 if cosine else 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1] if cosine else [3, 3]})
    steps, lr, q = load(run)
    ax = axes[0]
    _line(ax, steps, smooth(steps, *q['s'], half), color=COLORS['s'], lw=2, label=r'actual step $s$')
    _line(ax, steps, smooth(steps, *q['exact'], half), color=COLORS['s'], lw=1, ls=':',
          label='actual step, from the exact loss change')
    _line(ax, steps, smooth(steps, *q['nm'], half), color=COLORS['nm'], lw=1.2, label=f'no momentum: {NM[o]}')
    names = {'s.muon': 'actual step, Muon matrices only', 's.adam': 'actual step, AdamW part only',
             's+wd': 'actual step with weight decay'}
    for k, c in zip([k for k in names if k in q], (COLORS['part1'], COLORS['part2'])):
        _line(ax, steps, smooth(steps, *q[k], half), color=c, lw=1, label=names[k])
    ax.set_ylabel(r'$R_B = s^\top H_B s\,/\,(-2 g_B^\top s)$')
    ax.set_title(f'{run.id.split("/")[-1]}: on the step\'s own batch (ratio of sums over $\\pm${half} steps)',
                 fontsize=10)
    ax = axes[1]
    ps, _, pq = load(run, 'pop', 'pop')
    if len(ps):
        w = 2.5 * half
        _line(ax, ps, smooth(ps, *pq['s'], w), color=COLORS['pop.s'], lw=2, label=r'actual step $s$')
        _line(ax, ps, smooth(ps, *pq['exact'], w), color=COLORS['pop.s'], lw=1, ls=':',
              label='actual step, from the exact loss change')
        _line(ax, ps, smooth(ps, *pq['nm'], w), color=COLORS['pop.nm'], lw=1.2, label=f'no momentum: {NM[o]}')
    r = lam_ratio(run)
    if r is not None:
        ax.plot(*r, 's', ms=4, color=COLORS['lam'], label=r'$\eta\lambda_{\max}$ / threshold, 32 fixed sequences'
                + (r' ($P^{-1}H$)' if o == 'adamw' else ''))
    ax.set_ylabel(r'$\bar R = s^\top \bar H s\,/\,(-2 \bar g^\top s)$')
    ax.set_title(f'on {run.cfg.measure.step_sharpness.pop_seqs} fresh sequences (population; '
                 f'$\\pm${int(2.5 * half)} steps)', fontsize=10)
    if cosine:
        axes[2].plot(steps, lr / lr.max(), color='0.3', lw=1)
        axes[2].set_ylabel(r'$\eta(t)/\eta_{\max}$')
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(alpha=0.2)
    axes[-1].set_xlabel('step')
    for ax in axes[:2]:
        ax.axhline(1, color='0.3', ls='--', lw=1)
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.grid(alpha=0.2, which='both')
        ax.legend(fontsize=7, loc='best', ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, runs, start_frac=0.5):
    """Plateau against the rate, one panel per optimizer, constant-rate runs; batch-size variants open."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    series = [('B.s', COLORS['s'], 'o', r'own batch $R_B$, actual step'),
              ('B.nm', COLORS['nm'], 'o', 'own batch, no momentum'),
              ('pop.s', COLORS['pop.s'], 'D', r'population $\bar R$, actual step'),
              ('pop.exact', COLORS['pop.s'], 'x', 'population, exact loss change'),
              ('lam', COLORS['lam'], 's', r'$\eta\lambda_{\max}$/threshold')]
    for ax, o in zip(axes, ('sgd', 'adamw', 'muon')):
        mine = [r for r in runs if r.optimizer == o and r.cfg.schedule.decay == 'constant']
        for r in mine:
            s, bs = summarize(r, start_frac), r.cfg.train.batch_size
            for j, (k, c, mk, lab) in enumerate(series):
                if k not in s:
                    continue
                x = r.lr * 2 ** (0.06 * (j - 2) + (0.3 if bs != 128 else 0))
                ax.errorbar(x, s[k][0], s[k][1], fmt=mk, color=c, ms=6, mfc=c if bs == 128 else 'white',
                            label=lab if r is mine[0] else None)
                if bs != 128 and j == 0:
                    ax.annotate(f'B={bs}', (x, s[k][0]), textcoords='offset points', xytext=(4, 4), fontsize=7)
        ax.axhline(1, color='0.3', ls='--', lw=1)
        ax.set_xscale('log')
        lrs = sorted({r.lr for r in mine})
        ax.set_xticks(lrs, [f'{v:g}' for v in lrs])
        ax.minorticks_off()
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.set_title(NAMES[o], fontsize=10)
        ax.set_xlabel(r'$\eta$' + (' (Muon matrices; AdamW part scaled with it)' if o == 'muon' else ''), fontsize=8)
        ax.grid(alpha=0.2, which='both')
    axes[0].set_ylabel(f'plateau over steps $\\geq$ {start_frac:g} T (median of 6 blocks, bars: half IQR)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=8)
    fig.suptitle('Constant rate, 3000 steps: filled = batch 128 at x1/2, x1, x2 the main rate; '
                 'open, shifted right = batch 32 at the main rate', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def coherence_page(pdf, runs, half=250):
    """Own batch against population at the main rate: R_B = R_pop * kappa * rho, at the steps with both."""
    import matplotlib.pyplot as plt
    main = {'sgd': 0.03, 'adamw': 2e-3, 'muon': 0.01}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, o in zip(axes, ('sgd', 'adamw', 'muon')):
        r = [r for r in runs if r.optimizer == o and r.cfg.schedule.decay == 'constant' and r.lr == main[o]
             and r.cfg.train.batch_size == 128]
        if not r:
            continue
        bs, _, b = load(r[0], 'B', 'B')
        ps, _, p = load(r[0], 'pop', 'pop')
        i, j = np.isin(bs, ps), np.isin(ps, bs)
        st = ps[j]
        for y, c, lab, lw in (
                (smooth(st, b['s'][0][i], b['s'][1][i], half), COLORS['s'], r'own batch $R_B$', 2),
                (smooth(st, p['s'][0][j], p['s'][1][j], half), COLORS['pop.s'], r'population $\bar R$', 2),
                (smooth(st, p['s'][1][j], b['s'][1][i], half), COLORS['part1'],
                 r'coherence $\rho = \bar g^\top s / g_B^\top s$', 1.2),
                (smooth(st, b['s'][0][i], p['s'][0][j], half), COLORS['part2'],
                 r'$\kappa = s^\top H_B s / s^\top \bar H s$', 1.2)):
            _line(ax, st, y, color=c, lw=lw, label=lab)
        ax.axhline(1, color='0.3', ls='--', lw=1)
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.set_title(f'{NAMES[o]}, $\\eta$ = {main[o]:g}, constant', fontsize=10)
        ax.set_xlabel('step')
        ax.grid(alpha=0.2, which='both')
    axes[0].set_ylabel(f'ratio of sums over $\\pm${half} steps')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(r'Why the own-batch ratio is not 1: $R_B = \bar R\,\kappa\,\rho$ exactly; $\bar R \approx 1$, '
                 r'$\kappa \approx 1$, so $R_B \approx \rho$', fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def text_page(pdf, text, size=8):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.96, text, va='top', ha='left', family='monospace', fontsize=size)
    pdf.savefig(fig)
    plt.close(fig)


def _kind(run):
    return 'cosine' if run.cfg.schedule.decay == 'cosine' else 'bs32' if run.cfg.train.batch_size < 128 else 'const'


def header(runs):
    """The first page: what is measured, the runs, and the plateaus by optimizer (ranges over the runs)."""
    S = {r.id: summarize(r) for r in runs}

    def rng(k, kind='const'):
        out = []
        for o in ('sgd', 'adamw', 'muon'):
            v = [S[r.id][k] for r in runs if r.optimizer == o and _kind(r) == kind and k in S[r.id]]
            v = [x[0] if isinstance(x, tuple) else x for x in v]
            out.append(f'{NAMES[o].split()[0]} ' + ('-' if not v else f'{v[0]:.2f}' if len(v) == 1
                                                   else f'{min(v):.2f}-{max(v):.2f}'))
        return ', '.join(out)

    return f"""Stochastic edge of stability on TinyStories-10M (suite ts10m_stoch_eos, initialization and data order d0)

Does the curvature of each step's own minibatch along the update actually taken settle at the stability threshold?
With s the change of the parameters in one step (without decoupled weight decay) and g_B, H_B the gradient and
Hessian of the batch of that step, the step lowers the batch loss (to second order) iff
    R_B = s^T H_B s / (-2 g_B^T s) < 1          (s = -eta u:  R_B = eta Q / 2,  Q = u^T H_B u / g_B^T u).
Hessian-vector products on the step's batch every 10 steps (every step early on), and on 256 fresh training sequences
every 50 steps (population: R_pop = s^T H s / (-2 g^T s)). Also: the ratio from the exact one-step loss change, the
step without momentum (SGD: -eta g_B, Andreyev's eta * batch sharpness / 2; AdamW: -eta D^-1 g_B; Muon: -eta polar(g_B)
on its matrices), Muon's two parts, the step with weight decay, and eta lambda_max / threshold (Lanczos, 32 fixed seqs).
Runs: constant rate after 200 warmup steps, 3000 steps, batch 128, at x1/2, x1, x2 the main rate (SGD 0.03, momentum
0.9; AdamW 2e-3; Muon 0.01 with its AdamW part 2e-3, scaled together); batch 32 at the main rate; the main rate with
warmup and cosine decay over 10k steps. Plateau: median over 6 blocks of the second half (ratio of sums per block).

Plateaus, range over x1/2, x1, x2 (batch 128, constant rate):
  own batch, actual step     R_B        {rng('B.s')}
  population, actual step    R_pop      {rng('pop.s')}
      from the exact loss change        {rng('pop.exact')}
  own batch, no momentum                {rng('B.nm')}
  coherence rho = g^T s / g_B^T s       {rng('rho')}
  kappa = s^T H_B s / s^T H s           {rng('kappa')}
  eta lambda_max / threshold            {rng('lam')}
Batch 32:      R_B {rng('B.s', 'bs32')};  R_pop {rng('pop.s', 'bs32')}
Cosine, 2nd half (eta falls to 10%):  R_B {rng('B.s', 'cosine')};  R_pop {rng('pop.s', 'cosine')}

Findings
1. R_B does not settle at 1 for any optimizer: it falls throughout training, and it is far below 1 for AdamW and Muon.
2. R_pop, the same ratio with the population gradient and Hessian along the actual step, settles at 1 for all three
   optimizers, at every rate and batch size, within ~500 steps, and stays there while the cosine schedule lowers eta.
3. R_B = R_pop * kappa * rho exactly. kappa ~ 1-1.4, so R_B ~ rho: the own-batch ratio is the population threshold times
   the step's batch-population coherence (cf. the 2 rho_b / eta of 2609.34915), which falls as gradient noise dominates.
4. Along the literature directions on the own batch (no momentum) the ratio drifts too; it is not a plateau.
5. Exactly, 1 - R_pop(exact) = dL_pop / g^T S: R_pop ~ 1 means each step realizes only a few percent of its first-order
   decrease of the population loss. A stable step has R ~ eta lambda / 2 << 1, as early in training here.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out')
    args = ap.parse_args()
    runs = [r for r in load_runs(args.runs) if r.records_of('step_sharpness')]
    tab = table(runs)
    print(tab)
    if args.out:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.out) as pdf:
            text_page(pdf, header(runs), size=8.5)
            summary_page(pdf, runs)
            coherence_page(pdf, runs)
            text_page(pdf, 'Plateaus over the second half of each run: median over 6 blocks of the ratio of sums in '
                      'each block, ± half the IQR\n'
                      'B = own batch, pop = fresh sequences, mix = own-batch Hessian with the population gradient\n'
                      'rho = g_pop^T s / g_B^T s, kappa = s^T H_B s / s^T H_pop s, so that B.s = pop.s * kappa * rho\n'
                      'lam = eta lambda_max / threshold on the 32 fixed sequences (median)\n\n' + tab, size=6)
            for run in runs:
                run_page(pdf, run)


if __name__ == '__main__':
    main()
