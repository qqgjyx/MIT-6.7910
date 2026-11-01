"""Equal-loss values of the Step-2 metrics (from rank_metrics.py's metrics.json) and the ratio of two optimizers' values,
with its margin over data orders: [min A / max B, max A / min B]. Pure Python, so the table can be made anywhere.

    python metric_ratios.py metrics.json [training loss, default 1.81] > rows.tex   (rows of the appendix table)
"""
import json
import math
import sys

OPTS = ('adamw', 'muon', 'sgd')
PAIRS = (('adamw', 'muon'), ('adamw', 'sgd'), ('muon', 'sgd'))
NO_RATIO = {'gen_gap'}  # near zero and changes sign during training: a ratio is meaningless
NAMES = {
    'l2': r'weight norm $\|\theta\|_2$', 'dist_init_rel': r'distance from init.\ $/\|\theta_0\|$',
    'linf_over_l2': r'largest entry $/\|\theta\|_2$', 'stable_rank': 'stable rank of $W$',
    'eff_rank': 'effective rank of $W$', 'spectral': 'spectral norm of $W$',
    'sv_spread': r'$\sigma_1/\sigma_{\rm median}$ of $W$',
    'update_rel': r'step size $\|\Delta\theta\|/\|\theta\|$', 'update_stable_rank': r'stable rank of $\Delta W$',
    'update_eff_rank': r'effective rank of $\Delta W$', 'update_align': 'cosine of consecutive steps',
    'update_grad_cos': 'cosine of step and $-g$',
    'lambda_max': r'top Hessian eigenvalue', 'batch_sharpness': 'batch sharpness', 'hess_trace': 'Hessian trace',
    'trace_over_lmax': 'trace / top eigenvalue',
    'attn_entropy': 'attention entropy (nats)', 'attn_sink': 'attention on first token',
    'mlp_active': 'active MLP units', 'resid_rms': 'residual-stream RMS', 'feature_erank': 'effective rank of features',
    'entropy': 'prediction entropy (nats)', 'confidence': 'mean top-1 probability', 'ece': 'calibration error',
    'acc': 'top-1 accuracy', 'loss_freq': 'loss on frequent targets', 'loss_rare': 'loss on rare targets',
    'rare_minus_freq': 'rare minus frequent loss', 'js_unigram': 'JS to unigram model', 'js_bigram': 'JS to bigram model',
    'gen_gap': 'validation minus training loss',
}
GROUPS = [('weights', 'Weights'), ('updates', 'Updates (one step)'), ('curvature', 'Curvature'),
          ('internals', 'Activations'), ('function', 'Predictions')]


def at_loss(pts, L):
    """Value at training loss L, on the running-minimum loss (earliest point of each loss), linear in the loss."""
    loss, val = [], []
    for l, v in pts:
        if not loss or l < loss[-1]:
            loss.append(l)
            val.append(v)
    for j in range(len(loss) - 1):
        if loss[j + 1] <= L <= loss[j]:
            return val[j] + (loss[j] - L) / (loss[j] - loss[j + 1]) * (val[j + 1] - val[j])
    return math.nan


def values(m, level=1.81):
    """{metric: {optimizer: [value on each data order]}} at the loss level of metrics.json nearest to `level`."""
    L = min(m['levels'], key=lambda x: abs(x - level))
    out = {}
    for k, r in m['metrics'].items():
        if k in NAMES:
            v = out.setdefault(k, {})
            for rid, pts in sorted(r['points'].items()):
                v.setdefault(rid.split('/')[1].split('_d')[0], []).append(at_loss(pts, L))
    return L, out


def ratio(a, b):
    """(midpoint, half width, lower, upper) of [min a / max b, max a / min b], or None if a value is not positive."""
    if min(a + b) <= 0:
        return None
    lo, hi = min(a) / max(b), max(a) / min(b)
    return (lo + hi) / 2, (hi - lo) / 2, lo, hi


def exp10(x):
    """Decimal exponent of x after rounding to 3 significant digits."""
    return int(f'{x:.2e}'.split('e')[1])


def fmt(mu, hr, e=0, factor=True):
    """mu +- hr in units of 10^e (the factor written unless factor=False): 3 significant digits for mu, hr to the same
    decimal place, leading zero."""
    mu, hr = mu / 10 ** e, hr / 10 ** e
    d = 2 - exp10(mu)
    mu, hr = round(mu, d), round(hr, d)
    s = f'{mu:.{max(d, 0)}f} \\pm {hr:.{max(d, 0)}f}'
    return f'$({s}){{\\times}}10^{{{e}}}$' if e and factor else f'${s}$'


def row_exp(xs):
    """Scientific notation: one exponent for a row with values below 0.1 (the most common one, the larger on ties);
    otherwise none (0), except for each value from 10^4, which gets its own."""
    es = [exp10(x) for x in xs]
    if min(es) < -1:
        return [max(set(es), key=lambda e: (es.count(e), e))] * len(xs)
    return [e if e >= 4 else 0 for e in es]


def main():
    m = json.load(open(sys.argv[1]))
    L, v = values(m, float(sys.argv[2]) if len(sys.argv) > 2 else 1.81)
    rat = {k: [None if k in NO_RATIO else ratio(v[k][a], v[k][b]) for a, b in PAIRS] for k in v}
    bold = []
    for j in range(len(PAIRS)):
        far = sorted((abs(math.log(r[j][0])), k) for k, r in rat.items() if r[j] and not r[j][2] <= 1 <= r[j][3])
        bold.append({k for _, k in far[-5:]})
    out = []
    for g, name in GROUPS:
        out.append(r'\emph{' + name + r'} & & & & & & \\')
        for k in NAMES:
            if k in v and m['metrics'][k]['group'] == g:
                mh = [(sum(x) / len(x), (max(x) - min(x)) / 2) for x in (v[k][o] for o in OPTS)]
                es = row_exp([mu for mu, _ in mh])
                row = len(set(es)) == 1 and es[0]  # one exponent for the row: written once, after the name
                cells = [fmt(mu, hr, e, not row) for (mu, hr), e in zip(mh, es)]
                for j, r in enumerate(rat[k]):
                    c = '---$^\\dagger$' if r is None else fmt(r[0], r[1], row_exp([r[0]])[0])
                    cells.append(f'{{\\boldmath\\textbf{{{c}}}}}' if k in bold[j] else c)
                unit = f' ($\\times10^{{{row}}}$)' if row else ''
                out.append(f'{NAMES[k]}{unit} & ' + ' & '.join(cells) + r' \\')
    print(f'training loss {L:.3f}, {len(v)} metrics, no ratio: {sorted(k for k in rat if None in rat[k])}',
          file=sys.stderr)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
