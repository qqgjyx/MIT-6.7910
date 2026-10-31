"""Stochastic edge of stability: ratios R = d^T H d / (-2 g^T d) from the `step_sharpness` records of
optsplit/measure/step_sharpness.py, and the figures (one multi-page PDF).

    python -m optsplit.analysis.stochastic_eos --runs 'ts10m_stoch_eos/*' --out stochastic_eos.pdf
"""


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
    parts = [k[2:] for k in rec[H]['gd'] if k.startswith('s.')]
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
