"""Layer x layer split of the top eigenvalue of the operator that sets each optimizer's threshold (H for SGD
and Muon, P^-1/2 H P^-1/2 for AdamW). With v its top eigenvector and v_l the part of v in layer l (embeddings,
blocks 1..n, final norm + head), lambda = sum_{l,m} v_l^T A v_m; cell (l, m) is that term as a fraction of
lambda (diagonal: curvature within a layer, off-diagonal: between layers). Recomputed at the saved checkpoints
of the steps `anatomy_layers` averages (at the edge of stability; all steps for Muon). AdamW needs its
optimizer state, so only fork checkpoints count for it. `--split sublayer` splits each block into attnN and mlpN,
each with the LayerNorm that feeds it (ln_1 with attention, ln_2 with the MLP).
Curvature-balanced runs (optsplit/optim/balance.py) use the checkpoints of the second half of training instead (the
threshold of the plain operator does not apply to them), and record each layer's rate multiplier m_l logged by the
balancer nearest each checkpoint; `--multipliers` splits the top eigenvector of M^1/2 A M^1/2 instead, M = diag(m_l)
(the operator whose shares ||v_l||^2 SGD's and AdamW's balancer equalizes; for Muon, M covers only the matrices
Muon updates, whose balancer works on the spectral-norm sharpness instead).
python -m optsplit.analysis.curvature_layers 'ts10m_main/*' --out analysis/ts10m/curvature_layers.json
python -m optsplit.analysis.curvature_layers --plot analysis/ts10m/curvature_{layers,sublayers}.json \
    --out curvature.pdf  # one page per JSON"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis import eos
from optsplit.analysis.io import load_runs
from optsplit.analysis.plots import _group
from optsplit.data import build_data
from optsplit.measure.hessian import HVP, adam_preconditioner, lanczos
from optsplit.models import build_model, double_backward_ok, layer, muon_params
from optsplit.optim import build_optimizer, load_optimizer_state


def steps(run):
    key = 'anatomy_precond' if run.optimizer == 'adamw' else 'anatomy'
    recs = [r for r in run.records_of('measure') if r.get(key) and r.get('lr', 0) > 0]
    at = eos.at_eos(run)
    if run.cfg.get('balance', {}).get('variant'):
        recs = [r for r in recs if r['step'] >= run.cfg.train.max_steps / 2]
    elif at is not None:
        recs = [r for r in recs if r['step'] in set(at)]
    files = {int(p.stem): p for d in (['fork'] if run.optimizer == 'adamw' else ['ckpt', 'fork'])
             for p in (run.path / d).glob('*.pt')}
    return [(r, files[r['step']]) for r in recs if r['step'] in files], key


def rate_vector(cfg, model, named, rate):
    """Each parameter's rate multiplier as Balancer.apply sets it from a logged `rate` ({block: g m_a}): 1 for
    the blocks the balancer leaves out and, for Muon, for the parameters of its auxiliary AdamW. Flat, in the order
    of `named`."""
    hidden = set(muon_params(model)[0]) if cfg.optimizer == 'muon' else {n for n, _ in named}
    return torch.cat([torch.full((p.numel(),), rate.get(layer(n, cfg.balance.blocks), 1.0) if n in hidden else 1.0,
                                 device=p.device) for n, p in named])


def matrices(run, device, split='layer', multipliers=False):
    cfg = run.cfg
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size, num_classes=None).to(device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    names = list(dict.fromkeys(layer(n, split) for n, _ in named))
    masks = {g: torch.cat([torch.full((p.numel(),), layer(n, split) == g, device=device) for n, p in named])
             for g in names}
    x, y = data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)
    bal = [b for b in run.records_of('balance') if 'rate' in b]
    sel, key = steps(run)
    rows = []
    for rec, path in sel:
        ck = torch.load(path, map_location=device)
        model.load_state_dict(ck['model'])
        b = min(bal, key=lambda b: abs(b['step'] - rec['step'])) if bal else None
        with double_backward_ok(), torch.autocast(device.type, enabled=False):
            hvp = HVP(model, x, y, params)
            s = torch.ones(sum(p.numel() for p in params), device=device)
            if run.optimizer == 'adamw':
                opt = build_optimizer('adamw', model, cfg)
                load_optimizer_state(opt, ck['optimizer']['adamw'])
                s = adam_preconditioner(opt, params).rsqrt()
            if multipliers:
                s = s * rate_vector(cfg, model, named, b['rate']).sqrt()
            op = lambda u: s * hvp(s * u)  # noqa: E731
            lam, vecs = lanczos(op, sum(p.numel() for p in params), device, cfg.measure.lanczos_iters, 1)
            v = vecs[:, 0]
            parts = [v * masks[g] for g in names]
            Av = [op(u) for u in parts]
            M = torch.stack([torch.stack([a @ c for c in Av]) for a in parts])  # M[l, m] = v_l^T A v_m
            del hvp, op
        saved = rec[key]
        rows.append(dict(step=rec['step'], names=names, M=M.tolist(), mass=[(u @ u).item() for u in parts],
                         lam=M.sum().item(), lam_lanczos=lam[0],
                         lam_saved=rec['precond_lambda_max' if key == 'anatomy_precond' else 'lambda_max'],
                         saved_dir=_group(saved['dir'], 'layer'), saved_self=_group(saved['self'], 'layer'),
                         saved_cross=saved['cross']))
        if b:
            rows[-1].update(m=[b['rate'].get(g, 1.0) for g in names], m_step=b['step'], hat=multipliers)
        r = rows[-1]
        print(f"{run.id} step {r['step']}: lambda {r['lam']:.4g} (lanczos {r['lam_lanczos']:.4g}, saved "
              f"{r['lam_saved']:.4g}), between {1 - np.trace(M.cpu().numpy()) / r['lam']:.2f}, saved "
              f"{saved['cross'] / (sum(saved['self'].values()) + saved['cross']):.2f}", flush=True)
        if b:
            print(f"  shares {np.round(r['mass'], 3).tolist()}; logged at step {b['step']}: "
                  f"{[round(v, 3) for v in b.get('check_shares', b['shares']).values()]}", flush=True)
    return rows


def average(rows):
    """Mean over data orders of each run's mean over its steps, of M / lambda and of the mass."""
    per = [(np.mean([np.array(r['M']) / r['lam'] for r in rs], 0), np.mean([r['mass'] for r in rs], 0))
           for rs in rows.values() if rs]
    return np.mean([p[0] for p in per], 0), np.mean([p[1] for p in per], 0), len(per), sum(map(len, rows.values()))


TITLES = dict(sgd='SGD: top eigenvector of $H$', adamw='AdamW: top eigenvector of $P^{-1/2}HP^{-1/2}$',
              muon='Muon: top eigenvector of $H$')
NOTES = dict(muon="(threshold: spectral-norm sharpness)")


def plot(res):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42})
    opts = [o for o in TITLES if any(res.get(o, {}).values())]
    missing = [rid for o in res for rid, rs in res[o].items() if not rs]
    avg = {o: average(res[o]) for o in opts}
    rows = {o: [r for rs in res[o].values() for r in rs] for o in opts}
    names = rows[opts[0]][0]['names']
    bal = 'm' in rows[opts[0]][0]  # balanced runs: rate multipliers under the labels, row sums by the bars
    vmax = 30  # the colour saturates above it; the cells carry the numbers
    big = len(names) > 8
    h, bottom, bar, gap, fs = (1.8, 0.36, 0.18, 0.1, 4) if big else (1.62, 0.24, 0.3, 0.18, 4.5)  # inches
    top = 0.34
    if bal:
        bottom, top = bottom + 0.1, top + 0.3
    W, H, left = 7.1, bottom + h + top, 0.3
    fig = plt.figure(figsize=(W, H))
    if bal:
        ts = sorted({r['step'] for o in opts for r in rows[o]})
        hat = rows[opts[0]][0]['hat']
        fig.text(0.1 / W, 1 - 0.06 / H, (
            f'Balanced runs, mean over the checkpoints at steps {ts[0]}-{ts[-1]}. '
            + (r'$M=\mathrm{diag}(m_\ell)$, with $m_\ell$' if hat else r'Plain operator (without $M$); $m_\ell$:')
            + r" each layer's rate multiplier logged by the balancer (under the labels)."
            + '\nRow sums ' + r'$\|v_\ell\|^2$ by the bars.'
            + (f" Left out, no checkpoint with optimizer state (for $P$): {', '.join(missing)}." if missing else '')),
            fontsize=5.5, va='top')

    def axes(x, w):
        return fig.add_axes([x / W, bottom / H, w / W, h / H])

    for i, o in enumerate(opts):
        M, mass = 100 * avg[o][0], 100 * avg[o][1]
        x = left + i * (h + 0.06 + bar + gap)
        ax, bx = axes(x, h), axes(x + h + 0.06, bar)
        im = ax.imshow(M, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
        for (l, m), val in np.ndenumerate(M):
            ax.text(m, l, f'{val:.0f}' if round(val) else '0', ha='center', va='center', fontsize=fs,
                    color='white' if val > 0.65 * vmax else '0.6' if round(val) == 0 else 'black')
        labels, title = names, TITLES[o]
        if bal:
            labels = [f'{n}\n×{m:.2g}' for n, m in zip(names, np.mean([r['m'] for r in rows[o]], 0))]
            variant = '/'.join(sorted({rid.split('/')[-1].split('_')[1] for rid, rs in res[o].items() if rs}))
            title = title.replace(':', f' {variant}:', 1)
            if hat:
                title = title.replace('$H$', '$M^{1/2}HM^{1/2}$').replace(
                    '$P^{-1/2}HP^{-1/2}$', '$M^{1/2}P^{-1/2}HP^{-1/2}M^{1/2}$')
        ax.set_xticks(range(len(names)), labels, fontsize=5 if big else 4.5 if bal else 6, rotation=90 if big else 0)
        ax.set_yticks(range(len(names)), names if i == 0 else [], fontsize=5 if big else 6)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        xc = (x + (h + 0.06 + bar) / 2) / W
        fig.text(xc, (bottom + h + 0.14) / H, title, ha='center', va='bottom', fontsize=7)
        if o in NOTES:
            fig.text(xc, (bottom + h + 0.04) / H, NOTES[o], ha='center', va='bottom', fontsize=5.5)
        bx.barh(range(len(names)), mass, height=0.7, color='0.45')
        if bal:
            for l, val in enumerate(M.sum(1)):
                bx.text(mass[l] + 4, l, f'{val:.0f}', va='center', fontsize=fs)
        bx.set_ylim(len(names) - 0.5, -0.5)
        bx.set_xlim(0, 100)
        bx.set_xticks([0, 50], ['', '50%'], fontsize=5)
        bx.tick_params(length=1.5, pad=1)
        bx.set_yticks([])
        bx.set_xlabel(r'$\|v_\ell\|^2$', fontsize=6, labelpad=1)
        for side in ('top', 'right'):
            bx.spines[side].set_visible(False)
    x = left + len(opts) * (h + 0.06 + bar + gap) - gap + 0.1
    cb = fig.colorbar(im, cax=axes(x, 0.07), extend='max')
    cb.set_label(r'% of $\lambda$', fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5, length=1.5)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', help='only plot earlier JSONs, one page each')
    p.add_argument('--split', choices=['layer', 'sublayer'], default='layer')
    p.add_argument('--multipliers', action='store_true', help="balanced runs: M^1/2 A M^1/2, M = the logged rates")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.out) as pdf:
            for f in args.plot:
                pdf.savefig(plot(json.loads(Path(f).read_text())))
        return print(f'wrote {args.out}')
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training, where the saved anatomy was measured
    res = {}
    for run in load_runs(args.runs):
        if not run.cfg.fork.parent:
            res.setdefault(run.optimizer, {})[run.id] = matrices(run, torch.device(args.device), args.split,
                                                                 args.multipliers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
