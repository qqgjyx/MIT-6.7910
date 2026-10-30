"""Muon's spectral-norm sharpness (Islamov, Crawshaw, Cohen & Gower 2026, arXiv 2603.05002):
S = max <D, H[D]> over ||D|| <= 1, with ||D|| = max_l ||D_l||_2 / s_l over the matrices Muon updates and
s_l = max(1, rows/cols)^0.5 Muon's shape factor (so that Muon's step -eta s_l polar(G_l) has norm eta).
The dual norm is sum_l s_l ||G_l||_*, and a normalized step is stable when S <= 2 sum_l s_l ||G_l||_* / eta
(their eq. 21, without momentum). S is estimated by Frank-Wolfe, D <- (1 - g) D + g s_l polar((H D)_l) with
g = 2 / (k + 2), from several random starts; the best value of <D, H D> / ||D||^2 seen is kept (a lower bound
on S). Also records lambda_max(H) (Lanczos, all parameters), and the group x group split of <D, H D>.
python -m optsplit.analysis.spectral_sharpness 'ts10m_main/*' --out analysis/ts10m/spectral_sharpness/main.json
python -m optsplit.analysis.spectral_sharpness --plot analysis/ts10m/spectral_sharpness/main.json --out fig.pdf"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.curvature_layers import layer
from optsplit.analysis.io import load_runs
from optsplit.data import build_data
from optsplit.measure.hessian import HVP, flat, lanczos, split
from optsplit.models import build_model, double_backward_ok, muon_params
from optsplit.optim import lr_mult


def polar(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def norm(D, s):
    return max(torch.linalg.matrix_norm(d, 2).item() / c for d, c in zip(D, s))


def frank_wolfe(hvp, shapes, s, device, iters=50, starts=5, seed=0):
    """(S, maximizing D, best value per start) for hvp acting on flat vectors over matrices of `shapes`
    with scales `s`."""
    g = torch.Generator(device=device).manual_seed(seed)
    best, arg, per = -float('inf'), None, []
    for _ in range(starts):
        per.append(-float('inf'))
        D = [torch.randn(sh, generator=g, device=g.device) for sh in shapes]
        for k in range(iters + 1):
            n = norm(D, s)
            D = [d / n for d in D]
            HD = split(hvp(flat(D)), D)
            q = sum((d * h).sum() for d, h in zip(D, HD)).item()
            per[-1] = max(per[-1], q)
            if q > best:
                best, arg = q, [d.clone() for d in D]
            if k < iters:
                gam = 2 / (k + 2)
                D = [(1 - gam) * d + gam * c * polar(h) for d, h, c in zip(D, HD, s)]
    return best, arg, per


def measure(run, device, iters, starts, steps=None):
    cfg = run.cfg
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size, num_classes=None).to(device)
    named = dict(model.named_parameters())
    hidden = muon_params(model)[0]
    mats = [named[n] for n in hidden]
    s = [max(1, p.shape[0] / p.shape[1]) ** 0.5 for p in mats]
    groups = list(dict.fromkeys(layer(n) for n in hidden))
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)
    files = {int(p.stem): p for d in ('ckpt', 'fork') for p in (run.path / d).glob('*.pt')}
    rows = []
    for step in sorted(f for f in files if steps is None or f in steps):
        model.load_state_dict(torch.load(files[step], map_location=device)['model'])
        with double_backward_ok(), torch.autocast(device.type, enabled=False):
            hvp = HVP(model, x, y, mats)
            nuc = sum(c * torch.linalg.matrix_norm(gr.detach(), 'nuc').item() for gr, c in zip(hvp.grads, s))
            S, D, per = frank_wolfe(hvp, [p.shape for p in mats], s, device, iters, starts)
            parts = [flat([d if layer(n) == gname else torch.zeros_like(d) for n, d in zip(hidden, D)])
                     for gname in groups]
            HD = [hvp(b) for b in parts]
            M = [[(a @ h).item() for h in HD] for a in parts]  # M[l][m] = D_l^T H D_m
            del hvp, HD
            full = HVP(model, x, y, params)
            lam = lanczos(full, sum(p.numel() for p in params), device, cfg.measure.lanczos_iters, 1)[0][0]
            del full
        eta = cfg.optim[run.optimizer].lr * lr_mult(step, cfg)
        rows.append(dict(step=step, S=S, S_starts=per, lam=lam, nuc=nuc, eta=eta, ratio=S * eta / (2 * nuc),
                         groups=groups, M=M, D_norms=[torch.linalg.matrix_norm(d, 2).item() / c
                                                      for d, c in zip(D, s)]))
        print(f"{run.id} step {step}: S {S:.4g}, lam {lam:.4g}, nuc {nuc:.4g}, eta {eta:.3g}, "
              f"ratio {rows[-1]['ratio']:.3g}", flush=True)
    return rows


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42})
    colors = dict(muon='C2', sgd='C0', adamw='C3')
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.4), constrained_layout=True)
    shares, names = {}, next(iter(res.values()))[0]['groups']
    for rid, rows in res.items():
        opt = rid.split('/')[-1].split('_')[0]
        rows = [r for r in rows if r['step'] > 0]
        st = [r['step'] for r in rows]
        c = colors.get(opt, 'k')
        ax[0].plot(st, [r['S'] for r in rows], color=c, lw=0.8, alpha=0.7, label=opt)
        ax[1].plot(st, [r['lam'] for r in rows], color=c, lw=0.8, alpha=0.7, label=opt)
        if opt == 'muon':
            ax[2].plot(st, [r['ratio'] for r in rows], lw=0.8, marker='.', ms=3, label=rid.split('/')[-1])
        late = [np.array(r['M']).sum(1) / r['S'] for r in rows if r['step'] >= 200]
        if late:
            shares.setdefault(opt, []).append(np.mean(late, 0))
    ax[2].axhline(1, color='k', ls='--', lw=0.8)
    for a, t in zip(ax[:3], ['spectral sharpness $S$', r'$\lambda_{\max}(H)$',
                             r'$S\,\eta / (2\sum_\ell s_\ell\|\nabla_\ell L\|_*)$ (Muon)']):
        a.set(xscale='symlog', yscale='log', xlabel='step', title=t)
        h, lab = a.get_legend_handles_labels()
        keep = dict(zip(lab, h))
        a.legend(keep.values(), keep.keys(), fontsize=5)
    w = 0.8 / max(1, len(shares))
    for i, (opt, sh) in enumerate(shares.items()):
        ax[3].bar(np.arange(len(names)) + i * w, 100 * np.mean(sh, 0), w, color=colors.get(opt, 'k'), label=opt)
    ax[3].set_xticks(np.arange(len(names)) + 0.4 - w / 2, names)
    ax[3].set(title=r'share of $\langle D, HD\rangle$ by block (step $\geq$ 200)', ylabel='%')
    ax[3].legend(fontsize=5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--plot', nargs='+', help='only plot earlier JSONs (merged)')
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--starts', type=int, default=5)
    p.add_argument('--steps', type=int, nargs='+', help='only these checkpoint steps')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.plot:
        return plot({k: v for f in args.plot for k, v in json.loads(Path(f).read_text()).items()}, args.out)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}  # resume after preemption
    out.parent.mkdir(parents=True, exist_ok=True)
    for run in load_runs(args.runs):
        if not run.cfg.fork.parent and run.id not in res:
            res[run.id] = measure(run, torch.device(args.device), args.iters, args.starts,
                                  args.steps and set(args.steps))
            out.write_text(json.dumps(res, indent=1))
    plot(res, out.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
