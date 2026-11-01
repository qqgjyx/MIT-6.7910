"""Each layer's own curvature in normal training: the top eigenvalue of the Hessian block of layer l alone,
lambda_max(H_ll) (Lanczos on u -> (H u)_l over layer l's parameters), at saved checkpoints, beside the global
lambda_max. Layers as in curvature_layers (emb: wte and wpe; blocks 1..n; head: final norm and lm_head), and the
attention and MLP of each block apart (each with the LayerNorm that feeds it). H on the 32 fixed training sequences
of the saved sharpness and, with `--fresh N`, on N other training sequences (the same N for every checkpoint and run;
the 32 fixed ones cover few vocabulary rows, so they underestimate the embedding's curvature; layers only there).
AdamW also gets the blocks of P^-1/2 H P^-1/2 (op 'P'), whose threshold is (2 + 2 beta1) / ((1 - beta1) eta), at the
steps with optimizer state (fork checkpoints); SGD's threshold on H is (2 + 2 mu) / eta. Muon has none on H.
One JSON per run in --out (resumed from: rows already there are skipped).
python -m optsplit.layerlr.layer_lanczos ts10m_main/adamw_d0 --fresh 256 --out analysis/ts10m/layer_curvature
python -m optsplit.layerlr.layer_lanczos --plot analysis/ts10m/layer_curvature/*.json --out layer_curvature.pdf"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.io import load_runs
from optsplit.data import build_data
from optsplit.measure.hessian import HVP, adam_preconditioner
from optsplit.models import build_model, double_backward_ok, layer
from optsplit.optim import build_optimizer, load_optimizer_state, lr_mult
from optsplit.layerlr.balance import threshold

STEPS = [12, 28, 45, 66, 85, 152, 221, 351, 500, 811, 1000, 1874, 3000, 4000, 5000, 6000, 8000, 10000]
FRESH_SEED = 7 * 10**6


def block_lanczos(matvecs, dim, blocks, device, iters=20, seed=0):
    """Top eigenvalue of each diagonal block A_bb of a symmetric operator A on R^dim (`blocks`: index tensors, which
    may overlap; arange(dim) gives A's own): Lanczos with full reorthogonalization, as optsplit.measure.hessian.lanczos,
    on u -> (A u)_b, for every block in lockstep, so that `matvecs`, which applies A to each vector of a list, is
    called once per iteration (a Hessian on several chunks of data then builds each chunk's graph once per
    iteration). -> list of eigenvalues."""
    g = torch.Generator(device=device).manual_seed(seed)
    k = [min(iters, len(b)) for b in blocks]
    Q = [torch.zeros(n, len(b), device=device) for n, b in zip(k, blocks)]
    q = [torch.randn(len(b), generator=g, device=device) for b in blocks]
    q = [v / v.norm() for v in q]
    alpha, beta = [[] for _ in blocks], [[] for _ in blocks]
    live = list(range(len(blocks)))
    for j in range(iters):
        V = []
        for i in live:
            Q[i][j] = q[i]
            V.append(torch.zeros(dim, device=device).index_copy_(0, blocks[i], q[i]))
        nxt = []
        for i, w in zip(live, matvecs(V)):
            w = w[blocks[i]]
            alpha[i].append(torch.dot(w, q[i]).item())
            w -= alpha[i][-1] * q[i]
            if j > 0:
                w -= beta[i][-1] * Q[i][j - 1]
            for _ in range(2):
                w -= Q[i][:j + 1].T @ (Q[i][:j + 1] @ w)
            b = w.norm().item()
            if j < k[i] - 1 and b >= 1e-10:
                beta[i].append(b)
                q[i] = w / b
                nxt.append(i)
        live = nxt
        if not live:
            break
    out = []
    for a, b in zip(alpha, beta):
        T = torch.diag(torch.tensor(a, dtype=torch.float64))
        if b:
            off = torch.tensor(b[:len(a) - 1], dtype=torch.float64)
            T += torch.diag(off, 1) + torch.diag(off, -1)
        out.append(torch.linalg.eigvalsh(T)[-1].item() if torch.isfinite(T).all() else math.nan)
    return out


def index(named, split, device):
    """{layer (or sublayer): index tensor into the flat vector of the parameters `named`}."""
    out, i = {}, 0
    for n, p in named:
        out.setdefault(layer(n, split), []).append(torch.arange(i, i + p.numel(), device=device))
        i += p.numel()
    return {g: torch.cat(v) for g, v in out.items()}


def top_eigs(model, params, batches, groups, iters, s=None):
    """{group: top eigenvalue of its diagonal block} of H, or of S H S with S = diag(s); H the mean Hessian of the
    loss over `batches` (one graph at a time, shared by all the groups' Lanczos runs)."""
    def matvecs(V):
        V = V if s is None else [s * v for v in V]
        out = [0.0] * len(V)
        for x, y in batches:
            hvp = HVP(model, x, y, params)
            out = [o + hvp(v) / len(batches) for o, v in zip(out, V)]
            del hvp
        return out if s is None else [s * w for w in out]

    lam = block_lanczos(matvecs, sum(p.numel() for p in params), list(groups.values()), params[0].device, iters)
    return dict(zip(groups, lam))


def measure(run, out, device, fresh, iters):
    """Rows {step, op ('H', or 'P' for AdamW's preconditioned operator), data ('fixed' or 'fresh'), lr (eta(t)),
    lam: {'all' (global), layer or sublayer: lambda_max of its block}} at the checkpoints in STEPS (op 'H') and at
    AdamW's fork checkpoints (op 'P'), appended to the JSON `out` after each."""
    cfg = run.cfg
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size, num_classes=None).to(device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    lay = {'all': torch.arange(sum(p.numel() for p in params), device=device), **index(named, 'layer', device)}
    sub = {g: v for g, v in index(named, 'sublayer', device).items() if g not in lay}
    sets = {'fixed': [data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)]}
    if fresh:
        x, y = data.random_batches(1, fresh, seed=FRESH_SEED)[0]
        c = cfg.measure.hessian_batch
        sets['fresh'] = [(x[i:i + c], y[i:i + c]) for i in range(0, fresh, c)]
    res = json.loads(out.read_text()) if out.exists() else dict(
        run=run.id, optimizer=run.optimizer, data_seed=run.data_seed, lr=run.lr,
        thr=None if run.optimizer == 'muon' else threshold(cfg), rows=[])
    done = {(r['step'], r['op'], r['data']) for r in res['rows']}
    todo = [(t, 'H', run.ckpt_path(t)) for t in STEPS if t in run.ckpt_steps]
    if run.optimizer == 'adamw':
        todo += [(int(p.stem), 'P', p) for p in sorted((run.path / 'fork').glob('*.pt'))]
    saved = {r['step']: r for r in run.records_of('measure')}
    for t, op, path in todo:
        keys = [d for d in sets if (t, op, d) not in done]
        if not keys:
            continue
        model.load_state_dict(torch.load(path, map_location=device)['model'])
        s = None
        if op == 'P':
            opt = build_optimizer('adamw', model, cfg)
            load_optimizer_state(opt, torch.load(path, map_location=device)['optimizer']['adamw'])
            s = adam_preconditioner(opt, params).rsqrt()
        for d in keys:
            t0 = time.time()
            with double_backward_ok(), torch.autocast(device.type, enabled=False):
                lam = top_eigs(model, params, sets[d], {**lay, **(sub if d == 'fixed' else {})}, iters, s)
            row = dict(step=t, op=op, data=d, lr=run.lr * lr_mult(t, cfg), lam=lam, seconds=time.time() - t0)
            key = 'lambda_max' if op == 'H' else 'precond_lambda_max'
            if d == 'fixed' and key in saved.get(t, {}):
                row['saved'] = saved[t][key]
            res['rows'].append(row)
            tmp = out.with_suffix('.tmp')
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(out)
            print(f"{run.id} step {t} {op} {d}: all {lam['all']:.4g} (saved {row.get('saved', float('nan')):.4g}), "
                  + ' '.join(f'{g} {v:.3g}' for g, v in lam.items() if g != 'all')
                  + f" [{row['seconds']:.0f} s]", flush=True)


LAYERS = ['emb', '1', '2', '3', '4', '5', '6', 'head']
TITLES = dict(sgd='SGD', adamw='AdamW', muon='Muon')


def gmean(runs, op, data, group, f=lambda r, v: v):
    """(steps, geometric mean over `runs` of f(row, lambda of `group`)) at each step any of them has."""
    by = {}
    for res in runs:
        for r in res['rows']:
            if r['op'] == op and r['data'] == data and group in r['lam'] and r['lam'][group] > 0:
                by.setdefault(r['step'], []).append(f(r, r['lam'][group]))
    st = sorted(by)
    return np.array(st), np.array([np.exp(np.mean(np.log(by[t]))) for t in st])


def plot(runs):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    by = {o: [r for r in runs if r['optimizer'] == o] for o in TITLES}
    opts = [o for o in TITLES if by[o]]
    lr = {o: by[o][0]['lr'] for o in opts}
    seeds = {o: sorted({r['data_seed'] for r in by[o]}) for o in opts}
    fresh = {o: [r for r in by[o] if any(x['data'] == 'fresh' for x in r['rows'])] for o in opts}
    color = {g: plt.cm.viridis(0.9 * i / (len(LAYERS) - 1)) for i, g in enumerate(LAYERS)}
    order = lambda o: 'd' + '-'.join(map(str, seeds[o])) if len(seeds[o]) > 1 else f'd{seeds[o][0]}'  # noqa: E731
    figs = []

    def lines(ax, rs, op, data, f=lambda r, v: v, groups=LAYERS, ref=True):
        for g in groups:
            ax.plot(*gmean(rs, op, data, g, f), '-o', ms=2, lw=1, color=color[g], label=g)
        if ref:
            ax.plot(*gmean(rs, op, data, 'all', f), '--', color='k', lw=1.2, label='global')
        ax.set(xscale='log', yscale='log')
        ax.grid(alpha=0.3, which='major', lw=0.5)

    # page 1: lambda_max(H_ll)
    fig, axes = plt.subplots(2, len(opts), figsize=(11, 7.2), squeeze=False, sharex=True)
    fig.suptitle(r"Each layer's own top Hessian eigenvalue $\lambda_{\max}(H_{\ell\ell})$ in normal training "
                 r"(TinyStories-10M; Lanczos, 20 iterations), and the global $\lambda_{\max}(H)$", fontsize=10)
    for j, o in enumerate(opts):
        lines(axes[0, j], by[o], 'H', 'fixed')
        axes[0, j].set_title(f'{TITLES[o]} (lr {lr[o]:g}): 32 fixed sequences, {order(o)}', fontsize=8)
        if fresh[o]:
            lines(axes[1, j], fresh[o], 'H', 'fresh')
            axes[1, j].set_title(f'{TITLES[o]}: 256 fresh sequences, '
                                 f"d{','.join(str(r['data_seed']) for r in fresh[o])}", fontsize=8)
        axes[1, j].set_xlabel('step')
    axes[0, 0].set_ylabel(r'$\lambda_{\max}(H_{\ell\ell})$')
    axes[1, 0].set_ylabel(r'$\lambda_{\max}(H_{\ell\ell})$')
    axes[0, -1].legend(fontsize=6, ncol=2)
    fig.text(0.01, 0.005, 'Geometric mean over the data orders at each checkpoint. The 32 fixed sequences (those of '
             'the saved sharpness) cover few vocabulary rows, which lowers the embedding block.', fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    figs.append(fig)

    # page 2: eta(t) lambda_ll / threshold
    fig, axes = plt.subplots(2, len(opts), figsize=(11, 7.2), squeeze=False, sharex=True)
    fig.suptitle(r"How close each layer alone is to the classic edge: $\eta(t)\,\lambda_{\max}(A_{\ell\ell})$ / "
                 'threshold', fontsize=10)
    for j, o in enumerate(opts):
        for i, d in enumerate(['fixed', 'fresh']):
            ax = axes[i, j]
            rs = by[o] if d == 'fixed' else fresh[o]
            ax.set_xlabel('step' if i else None)
            if o == 'muon':
                ax.text(0.5, 0.5, "Muon: no classic threshold on $H$\n(its hidden-layer step is normalized; its "
                        "edge is\nthe spectral-norm sharpness). Raw values on page 1.", ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.axis('off')
                continue
            op = 'P' if o == 'adamw' else 'H'
            if o == 'adamw':  # P exists only at the fork checkpoints
                rs = [r for r in rs if any(x['op'] == 'P' and x['data'] == d for x in r['rows'])]
            if not rs:
                continue
            thr = rs[0]['thr']
            lines(ax, rs, op, d, lambda r, v: r['lr'] * v / thr)
            ax.axhline(1, color='C3', lw=0.8)
            A = r'$P^{-1/2}HP^{-1/2}$' if o == 'adamw' else '$H$'
            thr_s = r'$(2+2\beta_1)/((1-\beta_1)\eta)$' if o == 'adamw' else r'$(2+2\mu)/\eta$'
            seq = '32 fixed sequences' if d == 'fixed' else '256 fresh sequences'
            ax.set_title(f"{TITLES[o]}: {A}, threshold {thr_s}\n{seq}, "
                         f"d{','.join(str(r['data_seed']) for r in rs)}", fontsize=8)
    for i in range(2):
        axes[i, 0].set_ylabel(r'$\eta(t)\,\lambda_{\max}(A_{\ell\ell})$ / threshold')
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.text(0.01, 0.005, 'AdamW: P (from the optimizer state) only at the fork checkpoints of data order 0, steps '
             '50-4000; no optimizer state was saved later or for the other orders. Red line: the classic edge.',
             fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    figs.append(fig)

    # page 3: attention and MLP apart
    fig, axes = plt.subplots(1, len(opts), figsize=(11, 4), squeeze=False)
    fig.suptitle(r'Attention (solid) and MLP (dashed) of each block apart: $\lambda_{\max}(H_{\ell\ell})$, '
                 '32 fixed sequences', fontsize=10)
    for j, o in enumerate(opts):
        ax = axes[0, j]
        for i in range(1, 7):
            for part, ls in (('attn', '-'), ('mlp', '--')):
                ax.plot(*gmean(by[o], 'H', 'fixed', f'{part}{i}'), ls, marker='o', ms=1.5, lw=1,
                        color=color[str(i)], label=f'{part}{i}')
        ax.set(xscale='log', yscale='log', xlabel='step', title=f'{TITLES[o]}, {order(o)}')
        ax.grid(alpha=0.3, lw=0.5)
    axes[0, 0].set_ylabel(r'$\lambda_{\max}$ of the block')
    axes[0, -1].legend(fontsize=5.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs.append(fig)

    # page 4: table
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.03, 0.97, 'Summary: medians over the checkpoints of the geometric means over data orders', fontsize=11,
             va='top')
    head = (f"{'':8s}{'layer':>6s} {'lam_ll':>9s} {'lam_ll':>9s} {'lam_ll/':>8s} {'fresh/':>7s} "
            f"{'max eta*lam_ll/thr':>19s} {'at step':>8s} {'max, fresh':>11s}\n"
            f"{'':8s}{'':>6s} {'<=351':>9s} {'>=811':>9s} {'global':>8s} {'fixed':>7s} {'(32 fixed)':>19s} "
            f"{'':>8s} {'(256)':>11s}")
    lines_ = [head]
    for o in opts:
        for g in LAYERS + ['all']:
            st, v = gmean(by[o], 'H', 'fixed', g)
            _, a = gmean(by[o], 'H', 'fixed', 'all')
            early, late = np.median(v[st <= 351]), np.median(v[st >= 811])
            rel = np.median((v / a)[st >= 811]) if len(a) == len(v) else np.nan
            sf, vf = gmean(fresh[o], 'H', 'fresh', g)
            s0, v0 = gmean(fresh[o], 'H', 'fixed', g)
            ff = np.median(vf / v0[np.isin(s0, sf)]) if len(vf) else np.nan
            cells = ['-', '', '-']
            if o != 'muon':
                op = 'P' if o == 'adamw' else 'H'
                rs = [r for r in by[o] if any(x['op'] == op for x in r['rows'])]
                fs = [r for r in fresh[o] if any(x['op'] == op for x in r['rows'])]
                thr = rs[0]['thr']
                sr, vr = gmean(rs, op, 'fixed', g, lambda r, v: r['lr'] * v / thr)
                cells = [f'{vr.max():.2f}', f'{sr[vr.argmax()]}', '-']
                if fs:
                    _, vrf = gmean(fs, op, 'fresh', g, lambda r, v: r['lr'] * v / thr)
                    cells[2] = f'{vrf.max():.2f}'
            lines_.append(f"{TITLES[o] if g == 'emb' else '':8s}{g:>6s} {early:9.3g} {late:9.3g} {rel:8.2f} "
                          f"{ff:7.2f} {cells[0]:>19s} {cells[1]:>8s} {cells[2]:>11s}")
        lines_.append('')
    fig.text(0.03, 0.92, '\n'.join(lines_), family='monospace', fontsize=7.5, va='top')
    fig.text(0.03, 0.06, 'lam_ll: lambda_max(H_ll) on the 32 fixed sequences, raw H for every optimizer, steps <= 351 '
             'and >= 811. lam_ll/global: its ratio to the global lambda_max(H) (<= 1 by interlacing).\nfresh/fixed: '
             'lambda_max(H_ll) on 256 fresh sequences over that on the 32 fixed ones (d0). max eta*lam_ll/thr: SGD '
             'on H, AdamW on P^-1/2 H P^-1/2 (d0, steps 50-4000); 1 is the classic edge.', fontsize=7, va='bottom')
    figs.append(fig)
    return figs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', help='glob patterns of run ids')
    p.add_argument('--fresh', type=int, default=0, help='also on this many fresh training sequences (layers only)')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--plot', nargs='+', help='only plot: run JSONs, or JSONs of {run id: run JSON}')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='directory of the run JSONs, or the PDF with --plot')
    args = p.parse_args()
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        runs = []
        for f in args.plot:
            d = json.loads(Path(f).read_text())
            runs += [d] if 'rows' in d else list(d.values())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.out) as pdf:
            for fig in plot(runs):
                pdf.savefig(fig)
        return print(f'wrote {args.out}')
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training, where the saved sharpness was measured
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in load_runs(args.runs):
        measure(run, out / f"{run.id.replace('/', '__')}.json", torch.device(args.device), args.fresh, args.iters)


if __name__ == '__main__':
    main()
