"""Adaptive edge of stability of the parameters that Muon leaves to its auxiliary Adam (embeddings, head,
norms), at the fork checkpoints of a run (they hold the optimizer state). For each checkpoint: the top
eigenvalue of P^-1/2 H P^-1/2 restricted to those parameters, with P the Adam preconditioner of the run's
own optimizer (Muon's auxiliary Adam, or AdamW), against the threshold (2 + 2 beta1) / ((1 - beta1) lr).
python -m optsplit.analysis.aux_eos shakespeare_main/muon_d0 shakespeare_main/adamw_d0 [--device cuda]"""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from optsplit.analysis.io import load_runs
from optsplit.data import build_data
from optsplit.measure.hessian import HVP, adam_preconditioner, lanczos
from optsplit.models import build_model, double_backward_ok, muon_params
from optsplit.optim import build_optimizer, load_optimizer_state, lr_mult


def aux_sharpness(run, device):
    cfg = run.cfg
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size, num_classes=None).to(device)
    named = dict(model.named_parameters())
    aux = [named[n] for n in muon_params(model)[1]]
    x, y = data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)
    if run.optimizer == 'muon':
        o = cfg.optim.muon
        lr, beta1 = o.adam_lr, o.adam_betas[0]
    else:
        o = cfg.optim[run.optimizer]
        lr, beta1 = o.lr, o.betas[0]
    rows = []
    for step in cfg.log.fork_steps:
        ck = torch.load(run.path / 'fork' / f'{step:07d}.pt', map_location=device)
        model.load_state_dict(ck['model'])
        opt = build_optimizer(run.optimizer, model, cfg)
        load_optimizer_state(opt, ck['optimizer'][run.optimizer])
        s = adam_preconditioner(opt, aux).rsqrt()
        with double_backward_ok():
            hvp = HVP(model, x, y, aux)
            lam = lanczos(lambda u: s * hvp(s * u), len(s), device, cfg.measure.lanczos_iters, 1)[0][0]
            raw = lanczos(hvp, len(s), device, cfg.measure.lanczos_iters, 1)[0][0]
        eta = lr * lr_mult(step, cfg)
        rows.append(dict(step=step, precond_lambda=lam, lambda_block=raw, lr=eta,
                         ratio=lam * eta * (1 - beta1) / (2 + 2 * beta1)))
        print(f"{run.id} step {step}: precond {lam:.4g}, raw {raw:.4g}, ratio {rows[-1]['ratio']:.2f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True, help='JSON file')
    args = p.parse_args()
    res = {r.id: aux_sharpness(r, args.device) for r in load_runs(args.runs)}
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    plot(res, args.out.removesuffix('.json') + '.png')


def plot(res, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for rid, rows in res.items():
        ax.plot([r['step'] for r in rows], [r['ratio'] for r in rows], '-o', ms=3, label=rid)
    ax.axhline(1, color='k', lw=0.8)
    ax.set(xscale='log', yscale='log', xlabel='step', ylabel='sharpness / threshold',
           title='embeddings, head and norms')
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
