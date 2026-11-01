"""Whole-model metrics along training, from the checkpoints of a GPT run: weights, the update of the step (from the
checkpoints at t, t+1 and t+2), curvature of the training loss, and internals on validation sequences. Written to
<run>/traj_metrics/<step>.json, with the loss on the training probe for matching runs at equal loss.

    python -m optsplit.metrics.trajectory <suite/name or glob> ... [--steps all|s1,s2,...]
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from ..config import load_config, root_dir, run_dir
from ..data import build_data
from ..measure.hessian import HVP, flat, lanczos
from ..models import build_model, double_backward_ok
from ..probe import Probe
from .run import ckpt_path, init_ckpt


def erank(s):
    """Effective rank exp(H(p)), p = s / sum(s) (Roy and Vetterli, 2007)."""
    p = s / s.sum()
    return math.exp(-(p * p.clamp_min(1e-30).log()).sum().item())


def matrix_stats(Ws):
    """Means over matrices of stable rank, effective rank, spectral norm and sigma_1 / median sigma."""
    out = dict(stable_rank=0., eff_rank=0., spectral=0., sv_spread=0.)
    for W in Ws:
        s = torch.linalg.svdvals(W.float())
        out['stable_rank'] += (s.pow(2).sum() / s[0] ** 2).item()
        out['eff_rank'] += erank(s)
        out['spectral'] += s[0].item()
        out['sv_spread'] += (s[0] / s.median()).item()
    return {k: v / len(Ws) for k, v in out.items()}


def hidden(state):
    return [v for k, v in state.items() if k.startswith('transformer.h.') and v.ndim == 2]


@torch.no_grad()
def internals(model, x):
    """Attention entropy (nats) and weight on the first token, averaged over layers, heads and query positions
    1..T-1; fraction of MLP pre-activations > 0; RMS of the residual stream after the last block; effective rank of
    the centred final hidden states (after the last LayerNorm) over all tokens."""
    tr = model.transformer
    B, T = x.shape
    h = tr.wte(x) + tr.wpe(torch.arange(T, device=x.device))
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    ent, sink, active = [], [], []
    for blk in tr.h:
        a = blk.attn
        u = blk.ln_1(h)
        C = u.size(-1)
        hd = C // a.n_head
        q, k, v = (p(u).view(B, T, a.n_head, hd).transpose(1, 2) for p in (a.q, a.k, a.v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).masked_fill(~mask, float('-inf')).softmax(-1)
        ent.append(-(att * att.clamp_min(1e-30).log()).sum(-1)[..., 1:].mean().item())
        sink.append(att[..., 1:, 0].mean().item())
        h = h + a.c_proj((att @ v).transpose(1, 2).reshape(B, T, C))
        pre = blk.mlp.c_fc(blk.ln_2(h))
        active.append((pre > 0).float().mean().item())
        h = h + blk.mlp.c_proj(F.gelu(pre))
    z = tr.ln_f(h)
    err = (model.lm_head(z) - model(x)).abs().max().item()
    assert err < 1e-2, f'internals forward differs from the model by {err}'
    z = z.reshape(-1, z.size(-1))
    return dict(attn_entropy=sum(ent) / len(ent), attn_sink=sum(sink) / len(sink),
                mlp_active=sum(active) / len(active), resid_rms=h.pow(2).mean(-1).sqrt().mean().item(),
                feature_erank=erank(torch.linalg.svdvals(z - z.mean(0))))


def curvature(model, x, y, n_probes=10):
    """Top Hessian eigenvalue (Lanczos, as measured during training) and Hutchinson trace, training loss."""
    params = [p for p in model.parameters()]
    dim, device = sum(p.numel() for p in params), params[0].device
    with double_backward_ok():
        hvp = HVP(model, x, y, params)
        lam = lanczos(hvp, dim, device, 20, 1)[0][0]
        g = torch.Generator(device=device).manual_seed(0)
        tr = 0.
        for _ in range(n_probes):
            v = torch.randint(0, 2, (dim,), generator=g, device=device).float() * 2 - 1
            tr += torch.dot(v, hvp(v)).item() / n_probes
    return dict(lambda_max=lam, hess_trace=tr)


def compute(run_id, args):
    rd = run_dir(run_id)
    cfg = load_config(rd / 'config.yaml')
    data = build_data(cfg, args.device)
    model = build_model(cfg, vocab_size=data.vocab_size).to(args.device).float()
    probe = Probe(cfg, data)
    xh, yh = data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)
    xv, _ = data.fixed_batch('val', args.n_internal, seed=3)
    init = flat(torch.load(init_ckpt(run_id), map_location=args.device)['model'].values()).float()
    out_dir = rd / 'traj_metrics'
    out_dir.mkdir(exist_ok=True)
    have = sorted(int(p.stem) for p in (rd / 'ckpt').glob('*.pt'))
    if args.steps == 'all':  # t+1 and t+2 of a consecutive triple t, t+1, t+2 only serve the update of t
        starts = [s for s in have if s + 1 in have and s + 2 in have and s - 1 not in have]
        skip = {s + i for s in starts for i in (1, 2)} - {have[-1]}
        steps = [s for s in have if s not in skip]
    else:
        steps = [int(s) for s in args.steps.split(',')]
    load = lambda s: torch.load(ckpt_path(run_id, s), map_location=args.device)['model']
    for step in steps:
        out = out_dir / f'{step:07d}.json'
        if out.exists():
            continue
        state = load(step)
        model.load_state_dict(state)
        theta = flat(state.values()).float()
        res = dict(step=step, train_loss=probe.train_loss(model))
        res['l2'] = theta.norm().item()
        res['linf_over_l2'] = (theta.abs().max() / theta.norm()).item()
        res['dist_init_rel'] = ((theta - init).norm() / init.norm()).item()
        res.update(matrix_stats(hidden(state)))
        res.update(internals(model, xv))
        res.update(curvature(model, xh, yh))
        if step + 1 in have:
            nxt = load(step + 1)
            delta = {k: nxt[k].float() - state[k].float() for k in state}
            d = flat(delta.values())
            x, y = data.get_batch(step)
            model.zero_grad(set_to_none=True)
            model.loss(x, y).backward()
            g = flat([p.grad for p in model.parameters()])
            model.zero_grad(set_to_none=True)
            res['update_rel'] = (d.norm() / theta.norm()).item()
            res['update_grad_cos'] = -(torch.dot(d, g) / (d.norm() * g.norm())).item()
            res.update({f'update_{k}': v for k, v in matrix_stats(hidden(delta)).items()
                        if k in ('stable_rank', 'eff_rank')})
            if step + 2 in have:
                d2 = flat(load(step + 2).values()).float() - flat(nxt.values()).float()
                res['update_align'] = (torch.dot(d, d2) / (d.norm() * d2.norm())).item()
        out.write_text(json.dumps(res, indent=1))
        print(f'{run_id} step {step}: loss {res["train_loss"]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='<suite>/<name> or glob patterns of them')
    ap.add_argument('--steps', default='all')
    ap.add_argument('--n-internal', type=int, default=32, help='validation sequences for the internals')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    base = root_dir() / 'runs'
    runs = sorted({str(p.relative_to(base)) for pat in args.runs for p in base.glob(pat)
                   if (p / 'config.yaml').exists()})
    for run_id in runs:
        compute(run_id, args)


if __name__ == '__main__':
    main()
