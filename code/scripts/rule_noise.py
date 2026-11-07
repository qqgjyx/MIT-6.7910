"""No-signal baseline for next step (ii): each update rule fed gradient streams with no learning in them, on the
three matrix shapes of the TinyStories-10M blocks (attention 384x384, MLP in 1536x384, MLP out 384x1536), from the
model's initialization, with the main runs' learning rates and schedule (10k steps, warmup 200, cosine to 0.1).
Streams: `noise`, i.i.d. Gaussian gradients; `lowrank`, a fixed random rank-4 direction plus i.i.d. noise of the
same Frobenius norm (structure but no feedback). Muon and AdamW are scale invariant (up to AdamW's eps), so the
gradient scale does not matter. Reports, over steps, the stable rank, spectral norm and sigma_1/sigma_median of W
and the cosine of consecutive updates, as in metrics/trajectory.py.

    PYTHONPATH=code python code/scripts/rule_noise.py --out DIR [--steps 10000] [--device cuda]
    PYTHONPATH=code python code/scripts/rule_noise.py --plot DIR/rule_noise.json --out FIG.pdf
"""

import argparse
import json
import math
from pathlib import Path

import torch

from optsplit.config import load_config
from optsplit.models import build_model
from optsplit.optim import lr_mult
from optsplit.optim.muon import adam_update, muon_update

SHAPES = {'attn': 'transformer.h.0.attn.q.weight', 'mlp_in': 'transformer.h.0.mlp.c_fc.weight',
          'mlp_out': 'transformer.h.0.mlp.c_proj.weight'}


def stats(W):
    s = torch.linalg.svdvals(W.double())
    return dict(stable_rank=(s.pow(2).sum() / s[0] ** 2).item(), spectral=s[0].item(),
                sv_spread=(s[0] / s.median()).item())


def plot(path, out, kind='mlp_in', real=None):
    """Stable rank of W (noise and low-rank streams) and cosine of consecutive updates (noise stream) for one
    shape; `real`: the real runs' update cosines at training loss 1.81, drawn as dashed lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = json.loads(Path(path).read_text())['runs']
    color = {'adamw': 'C0', 'muon': 'C1'}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    panels = [('noise', 'stable_rank', '(a) stable rank of W, noise'),
              ('lowrank', 'stable_rank', '(b) stable rank of W, rank-4 signal + noise'),
              ('noise', 'cos', '(c) cosine of consecutive updates, noise')]
    for ax, (stream, key, title) in zip(axes, panels):
        for rule in ('adamw', 'muon'):
            pts = [(e['step'], e[key]) for e in runs[f'{kind}/{stream}/{rule}'] if e[key] is not None and e['step'] > 0]
            ax.plot(*zip(*pts), color=color[rule], lw=1.3, label={'adamw': 'AdamW rule', 'muon': 'Muon rule'}[rule])
            if key == 'cos' and real:
                ax.axhline(real[rule], color=color[rule], ls='--', lw=0.9)
        ax.set_xscale('log')
        ax.set_xlabel('step')
        ax.set_title(title, fontsize=9)
        if key == 'stable_rank':
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, frameon=False)
    if real:
        axes[2].text(0.98, 0.04, 'dashed: real runs at loss 1.81', transform=axes[2].transAxes, ha='right', fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plot', help='rule_noise.json to plot (to --out) instead of running')
    ap.add_argument('--steps', type=int, default=10_000)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.plot:
        plot(args.plot, args.out, real={'adamw': 0.886, 'muon': 0.738})
        return
    cfg = load_config(Path(__file__).resolve().parent.parent / 'configs/base/tinystories_10m.yaml',
                      ['optim.adamw.lr=2.0e-3', 'optim.muon.lr=0.01', f'train.max_steps={args.steps}'])
    torch.manual_seed(cfg.train.init_seed)
    named = dict(build_model(cfg, vocab_size=4096).named_parameters())
    log_steps = sorted({0, *[int(x) for x in torch.logspace(0, math.log10(args.steps), 30).round().tolist()],
                        *range(0, args.steps + 1, 1000)})
    res = {}
    for kind, pname in SHAPES.items():
        W0 = named[pname].detach().to(args.device, torch.float32)
        for stream in ('noise', 'lowrank'):
            gen = torch.Generator(args.device).manual_seed(1)
            S = sum(torch.randn(W0.shape[0], 1, generator=gen, device=args.device)
                    @ torch.randn(1, W0.shape[1], generator=gen, device=args.device) for _ in range(args.rank))
            S = S / S.norm()
            for rule in ('muon', 'adamw'):
                W, prev = W0.clone(), None
                m, v = torch.zeros_like(W), torch.zeros_like(W)
                o = cfg.optim[rule]
                rec = []
                for t in range(args.steps + 1):
                    if t in log_steps:
                        rec.append(dict(step=t, **stats(W), cos=None))
                    if t == args.steps:
                        break
                    N = torch.randn(W.shape, generator=gen, device=args.device)
                    G = N / N.norm() + (S if stream == 'lowrank' else 0)
                    lr = o.lr * lr_mult(t, cfg)
                    if rule == 'muon':
                        u = muon_update(G, m, beta=o.momentum).reshape(W.shape)
                        step = -lr * u - lr * o.weight_decay * W
                    else:
                        u = adam_update(G, m, v, t + 1, tuple(o.betas), o.eps)
                        step = -lr * u - lr * o.weight_decay * W
                    if prev is not None and t in log_steps:
                        rec[-1]['cos'] = torch.nn.functional.cosine_similarity(
                            prev.flatten(), step.flatten(), dim=0).item()
                    W += step
                    prev = step
                res[f'{kind}/{stream}/{rule}'] = rec
                last = rec[-1]
                print(f"{kind:8s} {stream:8s} {rule:6s} stable rank {last['stable_rank']:7.1f} "
                      f"spectral {last['spectral']:6.2f} s1/smed {last['sv_spread']:6.2f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rule_noise.json').write_text(json.dumps(dict(args=vars(args), runs=res)))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
