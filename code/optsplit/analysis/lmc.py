"""Linear mode connectivity (Frankle et al., 2020) between a parent run and its forks: the loss barrier
along the straight line between final checkpoints, as a function of the fork step."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.io import Run, load_runs
from optsplit.config import root_dir
from optsplit.models import build_model


def load_state(run, step=None, device='cpu'):
    return torch.load(run.ckpt_path(run.final_step if step is None else step), map_location=device)['model']


@torch.no_grad()
def mean_loss(model, batches):
    return float(np.mean([model.loss(x, y).item() for x, y in batches]))


@torch.no_grad()
def interpolation(model, sd1, sd2, batches, n_alphas=11):
    """Loss at (1-a) sd1 + a sd2 for a in linspace(0, 1, n_alphas)."""
    alphas = np.linspace(0, 1, n_alphas)
    losses = []
    for a in alphas:
        model.load_state_dict({k: torch.lerp(v.float(), sd2[k].float(), float(a)) if v.is_floating_point() else v
                               for k, v in sd1.items()})
        losses.append(mean_loss(model, batches))
    return alphas, np.array(losses)


def barrier(alphas, losses):
    return float(np.max(losses - ((1 - alphas) * losses[0] + alphas * losses[-1])))


def setup(cfg, n, chunk, seed, device):
    from optsplit.data import build_data
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=getattr(data, 'vocab_size', None),
                        num_classes=getattr(data, 'num_classes', None)).to(device).eval()
    x, y = data.fixed_batch('train', n, seed)
    return model, [(x[i:i + chunk], y[i:i + chunk]) for i in range(0, n, chunk)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, help="run id '<suite>/<name>'")
    p.add_argument('--forks', nargs='+', required=True, help='glob patterns of fork run ids')
    p.add_argument('--n', type=int, default=64, help='training examples for the loss')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alphas', type=int, default=11)
    p.add_argument('--pairs', action='store_true', help='also barriers between pairs of forks')
    p.add_argument('--others', nargs='*', default=[], help='glob patterns of other runs (e.g. other optimizers '
                   'from the same init) whose final weights are also interpolated with the parent\'s')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out')
    args = p.parse_args()

    parent = Run(args.parent)
    forks = [f for f in load_runs(args.forks) if f.cfg.fork.parent == parent.id]
    model, batches = setup(parent.cfg, args.n, args.chunk, args.seed, args.device)
    sd_parent = load_state(parent, device=args.device)
    res = dict(parent=parent.id, optimizer=parent.optimizer, forks=[], pairs=[], others=[])
    for f in sorted(forks, key=lambda f: f.fork_step):
        alphas, losses = interpolation(model, sd_parent, load_state(f, device=args.device), batches, args.alphas)
        res['forks'].append(dict(fork=f.id, fork_step=f.fork_step, optimizer=f.optimizer,
                                 barrier=barrier(alphas, losses), losses=losses.tolist()))
        print(f"{f.id} (step {f.fork_step}, {f.optimizer}): barrier {res['forks'][-1]['barrier']:.4f}")
    for o in load_runs(args.others):
        alphas, losses = interpolation(model, sd_parent, load_state(o, device=args.device), batches, args.alphas)
        res['others'].append(dict(run=o.id, optimizer=o.optimizer, barrier=barrier(alphas, losses),
                                  losses=losses.tolist()))
        print(f"{o.id}: barrier {res['others'][-1]['barrier']:.4f}")
    if args.pairs:
        for f, g in combinations(forks, 2):
            alphas, losses = interpolation(model, load_state(f, device=args.device),
                                           load_state(g, device=args.device), batches, args.alphas)
            res['pairs'].append(dict(a=f.id, b=g.id, barrier=barrier(alphas, losses), losses=losses.tolist()))
    out = Path(args.out or root_dir() / 'analysis' / 'lmc' / f"{parent.id.replace('/', '__')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
