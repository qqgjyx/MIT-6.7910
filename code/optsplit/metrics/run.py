"""Function metrics for the checkpoints of a run, written to <run>/function_metrics/<step>.json.

    python -m optsplit.metrics.run <suite/name or glob> ... [--steps all|final|s1,s2,...] [--n-points N]
"""

import argparse
import json

import torch

from ..config import load_config, root_dir, run_dir
from ..data import build_data
from ..models import build_model
from .function import input_metrics
from .robustness import robustness
from .weights import weight_metrics


def ckpt_path(run_id, step):
    return run_dir(run_id) / 'ckpt' / f'{step:07d}.pt'


def init_ckpt(run_id):
    """ckpt/0000000.pt of the run, or of its nearest ancestor that has one (forks start later)."""
    while not ckpt_path(run_id, 0).exists():
        parent = load_config(run_dir(run_id) / 'config.yaml').fork.parent
        assert parent is not None, f'no initial checkpoint for {run_id}'
        run_id = parent
    return ckpt_path(run_id, 0)


def select_steps(available, spec):
    if spec == 'all':
        return available
    if spec == 'final':
        return available[-1:]
    return [int(s) for s in spec.split(',')]


def compute(run_id, args):
    rd = run_dir(run_id)
    cfg = load_config(rd / 'config.yaml')
    data = build_data(cfg, args.device)
    model = build_model(cfg, vocab_size=getattr(data, 'vocab_size', None),
                        num_classes=getattr(data, 'num_classes', None)).to(args.device).float()
    init_state = torch.load(init_ckpt(run_id), map_location=args.device)['model']
    x, y = data.fixed_batch('val', args.n_eval, seed=2)

    out_dir = rd / 'function_metrics'
    out_dir.mkdir(exist_ok=True)
    available = sorted(int(p.stem) for p in (rd / 'ckpt').glob('*.pt'))
    for step in select_steps(available, args.steps):
        out = out_dir / f'{step:07d}.json'
        if out.exists():
            continue
        model.load_state_dict(torch.load(ckpt_path(run_id, step), map_location=args.device)['model'])
        res = dict(step=step,
                   input=input_metrics(model, x[:args.n_points], y[:args.n_points], context=args.context),
                   weights=weight_metrics(model, init_state),
                   robustness=robustness(model, x, y, context=args.context))
        out.write_text(json.dumps(res, indent=1))
        print(f'{run_id} step {step}: done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='<suite>/<name> or glob patterns of them')
    ap.add_argument('--steps', default='all')
    ap.add_argument('--n-points', type=int, default=32, help='test points for the input derivatives')
    ap.add_argument('--n-eval', type=int, default=1024, help='validation examples for accuracy')
    ap.add_argument('--context', type=int, default=64, help='LM: tokens per test point')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    base = root_dir() / 'runs'
    runs = sorted({str(p.relative_to(base)) for pat in args.runs for p in base.glob(pat)
                   if (p / 'config.yaml').exists()})
    for run_id in runs:
        compute(run_id, args)


if __name__ == '__main__':
    main()
