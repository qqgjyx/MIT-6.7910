"""Function-space distance between runs: Jensen-Shannon divergence and top-1 agreement of the
predictions on the fixed validation probe, aligned by step or by training loss. Distances are cached in $OPTSPLIT_ROOT/analysis/cache/distance/, one
file per pair of runs, reset when either run restarts (its preds/targets.pt is rewritten)."""

import json
import math
import os

import numpy as np
import torch

from optsplit.config import root_dir


def js_divergence(logp, logq):
    """Mean over rows of the Jensen-Shannon divergence in bits (in [0, 1])."""
    logp, logq = logp.float(), logq.float()
    logm = torch.logaddexp(logp, logq) - math.log(2)

    def kl(lp):
        p = lp.exp()
        return torch.where(p > 0, p * (lp - logm), 0.).sum(-1)

    return (0.5 * (kl(logp) + kl(logq)) / math.log(2)).mean().item()


def top1_agreement(logp, logq):
    return (logp.argmax(-1) == logq.argmax(-1)).float().mean().item()


_cache = {}


def _pair_cache(run_a, run_b):
    path = root_dir() / 'analysis' / 'cache' / 'distance' / f'{run_a.id}__{run_b.id}.json'.replace('/', '-')
    stamp = [(r.path / 'preds' / 'targets.pt').stat().st_mtime for r in (run_a, run_b)]
    if path not in _cache or _cache[path]['stamp'] != stamp:
        c = json.loads(path.read_text()) if path.exists() else {}
        _cache[path] = c if c.get('stamp') == stamp else {'stamp': stamp, 'd': {}}
    return path, _cache[path]


def distance(run_a, step_a, run_b, step_b):
    """(js, top-1 agreement) between run_a at step_a and run_b at step_b."""
    path, c = _pair_cache(run_a, run_b)
    key = f'{step_a},{step_b}'
    if key not in c['d']:
        la, lb = run_a.logp(step_a), run_b.logp(step_b)
        c['d'][key] = [js_divergence(la, lb), top1_agreement(la, lb)]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.{os.getpid()}')  # analyses may run concurrently
        tmp.write_text(json.dumps(c))
        tmp.replace(path)
    return tuple(c['d'][key])


def _curve(step_a, step_b, loss, d):
    d = np.array(d, dtype=float).reshape(-1, 2)
    return dict(step_a=np.asarray(step_a, dtype=int), step_b=np.asarray(step_b, dtype=int),
                loss=np.asarray(loss, dtype=float), js=d[:, 0], agree=d[:, 1], disagree=1 - d[:, 1])


def step_curve(a, b):
    """Distance at the prediction steps the two runs share."""
    steps = sorted(set(a.pred_steps) & set(b.pred_steps))
    ls, l = a.series('eval', 'train_loss')
    loss = np.interp(steps, ls, l) if len(ls) else np.full(len(steps), np.nan)
    return _curve(steps, steps, loss, [distance(a, s, b, s) for s in steps])


def loss_points(run):
    """(steps, training loss) at prediction steps, made monotone by a running minimum."""
    steps, loss = run.series('eval', 'train_loss')
    keep = np.isin(steps, run.pred_steps)
    return steps[keep], np.minimum.accumulate(loss[keep])


def loss_curve(a, b, tol=0.02):
    """Match each checkpoint of `a` with the first checkpoint of `b` at the closest training loss;
    drop matches whose relative loss gap exceeds `tol`. `loss` is a's (running-minimum) loss."""
    sa, la = loss_points(a)
    sb, lb = loss_points(b)
    if not len(sa) or not len(sb):
        return _curve([], [], [], [])
    j = np.abs(lb[None, :] - la[:, None]).argmin(1)
    ok = np.abs(lb[j] - la) <= tol * np.abs(la)
    pairs = list(zip(sa[ok], sb[j[ok]]))
    return _curve(sa[ok], sb[j[ok]], la[ok], [distance(a, s, b, t) for s, t in pairs])
