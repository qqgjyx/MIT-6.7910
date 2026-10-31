import math

import torch

from ..models import layer, muon_params
from .muon import SingleDeviceMuonWithAuxAdam

OPTIMIZERS = ('sgd', 'adamw', 'muon')


def build_optimizer(name, model, cfg):
    """Each param group keeps its base learning rate in `base_lr` and a multiplier `lr_mult` (1). With
    `balance.variant` set, every group is split by layer (`balance.blocks`), named in `block`."""
    o = cfg.optim[name]
    named = dict(model.named_parameters())
    if name == 'muon':
        hidden, other = muon_params(model)
        groups = [dict(params=hidden, lr=o.lr, momentum=o.momentum, weight_decay=o.weight_decay, use_muon=True),
                  dict(params=other, lr=o.adam_lr, betas=tuple(o.adam_betas), eps=o.adam_eps,
                       weight_decay=o.adam_weight_decay, use_muon=False)]
    else:
        groups = [dict(params=[n for n, p in named.items() if p.ndim >= 2], weight_decay=o.weight_decay),
                  dict(params=[n for n, p in named.items() if p.ndim < 2], weight_decay=0.0)]
    split = cfg.get('balance', {}).get('variant') and cfg.balance.blocks
    if split:
        groups = [{**g, 'params': [n for n in g['params'] if layer(n, split) == b]}
                  for g in groups for b in dict.fromkeys(layer(n, split) for n in g['params'])]
    blocks = [layer(g['params'][0], split) if split else None for g in groups]
    for g in groups:
        g['params'] = [named[n] for n in g['params']]
    if name == 'muon':
        opt = SingleDeviceMuonWithAuxAdam(groups)
    elif name == 'sgd':
        opt = torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, nesterov=o.nesterov)
    elif name == 'adamw':
        opt = torch.optim.AdamW(groups, lr=o.lr, betas=tuple(o.betas), eps=o.eps)
    else:
        raise ValueError(name)
    for g, b in zip(opt.param_groups, blocks):
        g['base_lr'], g['lr_mult'] = g['lr'], 1.0
        if b:
            g['block'] = b
    return opt


def load_optimizer_state(opt, state_dict):
    """Load moments and step counts but keep this optimizer's hyperparameters (they come from the
    config of the run being started, not from the run that saved the state)."""
    fresh = [{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups]
    opt.load_state_dict(state_dict)
    for g, f in zip(opt.param_groups, fresh):
        g.update(f)


def lr_mult(step, cfg):
    s, T = cfg.schedule, cfg.train.max_steps
    if step < s.warmup_steps:
        return (step + 1) / s.warmup_steps
    if s.decay == 'constant':
        return 1.0
    progress = (step - s.warmup_steps) / max(1, T - s.warmup_steps)
    return s.min_lr_frac + (1 - s.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def set_lr(opt, mult):
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * g.get('lr_mult', 1.0) * mult


class Shadow:
    """Optimizers stepped with lr 0 on the training gradients: their state (momentum, second moments,
    step count) evolves as if they were training, while the parameters stay untouched."""

    def __init__(self, model, cfg):
        self.opts = {n: build_optimizer(n, model, cfg) for n in cfg.shadow if n != cfg.optimizer}
        for opt in self.opts.values():
            set_lr(opt, 0.0)

    def step(self):
        for opt in self.opts.values():
            opt.step()

    def state_dicts(self):
        return {n: opt.state_dict() for n, opt in self.opts.items()}

    def load(self, state_dicts):
        for n, opt in self.opts.items():
            if n in state_dicts:
                load_optimizer_state(opt, state_dicts[n])
                set_lr(opt, 0.0)
