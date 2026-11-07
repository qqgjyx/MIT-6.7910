import copy
import math

import torch

from ..models import muon_params
from .muon import SingleDeviceMuonWithAuxAdam

OPTIMIZERS = ('sgd', 'adamw', 'muon')


def build_optimizer(name, model, cfg):
    o = cfg.optim[name]
    named = dict(model.named_parameters())
    if name == 'muon':
        hidden, other = muon_params(model)
        groups = [dict(params=[named[n] for n in hidden], lr=o.lr, momentum=o.momentum,
                       weight_decay=o.weight_decay, use_muon=True, orthogonalize=o.get('orthogonalize', True),
                       sv_power=o.get('sv_power')),
                  dict(params=[named[n] for n in other], lr=o.adam_lr, betas=tuple(o.adam_betas),
                       eps=o.adam_eps, weight_decay=o.adam_weight_decay, use_muon=False)]
        opt = SingleDeviceMuonWithAuxAdam(groups)
    else:
        decay = [p for p in named.values() if p.ndim >= 2]
        no_decay = [p for p in named.values() if p.ndim < 2]
        groups = [dict(params=decay, weight_decay=o.weight_decay), dict(params=no_decay, weight_decay=0.0)]
        if name == 'sgd':
            opt = torch.optim.SGD(groups, lr=o.lr, momentum=o.momentum, nesterov=o.nesterov)
        elif name == 'adamw':
            opt = torch.optim.AdamW(groups, lr=o.lr, betas=tuple(o.betas), eps=o.eps)
        else:
            raise ValueError(name)
    for g in opt.param_groups:
        g['base_lr'] = g['lr']
    if cfg.get('balance', {}).get('variant'):  # per-layer learning rates (optsplit/layerlr): one group per layer
        from ..layerlr.layer_groups import split_by_layer
        opt = split_by_layer(opt, model, cfg)
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
        g['lr'] = g['base_lr'] * mult


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


class Ghost:
    """Open-loop copies of the model (next steps (ii)): for each rule in `cfg.ghost`, a copy of the model that starts
    at the run's initialization and is updated by that rule (its own state, hyperparameters and learning rate
    schedule) on the gradients of the *run*, computed at the run's weights. The copy never feeds back into the
    gradients it receives. A ghost of the run's own rule reproduces the run."""

    def __init__(self, model, cfg):
        self.models, self.opts = {}, {}
        for rule in cfg.get('ghost', []) or []:
            g = copy.deepcopy(model)
            for p in g.parameters():
                p.requires_grad_(True)
                p.grad = None
            self.models[rule] = g
            self.opts[rule] = build_optimizer(rule, g, cfg)

    def step(self, model, mult):
        """After the run's optimizer step (which must leave .grad untouched), before zero_grad."""
        for rule, g in self.models.items():
            for p, q in zip(model.parameters(), g.parameters()):
                q.grad = None if p.grad is None else p.grad.detach().clone()
            set_lr(self.opts[rule], mult)
            self.opts[rule].step()
            self.opts[rule].zero_grad(set_to_none=True)

    def state_dicts(self):
        return {r: dict(model=self.models[r].state_dict(), optimizer=self.opts[r].state_dict()) for r in self.models}

    def load(self, state):
        for r, s in (state or {}).items():
            if r in self.models:
                self.models[r].load_state_dict(s['model'])
                load_optimizer_state(self.opts[r], s['optimizer'])
