"""The per-layer rates of a training run, driven by optsplit.train when the config sets `balance.variant`."""

import itertools

from ..measure.hessian import HVP
from ..models import muon_params
from ..optim import lr_mult
from . import balance_cfg
from .balance import Balancer, StepBalancer
from .layer_groups import set_lr


class LayerRates:
    """F: fixed multipliers `balance.mults` ({block: m_a}, others at 1; with `balance.ramp: [a, b]`, m_a^f with f
    rising linearly from 0 at step a to 1 at step b), no balancing; Muon's auxiliary AdamW keeps the schedule's rate.
    A, B: a Balancer, on the Hessian batch (`balance.fresh`: fresh training sequences at every refresh); for Muon on
    the matrices Muon updates. R (v2): a StepBalancer, with that Balancer on the old measure only logged."""

    def __init__(self, cfg, opt, model, data):
        b = self.b = balance_cfg(cfg)
        self.cfg, self.opt, self.balancer, self.mults = cfg, opt, None, []
        if b.variant == 'F':  # set as Balancer.apply does
            self.mults = [(g, b.mults.get(g['block'], 1.0)) for g in opt.param_groups if g.get('use_muon', True)]
            return
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if cfg.optimizer == 'muon':
            named = [(n, p) for n, p in named if n in muon_params(model)[0]]
        xh, yh = data.fixed_batch('train', cfg.measure.hessian_batch, seed=1)
        draws = itertools.count(1)

        def hvp():
            x, y = data.random_batches(1, cfg.measure.hessian_batch, seed=next(draws))[0] if b.fresh else (xh, yh)
            return HVP(model, x, y, [p for _, p in named])
        self.balancer = Balancer(cfg, opt, named, hvp)
        if b.variant == 'R':
            self.balancer = StepBalancer(cfg, opt, model, data, v1=self.balancer)

    def before(self, step, log):
        """Before the update of `step`, after optsplit.optim.set_lr: refresh the multipliers (logging the balancer's
        record as 'balance') and set each group's rate to its multiplier times the schedule's."""
        mult = lr_mult(step, self.cfg)
        if self.balancer:
            rec = self.balancer.step(step, mult)
            if rec:
                log('balance', step, **rec)
        r = self.b.get('ramp')
        f = 1.0 if r is None else min(1.0, max(0.0, (step - r[0]) / (r[1] - r[0])))
        for g, ma in self.mults:
            g['lr_mult'] = ma ** f
        set_lr(self.opt, mult)

    def after(self, step, log):
        """After the update of `step` (R updates its multipliers here)."""
        rec = self.balancer and self.balancer.after(step)
        if rec:
            log('balance', step, **rec)

    def state_dict(self):
        return self.balancer and self.balancer.state_dict()

    def load(self, state):
        if self.balancer:
            self.balancer.load(state)
