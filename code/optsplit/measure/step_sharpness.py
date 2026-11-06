"""Curvature of each step along the update it actually takes (stochastic edge of stability).

At a probed step with batch B, the parameters move by s + w, where w is the decoupled weight decay (AdamW,
Muon; zero for SGD, whose decay is in its gradient). To second order the step changes the batch loss by
g_B^T s + s^T H_B s / 2, so it lowers it iff R = s^T H_B s / (-2 g_B^T s) < 1. With s = -lr u, R = lr Q / 2 with
Q = u^T H_B u / g_B^T u. R does not depend on the step size, so it applies as is to Muon's mixed groups.

`before(step)` snapshots the parameters at the probed steps; `after(x, y, step)`, called after the optimizer
step, goes back to the old parameters and measures in float32, for each direction d below, the slope g^T d and
the curvatures d^T H d' on the step's batch ('B') and, at `pop_steps`, on `pop_seqs` fresh training sequences
('pop', a population estimate). Directions:
- `s.<part>`: the step without weight decay, by part (`muon`: matrices updated by Muon, `adam`: the rest;
  `all` for SGD and AdamW). The full step s is their sum.
- `wd`: the decoupled weight decay w (only if nonzero).
- `nm.<part>`: the step without momentum, from the batch gradient: -lr g (SGD), -lr g / (sqrt(v_hat) + eps)
  (AdamW and Muon's AdamW part, with the current v), -lr polar(g) (Muon matrices, same Newton-Schulz and scale
  as Muon). Along these, R is the quantity of the literature (Andreyev & Beneventano 2024 for SGD; along
  D^-1 g_B for Adam; Muon without momentum).
- `g`: the batch gradient (batch sharpness g^T H g / ||g||^2).
Batches are processed in chunks of `chunk` sequences (memory); the loss is the mean over chunks.
`analysis/stochastic_eos.py:ratios` turns a record into the ratios R."""

import torch

from ..config import step_grid
from ..models import double_backward_ok
from ..optim.muon import zeropower_via_newtonschulz5
from .hessian import _grad, flat, split


class StepSharpness:
    def __init__(self, model, opt, cfg, data):
        c, T = cfg.measure.step_sharpness, cfg.train.max_steps
        self.model, self.opt, self.data, self.chunk = model, opt, data, c.chunk
        self.pop_seqs = c.pop_seqs
        self.pop_steps = set(step_grid(c.pop_steps, T)) if c.pop_seqs else set()
        self.steps = set(step_grid(c.steps, T)) | self.pop_steps
        self.params = [p for p in model.parameters() if p.requires_grad]
        group = {p: g for g in opt.param_groups for p in g['params']}
        self.groups = [group[p] for p in self.params]
        muon = any(g.get('use_muon') for g in opt.param_groups)
        self.part = [('muon' if g['use_muon'] else 'adam') if muon else 'all' for g in self.groups]
        self.decoupled = not isinstance(opt, torch.optim.SGD)
        self.theta0 = None

    def before(self, step):
        """Call before the optimizer step. Returns whether `step` is probed."""
        if step not in self.steps:
            return False
        self.theta0 = [p.detach().clone() for p in self.params]
        return True

    def _chunks(self, x, y):
        n = len(x)
        assert n % self.chunk == 0 or n < self.chunk, f'batch {n} not a multiple of chunk {self.chunk}'
        c = min(self.chunk, n)
        return [(x[i:i + c], y[i:i + c]) for i in range(0, n, c)]

    def _pass(self, chunks, dirs):
        """Mean loss, gradient and Hessian-vector products along `dirs` (name -> flat vector)."""
        n, loss, g, Hd = len(chunks), 0.0, 0.0, {k: 0.0 for k in dirs}
        for x, y in chunks:
            out = self.model.loss(x, y)
            grads = _grad(out, self.params, create_graph=bool(dirs))
            loss += out.item() / n
            g = g + flat([t.detach() for t in grads]) / n
            for k, d in dirs.items():
                dot = sum((t * u).sum() for t, u in zip(grads, split(d, self.params)) if t.requires_grad)
                Hd[k] = Hd[k] + flat(_grad(dot, self.params, retain_graph=True)) / n
            del out, grads
        return loss, g, Hd

    def _no_momentum(self, g):
        out = []
        for p, gp, grp in zip(self.params, split(g, self.params), self.groups):
            st = self.opt.state.get(p, {})
            if grp.get('use_muon'):
                u = zeropower_via_newtonschulz5(gp, 5).float() * max(1, gp.size(-2) / gp.size(-1)) ** 0.5
            elif 'exp_avg_sq' in st:
                v_hat = st['exp_avg_sq'] / (1 - grp['betas'][1] ** float(st['step']))
                u = gp / (v_hat.sqrt() + grp['eps'])
            else:
                u = gp
            out.append(-grp['lr'] * u)
        return flat(out)

    def _mask(self, v, part):
        return flat([t if q == part else torch.zeros_like(t) for t, q in zip(split(v, self.params), self.part)])

    def _summary(self, loss0, loss1, g, Hd, dirs):
        names = list(dirs)
        rec = dict(loss0=loss0, loss1=loss1, gg=torch.dot(g, g).item(),
                   gd={k: torch.dot(g, d).item() for k, d in dirs.items()},
                   dHd={f'{a},{b}': torch.dot(dirs[a], Hd[b]).item()
                        for i, a in enumerate(names) for b in names[i:]})
        if 'muon' in self.part:  # sum of nuclear norms of the Muon matrices' gradients, times Muon's scale
            rec['nuc'] = sum(torch.linalg.matrix_norm(t, 'nuc').item() * max(1, t.size(-2) / t.size(-1)) ** 0.5
                             for t, q in zip(split(g, self.params), self.part) if q == 'muon')
        return rec

    def _losses(self, chunks):
        return sum(self.model.loss(x, y).item() for x, y in chunks) / len(chunks)

    def after(self, x, y, step):
        """Call after the optimizer step on batch (x, y). Leaves the parameters as the step left them."""
        theta1 = [p.detach().clone() for p in self.params]
        wd = [-g['lr'] * g['weight_decay'] * t if self.decoupled else torch.zeros_like(t)
              for g, t in zip(self.groups, self.theta0)]
        s = flat([t1 - t0 - w for t0, t1, w in zip(self.theta0, theta1, wd)])
        wd = flat(wd)
        parts = list(dict.fromkeys(self.part))
        batch = self._chunks(x, y)
        pop = None
        if step in self.pop_steps:
            pop = self._chunks(*self.data.random_batches(1, self.pop_seqs, seed=3 * 10**6 + step)[0])
        device = s.device
        with double_backward_ok(), torch.autocast(device.type, enabled=False):
            with torch.no_grad():
                for p, t in zip(self.params, self.theta0):
                    p.copy_(t)
            loss0, g, _ = self._pass(batch, {})
            nm = self._no_momentum(g)
            dirs = {f's.{q}': self._mask(s, q) for q in parts}
            if wd.any():
                dirs['wd'] = wd
            dirs.update({f'nm.{q}': self._mask(nm, q) for q in parts})
            dirs['g'] = g
            _, _, Hd = self._pass(batch, dirs)
            if pop:
                ploss0, pg, pHd = self._pass(pop, dirs)
            with torch.no_grad():
                for p, t in zip(self.params, theta1):
                    p.copy_(t)
                loss1 = self._losses(batch)
                ploss1 = self._losses(pop) if pop else None
        rec = dict(lr={q: grp['lr'] for q, grp in zip(self.part, self.groups)},
                   d2={k: torch.dot(d, d).item() for k, d in dirs.items()},
                   B=self._summary(loss0, loss1, g, Hd, dirs))
        if pop:
            rec['pop'] = self._summary(ploss0, ploss1, pg, pHd, dirs)
        self.theta0 = None
        return rec
