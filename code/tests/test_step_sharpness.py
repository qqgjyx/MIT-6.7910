import copy
import json

import pytest
import torch
from test_measure import FakeData, exact_hessian

from optsplit.analysis.stochastic_eos import ratios, terms
from optsplit.config import load_config
from optsplit.measure.hessian import flat
from optsplit.measure.step_sharpness import StepSharpness
from optsplit.models import build_model
from optsplit.optim import build_optimizer

STEP = ['measure.step_sharpness.steps={every: 1}', 'measure.step_sharpness.pop_steps={every: 1}',
        'measure.step_sharpness.pop_seqs=8', 'measure.step_sharpness.chunk=2', 'train.max_steps=10']


class Quadratic(torch.nn.Module):
    """loss = theta^T A theta / 2 - mean(x) . theta: Hessian A on every batch, gradient depends on the batch."""

    def __init__(self, A):
        super().__init__()
        self.A = A
        self.theta = torch.nn.Parameter(torch.randn(len(A)))

    def loss(self, x, y):
        return 0.5 * self.theta @ self.A @ self.theta - x.mean(0) @ self.theta


class QuadData:
    def __init__(self, d):
        self.x = torch.randn(64, d, generator=torch.Generator().manual_seed(1))

    def get_batch(self, step):
        idx = torch.randperm(64, generator=torch.Generator().manual_seed(step))[:8]
        return self.x[idx], torch.zeros(8)

    def random_batches(self, n_batches, batch_size, seed):
        return [self.get_batch(10**4 + seed + i) for i in range(n_batches)]


def _step(model, opt, tracker, data, step):
    x, y = data.get_batch(step)
    probed = tracker.before(step)
    model.loss(x, y).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return x, y, tracker.after(x, y, step) if probed else None


@pytest.mark.parametrize('opt_name', ['adamw', 'sgd'])
def test_quadratic_matches_analytic(opt_name):
    torch.manual_seed(0)
    d = 5
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    A = Q @ torch.diag(torch.tensor([10.0, 3.0, 1.0, 0.5, 0.1])) @ Q.T
    model, data = Quadratic(A), QuadData(d)
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.1)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    cfg = load_config(None, STEP)
    tracker = StepSharpness(model, opt, cfg, data)
    for step in range(3):
        _step(model, opt, tracker, data, step)
    theta0 = model.theta.detach().clone()
    x, y, rec = _step(model, opt, tracker, data, 3)
    json.dumps(rec)
    theta1 = model.theta.detach()
    wd = -0.05 * 0.1 * theta0 if opt_name == 'adamw' else torch.zeros(d)
    s = theta1 - theta0 - wd
    g = A @ theta0 - x.mean(0)
    R = ratios(rec)
    assert R['s'] == pytest.approx((s @ A @ s / (-2 * g @ s)).item(), rel=1e-4)
    if opt_name == 'adamw':
        assert R['s+wd'] == pytest.approx(((s + wd) @ A @ (s + wd) / (-2 * g @ (s + wd))).item(), rel=1e-4)
        st = opt.state[model.theta]
        nm = -0.05 * g / ((st['exp_avg_sq'] / (1 - 0.999 ** 4)).sqrt() + 1e-8)
    else:
        assert 'wd' not in rec['d2']
        nm = -0.05 * g
    assert R['nm'] == pytest.approx((nm @ A @ nm / (-2 * g @ nm)).item(), rel=1e-4)
    assert R['g'] == pytest.approx((g @ A @ g / (2 * g @ g)).item(), rel=1e-4)
    # the loss of a quadratic changes exactly by g^T step + step^T A step / 2
    full = theta1 - theta0
    assert rec['B']['loss1'] - rec['B']['loss0'] == pytest.approx((g @ full + full @ A @ full / 2).item(), abs=1e-5)
    # population: the same Hessian, the gradient of the fresh batch
    xp, _ = data.random_batches(1, 8, seed=3 * 10**6 + 3)[0]
    gp = A @ theta0 - xp.mean(0)
    n, den = terms(rec, H='pop', g='pop')['s']
    assert n / den == pytest.approx((s @ A @ s / (-2 * gp @ s)).item(), rel=1e-4)


def test_muon_parts_match_exact_hessian():
    torch.manual_seed(0)
    cfg = load_config(None, ['task=gpt', 'model.n_layer=1', 'model.n_embd=8', 'model.n_head=2',
                             'model.block_size=6', 'optimizer=muon', 'optim.muon.lr=0.05', *STEP])
    model, data = build_model(cfg, vocab_size=11), FakeData('lm')
    opt = build_optimizer('muon', model, cfg)
    tracker = StepSharpness(model, opt, cfg, data)
    for step in range(2):
        _step(model, opt, tracker, data, step)
    theta0 = flat([p.detach().clone() for p in model.parameters()])
    x, y, rec = _step(model, opt, tracker, data, 2)
    s = flat([p.detach() for p in model.parameters()]) - theta0
    m0 = _at(model, theta0)
    H = exact_hessian(m0, x, y)
    g = flat(torch.autograd.grad(m0.loss(x, y), list(m0.parameters())))
    assert set(rec['d2']) == {'s.muon', 's.adam', 'nm.muon', 'nm.adam', 'g'}
    R = ratios(rec)
    assert R['s'] == pytest.approx((s @ H @ s / (-2 * g @ s)).item(), rel=1e-3)
    hidden = flat([torch.full_like(p, float(p.ndim == 2 and n.startswith('transformer.h.')))
                   for n, p in model.named_parameters()])
    sm = s * hidden
    assert R['s.muon'] == pytest.approx((sm @ H @ sm / (-2 * g @ sm)).item(), rel=1e-3)
    assert rec['B']['gg'] == pytest.approx((g @ g).item(), rel=1e-4)


def _at(model, theta):
    """A copy of `model` with parameters `theta`."""
    m = copy.deepcopy(model)
    i = 0
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(theta[i:i + p.numel()].view_as(p))
            i += p.numel()
    return m
