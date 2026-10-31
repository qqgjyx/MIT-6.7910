import json

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from optsplit.config import load_config
from optsplit.measure import measure_step
from optsplit.measure.gradcov import gradcov
from optsplit.measure.hessian import HVP, adam_preconditioner, batch_sharpness, flat, lanczos, split
from optsplit.models import build_model, double_backward_ok

MEASURE = ['measure.hessian_batch=6', 'measure.lanczos_iters=200', 'measure.n_eigs=3',
           'measure.batch_sharpness_batches=3', 'measure.gradcov=true', 'measure.gradcov_L=6',
           'train.batch_size=4']
SETUPS = {
    'gpt': ['task=gpt', 'model.n_layer=1', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=6'],
    'cnn': ['task=cnn', 'dataset=cifar10', 'model.cnn_width=2', 'model.cnn_hidden=4'],
}


class FakeData:
    def __init__(self, kind, n=64, vocab=11, T=6):
        g = torch.Generator().manual_seed(0)
        self.kind = kind
        if kind == 'lm':
            tok = torch.randint(0, vocab, (n, T + 1), generator=g)
            self.x, self.y = tok[:, :-1], tok[:, 1:]
        else:
            self.x, self.y = torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)

    def _take(self, n, seed):
        idx = torch.randperm(len(self.x), generator=torch.Generator().manual_seed(seed))[:n]
        return self.x[idx], self.y[idx]

    def fixed_batch(self, split, n, seed):
        return self._take(n, seed)

    def random_batches(self, n_batches, batch_size, seed):
        return [self._take(batch_size, seed * 1000 + i) for i in range(n_batches)]

    def get_batch(self, step, batch_size=4, seed=0):
        return self._take(batch_size, seed * 10**6 + step)


def setup(name, extra=()):
    torch.manual_seed(0)
    cfg = load_config(None, SETUPS[name] + MEASURE + list(extra))
    model = build_model(cfg, vocab_size=11, num_classes=10)
    return cfg, model, FakeData('lm' if name == 'gpt' else 'image')


def exact_hessian(model, x, y):
    named = [(n, p) for n, p in model.named_parameters()]
    params = [p for _, p in named]

    def loss(theta):
        d = {n: t for (n, _), t in zip(named, split(theta, params))}
        logits = functional_call(model, d, (x,))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    with double_backward_ok():
        return torch.autograd.functional.hessian(loss, flat([p.detach() for p in params]))


@pytest.mark.parametrize('name', ['gpt', 'cnn'])
def test_lanczos_matches_exact_hessian(name):
    cfg, model, data = setup(name)
    x, y = data.fixed_batch('train', 6, seed=1)
    params = list(model.parameters())
    H = exact_hessian(model, x, y)
    exact = torch.linalg.eigvalsh(H).flip(0)[:3]
    with double_backward_ok():
        evals, vecs = lanczos(HVP(model, x, y, params), len(H), 'cpu', 200, 3)
    torch.testing.assert_close(torch.tensor(evals), exact, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(vecs.norm(dim=0), torch.ones(3))


@pytest.mark.parametrize('name', ['gpt', 'cnn'])
@pytest.mark.parametrize('opt', ['adamw', 'sgd', 'muon'])
def test_measure_step_end_to_end(name, opt):
    cfg, model, data = setup(name, [f'optimizer={opt}'])
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    else:  # SGD stands in for Muon: measure_step only reads optimizer state for AdamW
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    for step in range(3):
        optimizer.zero_grad()
        model.loss(*data.get_batch(step)).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    out = measure_step(model, optimizer, cfg, data, step=3)
    json.dumps(out)
    assert all(p.grad is None for p in model.parameters())
    assert len(out['eigs']) == 3 and out['lambda_max'] == out['eigs'][0]
    assert sum(out['anatomy']['dir'].values()) == pytest.approx(1, rel=1e-4)
    total = sum(out['anatomy']['self'].values()) + out['anatomy']['cross']
    assert total == pytest.approx(out['lambda_max'], rel=1e-3)
    assert set(out['gradcov']) == {'lambda_K', 'lambda_K_min', 'trace_K', 'g_over_g5'}
    assert ('precond_lambda_max' in out) == (opt == 'adamw')
    if opt == 'adamw':
        P = adam_preconditioner(optimizer, list(model.parameters()))
        x, y = data.fixed_batch('train', 6, seed=1)
        s = P.rsqrt()
        exact = torch.linalg.eigvalsh(s[:, None] * exact_hessian(model, x, y) * s[None, :])[-1]
        assert out['precond_lambda_max'] == pytest.approx(exact.item(), rel=1e-3)
        a = out['anatomy_precond']
        assert sum(a['self'].values()) + a['cross'] == pytest.approx(out['precond_lambda_max'], rel=1e-3)


def test_no_preconditioner_before_first_step():
    cfg, model, data = setup('gpt', ['optimizer=adamw'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert adam_preconditioner(optimizer, list(model.parameters())) is None
    assert 'precond_lambda_max' not in measure_step(model, optimizer, cfg, data, step=0)


@pytest.mark.parametrize('name', ['gpt', 'cnn'])
def test_batch_sharpness_matches_direct(name):
    cfg, model, data = setup(name)
    batches = data.random_batches(3, 4, seed=5)
    params = list(model.parameters())
    num = den = 0.0
    for x, y in batches:
        g = flat(torch.autograd.grad(model.loss(x, y), params))
        num += (g @ exact_hessian(model, x, y) @ g).item()
        den += (g @ g).item()
    with double_backward_ok():
        assert batch_sharpness(model, params, batches) == pytest.approx(num / den, rel=1e-4)


@pytest.mark.parametrize('name', ['gpt', 'cnn'])
def test_gradcov_matches_direct(name):
    cfg, model, data = setup(name)
    batches = data.random_batches(6, 4, seed=7)
    params = list(model.parameters())
    G = torch.stack([flat(torch.autograd.grad(model.loss(x, y), params)) for x, y in batches]).double()
    Gc = G - G.mean(0)
    evals, E = torch.linalg.eigh(Gc.T @ Gc / len(G))
    out = gradcov(model, params, batches)
    assert out['lambda_K'] == pytest.approx(evals[-1].item(), rel=1e-4)
    assert out['lambda_K_min'] == pytest.approx(evals[-5].item(), rel=1e-3)  # rank L-1 = 5
    assert out['trace_K'] == pytest.approx(evals.sum().item(), rel=1e-4)
    g5 = (G @ E[:, -5:]).norm(dim=1)
    assert out['g_over_g5'] == pytest.approx((G.norm(dim=1) / g5).mean().item(), rel=1e-3)
