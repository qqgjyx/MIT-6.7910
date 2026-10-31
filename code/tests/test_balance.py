import copy

import pytest
import torch

from optsplit.config import load_config
from optsplit.models import build_model, layer
from optsplit.optim import build_optimizer, set_lr
from optsplit.optim.balance import Balancer

GPT = ['task=gpt', 'model.n_layer=2', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=6']


@pytest.mark.parametrize('name', ['sgd', 'adamw', 'muon'])
def test_group_rates(name):
    """With balancing on, the groups split by layer, and each layer's first step is its multiplier times the
    step with every multiplier 1."""
    cfg = load_config(None, GPT + [f'optimizer={name}', 'balance.variant=A'])
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=11)
    x = torch.randint(0, 11, (4, 7))
    blocks = list(dict.fromkeys(layer(n) for n, _ in model.named_parameters()))
    assert blocks == ['emb', '1', '2', 'head']

    def first_step(mult):
        m = copy.deepcopy(model)
        opt = build_optimizer(name, m, cfg)
        name_of = {p: n for n, p in m.named_parameters()}
        members = [name_of[p] for g in opt.param_groups for p in g['params']]
        assert sorted(members) == sorted(name_of.values())
        assert all(layer(name_of[p]) == g['block'] for g in opt.param_groups for p in g['params'])
        for g in opt.param_groups:
            g['lr_mult'] = mult[g['block']]
        set_lr(opt, 0.7)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        m.loss(x[:, :-1], x[:, 1:]).backward()
        opt.step()
        return {n: p.detach() - before[n] for n, p in m.named_parameters()}

    mult = {b: 0.5 + i for i, b in enumerate(blocks)}
    plain, scaled = first_step(dict.fromkeys(blocks, 1.0)), first_step(mult)
    for n in plain:
        torch.testing.assert_close(scaled[n], mult[layer(n)] * plain[n])


@pytest.mark.parametrize('variant', ['A', 'B'])
def test_quadratic(variant):
    """On a quadratic with known H (dominated by a rank-one part spread unevenly over 4 layers), A equalizes the
    shares of the top eigenvector of D^1/2 H D^1/2, and B also brings its eigenvalue to the target."""
    names = ['transformer.wte.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.1.mlp.c_fc.weight',
             'lm_head.weight']
    sizes, norms = [3, 4, 2, 3], [2.0, 1.0, 0.6, 0.4]
    g = torch.Generator().manual_seed(0)
    u = torch.cat([c * torch.nn.functional.normalize(torch.randn(n, generator=g), dim=0) for n, c in zip(sizes, norms)])
    E = torch.randn(12, 12, generator=g)
    H = 120 * torch.outer(u, u) + E @ E.T / 12
    cfg = load_config(None, ['optimizer=sgd', 'optim.sgd.lr=0.01', f'balance.variant={variant}', 'balance.every=1',
                             'balance.start=0', 'balance.ema=0.5', 'balance.alpha=0.5', 'balance.gain=0.5'])
    params = [torch.nn.Parameter(torch.zeros(n)) for n in sizes]
    opt = torch.optim.SGD([dict(params=[p], block=layer(n)) for n, p in zip(names, params)], lr=0.01, momentum=0.9)
    for group in opt.param_groups:
        group['base_lr'] = group['lr']
    bal = Balancer(cfg, opt, list(zip(names, params)), lambda: (lambda v: H @ v))
    for step in range(300):
        rec = bal.step(step, 1.0)
    r = bal.rates()
    assert 0.1 < r.min() and r.max() < 10  # the optimum is inside the clip range
    assert [group['lr_mult'] for group in opt.param_groups] == pytest.approx(r.tolist())
    d = (0.01 * r.float())[bal.bid].sqrt()
    evals, V = torch.linalg.eigh(d[:, None] * H * d[None, :])
    shares = torch.zeros(4).index_add_(0, bal.bid, V[:, -1] ** 2)
    torch.testing.assert_close(shares, torch.full((4,), 0.25), atol=0.01, rtol=0)
    assert rec['lam'] == pytest.approx(evals[-1].item(), rel=1e-3)
    if variant == 'A':
        assert r.log().mean().item() == pytest.approx(0, abs=1e-6)
    else:
        assert evals[-1].item() == pytest.approx(0.9 * 3.8, rel=1e-3)
