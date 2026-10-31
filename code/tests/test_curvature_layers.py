import pytest
import torch

from optsplit.analysis.curvature_layers import rate_vector
from optsplit.config import load_config
from optsplit.models import build_model, muon_params
from optsplit.optim import build_optimizer
from optsplit.optim.balance import Balancer

GPT = ['task=gpt', 'model.n_layer=2', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=6']


@pytest.mark.parametrize('name,fixed', [('sgd', []), ('adamw', ['emb']), ('muon', [])])
def test_rate_vector(name, fixed):
    """The per-parameter multipliers rebuilt from a logged `rate` are those Balancer.apply puts in the optimizer."""
    cfg = load_config(None, GPT + [f'optimizer={name}', 'balance.variant=A', f'balance.fixed={fixed}'])
    model = build_model(cfg, vocab_size=11)
    opt = build_optimizer(name, model, cfg)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    covered = [(n, p) for n, p in named if name != 'muon' or n in muon_params(model)[0]]
    bal = Balancer(cfg, opt, covered, hvp=None)
    bal.log_m = torch.linspace(-1, 1, len(bal.blocks), dtype=torch.float64)
    bal.apply()
    rate = dict(zip(bal.blocks, bal.rates().tolist()))
    mult = {p: g['lr_mult'] for g in opt.param_groups for p in g['params']}
    expect = torch.cat([torch.full((p.numel(),), mult[p]) for _, p in named])
    torch.testing.assert_close(rate_vector(cfg, model, named, rate), expect)
    assert len(set(expect.tolist())) > 2
