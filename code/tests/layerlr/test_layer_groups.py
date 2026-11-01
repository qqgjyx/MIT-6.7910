import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from optsplit.config import load_config
from optsplit.layerlr.layer_groups import set_lr as set_layer_lr
from optsplit.models import build_model
from optsplit.optim import build_optimizer, set_lr

GPT = ['task=gpt', 'model.n_layer=2', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=6']


@pytest.mark.parametrize('name', ['sgd', 'adamw', 'muon'])
def test_split_matches_core(name):
    """With every multiplier at 1, the optimizer split by layer keeps the hyperparameters of the core's groups and
    takes bitwise the core's steps."""
    torch.manual_seed(0)
    model = build_model(load_config(None, GPT), vocab_size=11)
    x = torch.randint(0, 11, (4, 7))
    runs = []
    for over, setter in (([], set_lr), (['balance.variant=A'], set_layer_lr)):
        m = copy.deepcopy(model)
        opt = build_optimizer(name, m, load_config(None, GPT + [f'optimizer={name}', *over]))
        for _ in range(3):
            setter(opt, 0.5)
            m.loss(x[:, :-1], x[:, 1:]).backward()
            opt.step()
            opt.zero_grad()
        runs.append((m, opt))
    (core, core_opt), (split, split_opt) = runs
    assert len(split_opt.param_groups) > len(core_opt.param_groups)
    for (n, p), (_, q) in zip(core.named_parameters(), split.named_parameters()):
        assert torch.equal(p, q), n
    hyper = lambda g: {k: v for k, v in g.items() if k not in ('params', 'lr_mult', 'block')}  # noqa: E731
    name = {p: n for m in (core, split) for n, p in m.named_parameters()}
    core_group = {name[p]: g for g in core_opt.param_groups for p in g['params']}
    for g in split_opt.param_groups:
        assert hyper(g) == hyper(core_group[name[g['params'][0]]])


def test_core_without_layerlr():
    """With optsplit.layerlr unavailable, every core module imports and a plain optimizer builds."""
    code = """
import importlib, pkgutil, sys
sys.modules['optsplit.layerlr'] = None
import optsplit
for m in pkgutil.walk_packages(optsplit.__path__, 'optsplit.'):
    if not m.name.startswith('optsplit.layerlr'):
        importlib.import_module(m.name)
from optsplit.config import load_config
from optsplit.models import build_model
from optsplit.optim import build_optimizer
cfg = load_config(None, ['model.n_layer=1', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=6'])
build_optimizer('muon', build_model(cfg, vocab_size=11), cfg)
"""
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parents[2])
