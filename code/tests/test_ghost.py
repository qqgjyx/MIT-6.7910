"""Open-loop ghosts (optim.Ghost) and Muon without orthogonalization (next steps (ii))."""

import json

import numpy as np
import pytest
import torch

from optsplit import train
from optsplit.config import load_config, save_config
from optsplit.models import build_model
from optsplit.optim import Ghost, build_optimizer, set_lr
from optsplit.optim.muon import muon_update

TINY = ['model.n_layer=2', 'model.n_embd=16', 'model.n_head=2', 'model.block_size=8']


def run(rule, ghost, steps=6, extra=()):
    cfg = load_config(None, [*TINY, f'optimizer={rule}', f'ghost={ghost}', *extra])
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=11)
    opt = build_optimizer(rule, model, cfg)
    g = Ghost(model, cfg)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randint(0, 11, (4, 8), generator=gen)
        model.loss(x, x.roll(-1, 1)).backward()
        set_lr(opt, 1.0)
        opt.step()
        g.step(model, 1.0)
        opt.zero_grad(set_to_none=True)
    return model, g


@pytest.mark.parametrize('rule', ['adamw', 'muon', 'sgd'])
def test_ghost_of_own_rule_is_the_run(rule):
    model, g = run(rule, f'[{rule}]')
    for (n, p), q in zip(model.named_parameters(), g.models[rule].parameters()):
        assert torch.equal(p, q), n


def test_ghost_is_open_loop():
    alone, _ = run('adamw', '[]')
    model, g = run('adamw', '[muon]')
    for p, q in zip(alone.parameters(), model.parameters()):
        assert torch.equal(p, q)  # the ghost does not change the run
    moved = [not torch.equal(p, q) for p, q in zip(model.parameters(), g.models['muon'].parameters())]
    assert any(moved)


def test_muon_without_orthogonalization():
    torch.manual_seed(0)
    for shape in [(16, 48), (48, 16)]:
        grad = torch.randn(shape)
        u = muon_update(grad, torch.zeros(shape), orthogonalize=False)
        scale = max(1, shape[0] / shape[1]) ** 0.5
        assert u.norm().item() == pytest.approx(min(shape) ** 0.5 * scale, rel=1e-5)
        m = grad.lerp(grad * 0.05, 0.95)  # Nesterov momentum after one step from zero
        assert torch.allclose(u / u.norm(), m / m.norm(), atol=1e-6)  # direction unchanged
    model, _ = run('muon', '[]', extra=['optim.muon.orthogonalize=false'])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_writes_ghost_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path))
    d = tmp_path / 'data' / 'shakespeare_char'
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split in ('train', 'val'):
        rng.integers(0, 11, 5000).astype(np.uint16).tofile(d / f'{split}.bin')
    (d / 'meta.json').write_text(json.dumps({'vocab_size': 11, 'itos': list('abcdefghijk')}))
    cfg = load_config(None, [*TINY, 'suite=t', 'name=adamw_d0', 'optimizer=adamw', 'ghost=[adamw, muon]',
                             'train.max_steps=6', 'train.batch_size=4', 'probe.n_seqs=4', 'probe.train_seqs=4',
                             'log.ckpt_steps={every: 3}', 'measure.hessian=false', 'measure.delta_loss=false'])
    path = tmp_path / 'cfg.yaml'
    save_config(cfg, path)
    assert train.main([str(path)]) == 0
    runs = tmp_path / 'runs' / 't'
    for rule in ('adamw', 'muon'):
        g = runs / f'adamw_d0__ghost_{rule}'
        assert (g / 'done').exists()
        gcfg = load_config(g / 'config.yaml')
        assert gcfg.optimizer == rule and gcfg.ghost == [] and gcfg.ghost_of == 't/adamw_d0'
        names = sorted(p.name for p in (g / 'ckpt').iterdir())
        assert names == sorted(p.name for p in (runs / 'adamw_d0' / 'ckpt').iterdir()) and '0000006.pt' in names
    own = torch.load(runs / 'adamw_d0__ghost_adamw' / 'ckpt' / '0000006.pt')['model']
    real = torch.load(runs / 'adamw_d0' / 'ckpt' / '0000006.pt')['model']
    assert all(torch.equal(own[k], real[k]) for k in real)


def test_muon_sv_power():
    torch.manual_seed(0)
    grad = torch.randn(16, 48)
    m = grad.lerp(grad * 0.05, 0.95)
    U, S, Vh = torch.linalg.svd(m)
    for p in (0.0, 0.5, 1.0):
        u = muon_update(grad, torch.zeros_like(grad), sv_power=p)
        assert u.norm().item() == pytest.approx(16 ** 0.5, rel=1e-5)
        s = torch.linalg.svdvals(u)
        ref = S.pow(p) * (16 ** 0.5 / S.pow(p).norm())
        assert torch.allclose(s, ref, atol=1e-4)
    assert torch.allclose(muon_update(grad, torch.zeros_like(grad), sv_power=1.0),
                          muon_update(grad, torch.zeros_like(grad), orthogonalize=False), atol=1e-5)
    model, _ = run('muon', '[]', extra=['optim.muon.sv_power=0.5'])
    assert all(torch.isfinite(p).all() for p in model.parameters())
