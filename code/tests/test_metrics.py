import copy
import json
import math
import sys
import types

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from optsplit.config import load_config, save_config
from optsplit.metrics.function import derivatives, input_metrics, jacobian_fro, point_metrics, tensor_norm
from optsplit.metrics.robustness import cnn_robustness, hahn_sensitivity
from optsplit.metrics.weights import weight_metrics
from optsplit.models import build_model

try:
    import optsplit.data  # noqa: F401
except ImportError:  # data.py is written separately; run.py only needs the name at import time
    sys.modules['optsplit.data'] = types.SimpleNamespace(build_data=None)
from optsplit.metrics import run as run_mod

torch.manual_seed(0)
D = torch.float64


def smooth_f(dim=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g, dtype=D)
    b = torch.randn(dim, generator=g, dtype=D)
    return lambda x: torch.tanh(A @ x).pow(2).sum() + (b @ x) ** 3 + torch.sin(x).sum()


def hessian(f, x):
    return torch.autograd.functional.hessian(f, x)


def top_eig(Hm):
    w, V = torch.linalg.eigh(Hm)
    i = w.abs().argmax()
    return w[i], V[:, i]


def tiny_cnn():
    cfg = load_config(None, ['task=cnn', 'model.cnn_width=2', 'model.cnn_hidden=4', 'model.activation=gelu'])
    return build_model(cfg, num_classes=10).double()


def tiny_gpt(vocab=11):
    cfg = load_config(None, ['model.n_layer=1', 'model.n_embd=8', 'model.n_head=2', 'model.block_size=8'])
    return build_model(cfg, vocab_size=vocab).double()


def test_second_order_vs_exact_hessian():
    f, x = smooth_f(), torch.randn(5, dtype=D)
    gen = torch.Generator().manual_seed(1)
    m = point_metrics(f, x, n_probes=4000, n_iter=300, n_starts=1, gen=gen)
    Hm = hessian(f, x)
    g = torch.autograd.functional.jacobian(f, x)
    lam, _ = top_eig(Hm)
    g_hat = g / g.norm()
    P = torch.eye(5, dtype=D) - torch.outer(g_hat, g_hat)
    assert m['grad_norm'] == pytest.approx(g.norm().item(), rel=1e-10)
    assert m['hess_top'] == pytest.approx(lam.item(), rel=1e-6)
    assert m['hess_fro'] == pytest.approx(Hm.norm().item(), rel=0.08)
    assert m['level_curv'] == pytest.approx((P @ Hm @ P).norm().item() / g.norm().item(), rel=0.08)


def test_third_order_vs_finite_differences():
    f, x = smooth_f(), torch.randn(5, dtype=D)
    _, _, T = derivatives(f, x)
    u, h = torch.randn(5, dtype=D), 1e-4
    q = lambda z: u @ hessian(f, z) @ u
    fd = torch.stack([(q(x + h * e) - q(x - h * e)) / (2 * h) for e in torch.eye(5, dtype=D)])
    assert torch.allclose(T(u), fd, rtol=1e-6, atol=1e-6)

    # T[v,v,v] (sign fixed by v.g >= 0) and |lambda| / ||grad lambda|| with grad lambda by differences
    m = point_metrics(f, x, n_probes=2, n_iter=300, n_starts=1, gen=torch.Generator().manual_seed(0))
    lam, v = top_eig(hessian(f, x))
    if v @ torch.autograd.functional.jacobian(f, x) < 0:
        v = -v
    t_vvv = v @ (hessian(f, x + h * v) - hessian(f, x - h * v)) @ v / (2 * h)
    grad_lam = torch.stack([(top_eig(hessian(f, x + h * e))[0] - top_eig(hessian(f, x - h * e))[0]) / (2 * h)
                            for e in torch.eye(5, dtype=D)])
    assert m['t_vvv'] == pytest.approx(t_vvv.item(), rel=1e-5)
    assert m['curv_length'] == pytest.approx((lam.abs() / grad_lam.norm()).item(), rel=1e-5)


def brute_max(T3, n=200_000, seed=0):
    u = torch.randn(n, T3.shape[0], generator=torch.Generator().manual_seed(seed), dtype=D)
    u = u / u.norm(dim=1, keepdim=True)
    return torch.einsum('ijk,ni,nj,nk->n', T3, u, u, u).max().item()


def test_tensor_norm_on_cubic_form():
    A = torch.randn(3, 3, 3, dtype=D)
    A = sum(A.permute(*p) for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6
    f = lambda x: torch.einsum('ijk,i,j,k->', A, x, x, x) / 6
    _, _, T = derivatives(f, torch.randn(3, dtype=D))
    assert torch.allclose(T(torch.ones(3, dtype=D)), torch.einsum('ijk,j,k->i', A, torch.ones(3, dtype=D), torch.ones(3, dtype=D)))
    gen = torch.Generator().manual_seed(0)
    est = tensor_norm(T, lambda: torch.randn(3, generator=gen, dtype=D), n_starts=5, n_iter=100)
    brute = brute_max(A)
    assert brute - 1e-9 <= est <= brute + 1e-3 * A.abs().max().item()


def test_tensor_norm_default_settings_on_smooth_function():
    f, x = smooth_f(), torch.randn(5, dtype=D)
    _, _, T = derivatives(f, x)
    T3 = torch.stack([torch.autograd.functional.jacobian(lambda z: hessian(f, z)[i], x) for i in range(5)])
    m = point_metrics(f, x, gen=torch.Generator().manual_seed(0))
    assert m['t_sigma'] >= 0.98 * brute_max(T3)


def test_jacobian_exact_and_projected():
    cnn = tiny_cnn()
    x = torch.randn(3, 32, 32, dtype=D)
    out = lambda e: F.softmax(cnn(e[None])[0], -1)
    exact = torch.autograd.functional.jacobian(out, x).norm().item()
    assert jacobian_fro(out, x, True, 0, None) == pytest.approx(exact, rel=1e-8)

    gpt = tiny_gpt()
    e = gpt.embed(torch.randint(0, 11, (8,))).detach()
    out = lambda z: F.softmax(gpt.forward_embeds(z[None])[0], -1)
    exact = torch.autograd.functional.jacobian(out, e).norm().item()
    est = jacobian_fro(out, e, False, 3000, torch.Generator().manual_seed(0))
    assert est == pytest.approx(exact, rel=0.1)


@pytest.mark.parametrize('kind', ['cnn', 'gpt'])
def test_input_metrics_runs(kind):
    if kind == 'cnn':
        model, x, y = tiny_cnn(), torch.randn(3, 3, 32, 32, dtype=D), torch.randint(0, 10, (3,))
    else:
        model, x = tiny_gpt(), torch.randint(0, 11, (3, 9))
        x, y = x[:, :-1], x[:, 1:]
    res = input_metrics(model, x, y, context=8, n_probes=2, n_iter=5, n_starts=2)
    assert set(res) == {'grad_norm', 'hess_fro', 'hess_top', 'level_curv', 't_sigma', 't_vvv', 'curv_length', 'jac_fro'}
    assert all(math.isfinite(v['mean']) and math.isfinite(v['std']) for v in res.values())
    assert all(p.requires_grad for p in model.parameters())


def test_weight_metrics_vs_numpy():
    model = tiny_cnn()
    init = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m = weight_metrics(model, init)
    w = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
    w0 = np.concatenate([init[n].numpy().ravel() for n, _ in model.named_parameters()])
    assert m['l0'] == np.count_nonzero(np.abs(w) > 1e-8)
    assert m['l1'] == pytest.approx(np.abs(w).sum(), rel=1e-5)
    assert m['l2'] == pytest.approx(np.linalg.norm(w), rel=1e-5)
    assert m['l3'] == pytest.approx((np.abs(w) ** 3).sum() ** (1 / 3), rel=1e-5)
    assert m['linf'] == pytest.approx(np.abs(w).max(), rel=1e-6)
    assert m['dist_init'] == pytest.approx(np.linalg.norm(w - w0), rel=1e-5)

    log_prod, ratio = 0.0, 0.0
    for n, p in model.named_parameters():
        if p.ndim < 2:
            continue
        W = p.detach().numpy().reshape(p.shape[0], -1)
        s = np.linalg.svd(W, compute_uv=False)
        layer = m['layers'][n]
        assert layer['spectral'] == pytest.approx(s[0], rel=1e-5)
        assert layer['nuclear'] == pytest.approx(s.sum(), rel=1e-5)
        assert layer['stable_rank'] == pytest.approx((s ** 2).sum() / s[0] ** 2, rel=1e-5)
        log_prod += np.log(s[0])
        ratio += (np.linalg.norm(W - init[n].numpy().reshape(W.shape), axis=1).sum() / s[0]) ** (2 / 3)
    assert m['log_spectral_complexity'] == pytest.approx(log_prod + 1.5 * np.log(ratio), rel=1e-5)

    sq = {n: p.detach() ** 2 for n, p in model.named_parameters()}
    h = torch.ones(1, 3, 32, 32, dtype=D)
    h = F.conv2d(F.conv2d(h, sq['conv1.weight'], sq['conv1.bias'], padding=1), sq['conv2.weight'], sq['conv2.bias'])
    h = F.max_pool2d(h, 2)
    h = F.conv2d(F.conv2d(h, sq['conv3.weight'], sq['conv3.bias'], padding=1), sq['conv4.weight'], sq['conv4.bias'])
    h = F.max_pool2d(h, 2).flatten(1)
    h = F.linear(F.linear(h, sq['fc1.weight'], sq['fc1.bias']), sq['fc2.weight'], sq['fc2.bias'])
    assert m['path_norm'] == pytest.approx(h.sum().item(), rel=1e-6)
    assert weight_metrics(tiny_gpt())['path_norm'] is None


def test_robustness():
    cnn = tiny_cnn()
    x, y = torch.randn(20, 3, 32, 32, dtype=D), torch.randint(0, 10, (20,))
    r = cnn_robustness(cnn, x, y, sigmas=(0.0, 1.0))
    assert r['acc_noise_0'] == r['acc']

    gpt = tiny_gpt()
    seq = torch.randint(0, 11, (8,))
    res = hahn_sensitivity(gpt, seq[None], 11, seed=3)
    shift = torch.randint(1, 11, (8,), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        base = gpt(seq[None])[0, -1].argmax()
        count = 0
        for i in range(8):
            v = seq.clone()
            v[i] = (seq[i] + shift[i]) % 11
            assert v[i] != seq[i]
            count += int(gpt(v[None])[0, -1].argmax() != base)
    assert res['sensitivity_count'] == count
    assert res['sensitivity'] == count / 8


def test_run_cli(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path))
    cfg = load_config(None, ['task=cnn', 'dataset=cifar10', 'model.cnn_width=2', 'model.cnn_hidden=4'])
    model = build_model(cfg, num_classes=10)
    parent = tmp_path / 'runs' / 's' / 'parent'
    child = tmp_path / 'runs' / 's' / 'child'
    for rd in (parent, child):
        (rd / 'ckpt').mkdir(parents=True)
    save_config(cfg, parent / 'config.yaml')
    torch.save({'model': model.state_dict()}, parent / 'ckpt' / '0000000.pt')
    cfg.fork.parent, cfg.fork.step = 's/parent', 10
    save_config(cfg, child / 'config.yaml')
    for step in (10, 20):
        torch.save({'model': build_model(cfg, num_classes=10).state_dict()}, child / 'ckpt' / f'{step:07d}.pt')

    assert run_mod.init_ckpt('s/child') == parent / 'ckpt' / '0000000.pt'
    assert run_mod.select_steps([0, 10, 20], 'final') == [20]
    assert run_mod.select_steps([0, 10, 20], '0,20') == [0, 20]

    class FakeData:
        kind, num_classes = 'image', 10

        def fixed_batch(self, split, n, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)

    monkeypatch.setattr(run_mod, 'build_data', lambda cfg, device: FakeData())
    monkeypatch.setattr(sys, 'argv', ['run', 's/child', '--n-points', '2', '--n-eval', '16', '--device', 'cpu'])
    run_mod.main()
    out = json.loads((child / 'function_metrics' / '0000020.json').read_text())
    assert out['step'] == 20 and out['weights']['dist_init'] > 0 and 'acc_noise_0.1' in out['robustness']
    assert (child / 'function_metrics' / '0000010.json').exists()
