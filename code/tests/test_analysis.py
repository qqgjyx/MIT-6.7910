"""Analysis on synthetic run directories: two optimizers that agree until T0 and then diverge,
noise-floor runs that differ only by a small seed-dependent perturbation, a time-warped run, and forks."""

import json
import math
import sys

import numpy as np
import pytest
import torch

from optsplit.analysis import breakeven, lmc, plots, split, swap
from optsplit.analysis.distance import js_divergence, loss_curve, step_curve, top1_agreement
from optsplit.analysis.io import Run, load_runs
from optsplit.config import load_config, save_config

T, T0, EVERY, N, V = 2000, 1000, 50, 400, 6
STEPS = list(range(0, T + 1, EVERY))
gen = torch.Generator().manual_seed(0)
W = 3 * torch.randn(N, V, generator=gen)
D = torch.randn(N, V, generator=gen)
E = {s: torch.randn(N, V, generator=gen) for s in range(3)}


def logits(t, seed, diverge=False, speed=1.0):
    u = min(1.0, speed * t / T)
    z = u * (W + 0.3 * E[seed])
    if diverge and t > T0:
        z = z + 8 * (t - T0) / T * D
    return z


def write_run(root, name, optimizer, seed, steps=STEPS, speed=1.0, diverge=False, fork=None, z_final=None):
    over = {'suite': 'syn', 'name': name, 'optimizer': optimizer, 'train': {'max_steps': T, 'data_seed': seed},
            'schedule': {'warmup_steps': 100}}
    if fork:
        over['fork'] = {'parent': fork[0], 'step': fork[1]}
    d = root / 'runs' / 'syn' / name
    (d / 'preds').mkdir(parents=True)
    save_config(load_config(None, over), d / 'config.yaml')
    torch.save({'y': W.argmax(-1)}, d / 'preds' / 'targets.pt')
    blocks = ['emb', 'norms', 'L0.qk', 'L0.ov', 'L0.mlp', 'L1.qk', 'L1.ov', 'L1.mlp', 'head']
    with open(d / 'metrics.jsonl', 'w') as f:
        for t in steps:
            z = z_final if (z_final is not None and t == T) else logits(t, seed, diverge, speed)
            torch.save({'logp': torch.log_softmax(z, -1).half()}, d / 'preds' / f'{t:07d}.pt')
            loss = 3 - 2 * min(1.0, speed * t / T)
            f.write(json.dumps({'kind': 'eval', 'step': t, 'train_loss': loss, 'val_loss': loss + 0.1,
                                'val_acc': 0.3 + 0.5 * t / T}) + '\n')
            share = {b: (i + 1 + t / 500) for i, b in enumerate(blocks)}
            lam = 10 + t / 100
            f.write(json.dumps({
                'kind': 'measure', 'step': t, 'lr': 1e-3 * min(1, t / 100), 'lambda_max': lam,
                'batch_sharpness': lam / 2, 'precond_lambda_max': 3 * lam, 'delta_loss': 0.01 - t / 1e5,
                'anatomy': {'dir': share, 'self': {b: 0.8 * v * lam for b, v in share.items()},
                            'cross': 0.2 * lam},
                'gradcov': {'lambda_K': 1 + t / 1000, 'lambda_K_min': 0.1, 'trace_K': 5.0, 'g_over_g5': 1.5},
            }) + '\n')
    (d / 'function_metrics').mkdir()
    for t in (0, T // 2, T):
        fm = {'input': {'jac_fro': 1 + t / T + (optimizer == 'muon')}, 'weights': {'l2': 10 + seed, 'linf': 1}}
        (d / 'function_metrics' / f'{t:07d}.json').write_text(json.dumps(fm))
    return Run(f'syn/{name}')


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp('root')
    mp = pytest.MonkeyPatch()
    mp.setenv('OPTSPLIT_ROOT', str(root))
    for s in range(3):
        write_run(root, f'adamw_d{s}', 'adamw', s)
        write_run(root, f'muon_d{s}', 'muon', s, diverge=True)
    write_run(root, 'fast_d0', 'sgd', 0, speed=1.25)
    write_run(root, 'swap_early', 'muon', 0, steps=[300, T], fork=('syn/adamw_d0', 300),
              z_final=logits(T, 0, diverge=True))
    write_run(root, 'swap_late', 'muon', 0, steps=[1500, T], fork=('syn/adamw_d0', 1500),
              z_final=logits(T, 0))
    yield root
    mp.undo()


def test_js_and_top1():
    lp = lambda p: torch.tensor(p).log()
    assert js_divergence(lp([[1., 0.]]), lp([[0., 1.]])) == pytest.approx(1.0)
    assert js_divergence(lp([[0.3, 0.7]]), lp([[0.3, 0.7]])) == pytest.approx(0.0, abs=1e-7)
    h = lambda p: -sum(x * math.log2(x) for x in p if x > 0)
    expected = h([0.75, 0.25]) - (h([0.5, 0.5]) + h([1., 0.])) / 2
    assert js_divergence(lp([[0.5, 0.5]]), lp([[1., 0.]])) == pytest.approx(expected, rel=1e-5)
    assert top1_agreement(lp([[0.6, 0.4], [0.2, 0.8]]), lp([[0.9, 0.1], [0.7, 0.3]])) == 0.5


def test_io(root):
    r = Run('syn/swap_late')
    assert r.parent.id == 'syn/adamw_d0' and r.fork_step == 1500 and r.pred_steps == [1500, T]
    assert len(load_runs('syn/adamw_d*')) == 3
    steps, lam = Run('syn/adamw_d0').series('measure', 'gradcov.lambda_K')
    assert len(steps) == len(STEPS) and lam[-1] == pytest.approx(3.0)


def test_loss_alignment(root):
    a, fast = Run('syn/adamw_d0'), Run('syn/fast_d0')
    by_step, by_loss = step_curve(a, fast), loss_curve(a, fast)
    assert by_loss['js'].max() < 0.2 * by_step['js'].max()
    inner = (by_loss['step_a'] > 0) & (by_loss['step_a'] < 1500)
    assert np.allclose(by_loss['step_b'][inner], 0.8 * by_loss['step_a'][inner], atol=EVERY)


def test_split_recovers_t_star(root):
    res = split.locate_split(load_runs('syn/adamw_d*'), load_runs('syn/muon_d*'), n_boot=50)
    step = res['steps']['syn/adamw_d0']
    assert T0 < step <= T0 + 6 * EVERY
    assert res['boot_found'] > 0.8  # resamples with a single distinct seed have no floor
    assert res['ci_loss'][1] >= res['t_star_loss'] >= res['ci_loss'][0]
    assert res['persist_loss'] == res['t_star_loss'] == res['episodes'][-1][0]  # it never comes back
    grid = [4, 3, 2, 1, 0, -1, -2]
    eps = split.episodes(grid, np.array([2, 2, 2, 0, 2, 2, 2.]), np.ones(7), 0.5, 0, 3)
    assert eps == [(4, 2), (0, -2)] and split.persistent(grid, eps) == 0
    assert np.isnan(split.persistent(grid, eps[:1]))


def test_swap_landing(root):
    pures = load_runs('syn/*_d*')
    early, late = Run('syn/swap_early'), Run('syn/swap_late')
    r_early = swap.landing(early, early.parent, swap.pure_b_of(early, pures))
    r_late = swap.landing(late, late.parent, swap.pure_b_of(late, pures))
    assert r_early['r_js'] > 0.95 and r_late['r_js'] < 0.05
    assert r_early['fm.input.jac_fro'] == pytest.approx(1.0)


def test_lmc_barrier():
    a = np.linspace(0, 1, 11)
    assert lmc.barrier(a, 1 + 4 * a * (1 - a)) == pytest.approx(1.0)
    model = torch.nn.Linear(3, 2)
    model.loss = lambda x, y: torch.nn.functional.cross_entropy(model(x), y)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    batches = [(torch.randn(8, 3), torch.randint(0, 2, (8,)))]
    alphas, losses = lmc.interpolation(model, sd, sd, batches, 5)
    assert lmc.barrier(alphas, losses) == pytest.approx(0.0, abs=1e-6)


def test_figures(root, tmp_path, monkeypatch):
    runs_a, runs_b = load_runs('syn/adamw_d*'), load_runs('syn/muon_d*')
    res = split.locate_split(runs_a, runs_b, n_boot=20)
    plots.divergence(runs_a, runs_b, res, tmp_path / 'div.png')
    plots.sharpness(runs_a[:1] + runs_b[:1] + load_runs('syn/fast_d0'), res, tmp_path / 'sharp.png')
    plots.anatomy(runs_a[:1] + runs_b[:1], tmp_path / 'anat.png')
    plots.anatomy(runs_a[:1], tmp_path / 'anat_layer.png', group='layer')
    pures = load_runs('syn/*_d*')
    rows = [swap.landing(f, f.parent, swap.pure_b_of(f, pures)) for f in load_runs('syn/swap_*')]
    plots.swap_landing(rows, tmp_path / 'swap.png', res)
    plots.swap_metrics(rows, tmp_path / 'swap_metrics.png', {'input.jac_fro': 'Jacobian'})
    plots.lmc([{'parent': 'syn/adamw_d0', 'optimizer': 'adamw',
                'forks': [{'fork': f'syn/adamw_to_muon_at{s}', 'fork_step': s, 'optimizer': 'muon', 'barrier': s / T}
                          for s in (300, 1500)]}], tmp_path / 'lmc.png')
    plots.function_metrics(runs_a + runs_b, ['input.jac_fro', 'weights.l2'], tmp_path / 'fm.png')
    plots.sweep(load_runs('syn/*_d*'), tmp_path / 'sweep.png', select='min')
    plots.eos_ratio(runs_a[:1] + load_runs('syn/fast_d0'), res, tmp_path / 'eos.png')
    plots.losses(runs_a + runs_b, tmp_path / 'losses.png')
    sgd_like = runs_a[:2]
    breakeven.curves(sgd_like, tmp_path / 'be_curves.png', acc_threshold=0.5)
    breakeven.scatter(sgd_like, tmp_path / 'be_scatter.png')
    breakeven.trajectories(sgd_like, tmp_path / 'be_traj.png', use_umap=False)
    spec = tmp_path / 'spec.yaml'
    spec.write_text(f"- {{fig: sharpness, runs: ['syn/muon_d0'], out: '{tmp_path}', name: spec_sharp}}\n")
    monkeypatch.setattr(sys, 'argv', ['plots', str(spec)])
    plots.main()
    for f in ('div', 'sharp', 'anat', 'anat_layer', 'swap', 'swap_metrics', 'lmc', 'fm', 'be_curves', 'be_scatter', 'be_traj',
              'spec_sharp', 'sweep', 'eos', 'losses'):
        assert (tmp_path / f'{f}.png').stat().st_size > 0


def test_split_cli(root, tmp_path, monkeypatch):
    out = tmp_path / 'split.json'
    monkeypatch.setattr(sys, 'argv', ['split', '--runs-a', 'syn/adamw_d*', '--runs-b', 'syn/muon_d*',
                                      '--n-boot', '10', '--out', str(out)])
    split.main()
    assert json.loads(out.read_text())['optimizer_b'] == 'muon'
