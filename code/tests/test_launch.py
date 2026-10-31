import importlib.util
from pathlib import Path

import pytest
import yaml

CODE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('launch', CODE / 'scripts' / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def write(path, obj):
    path.write_text(yaml.safe_dump(obj))
    return path


@pytest.fixture
def suite_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path / 'root'))
    d = tmp_path / 'suites'
    d.mkdir()
    write(d / 'base.yaml', {'task': 'gpt', 'dataset': 'shakespeare_char', 'train': {'max_steps': 100}})
    return d


def suite(d, runs, name='{optimizer}_d{train.data_seed}', **extra):
    return write(d / 'suite.yaml', {'suite': 's', 'base': 'base.yaml', 'name': name,
                                    'overrides': {'train.max_steps': 50}, 'runs': runs, **extra})


def test_expand(suite_dir):
    path = suite(suite_dir, [{'optimizer': ['sgd', 'adamw'], 'train.data_seed': [0, 1]},
                             {'optimizer': 'muon', 'shadow': [['adamw', 'sgd']], 'name': 'muon_shadow'}])
    name, runs = launch.expand(path)
    assert name == 's'
    assert [c.name for c in runs] == ['sgd_d0', 'sgd_d1', 'adamw_d0', 'adamw_d1', 'muon_shadow']
    assert all(c.train.max_steps == 50 and c.suite == 's' for c in runs)
    assert runs[-1].shadow == ['adamw', 'sgd']
    assert runs[0].optim.adamw.lr == 1e-3  # defaults still merged


def test_name_format(suite_dir):
    parent = suite_dir.parent / 'root/runs/main/adamw_d0'
    parent.mkdir(parents=True)
    write(parent / 'config.yaml', {'task': 'gpt', 'train': {'max_steps': 70}, 'shadow': ['muon']})
    path = suite(suite_dir, [{'optim.adamw.lr': [1e-3, 3e-4], 'fork.parent': 'main/adamw_d0'}],
                 name='{fork.parent}_lr{optim.adamw.lr}')
    _, runs = launch.expand(path)
    assert [c.name for c in runs] == ['main-adamw_d0_lr0.001', 'main-adamw_d0_lr0.0003']
    # forks start from the parent's config, then the suite's overrides
    assert runs[0].shadow == ['muon'] and runs[0].train.max_steps == 50


def test_fork_needs_launched_parent(suite_dir):
    path = suite(suite_dir, [{'optimizer': 'muon', 'fork.parent': 'main/adamw_d0', 'fork.step': 10}])
    with pytest.raises(SystemExit, match='launch the parent'):
        launch.expand(path)


def test_duplicates(suite_dir):
    path = suite(suite_dir, [{'optimizer': 'sgd', 'train.data_seed': [0, 1]}], name='{optimizer}')
    with pytest.raises(SystemExit, match='duplicate'):
        launch.expand(path)


def test_dry_run(suite_dir, capsys):
    path = suite(suite_dir, [{'optimizer': ['sgd', 'adamw'], 'train.data_seed': [0, 1]}])
    launch.main([str(path), '--dry-run', '--per-gpu', '2', '--gpu-type', 'h200'])
    out = capsys.readouterr().out
    assert '--array=0-1%2' in out and '--gres=gpu:h200:1' in out and 'mit_normal_gpu' in out
    assert not (suite_dir.parent / 'root').exists()


def test_submit_conflict_and_skips(suite_dir, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(launch.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    path = suite(suite_dir, [{'optimizer': ['sgd', 'adamw']}])
    launch.main([str(path), '--partition', 'preemptable'])
    root = suite_dir.parent / 'root'
    assert (root / 'runs/s/sgd_d0/config.yaml').exists()
    cmd = calls[-1]
    assert 'mit_preemptable' in ' '.join(cmd) and '--array=0-1%4' in cmd
    assert Path(cmd[-2]).read_text().splitlines() == [str(root / f'runs/s/{n}/config.yaml')
                                                      for n in ('sgd_d0', 'adamw_d0')]

    forks = write(suite_dir / 'forks.yaml', {'suite': 'f', 'base': 'base.yaml', 'runs': [
        {'optimizer': 'muon', 'fork.parent': 's/adamw_d0', 'fork.step': 10, 'name': 'fork_muon'}]})
    launch.main([str(forks)])
    assert not (root / 'runs/f/fork_muon').exists()  # parent fork checkpoint missing

    (root / 'runs/s/sgd_d0/done').touch()
    (root / 'runs/s/adamw_d0/fork').mkdir()
    (root / 'runs/s/adamw_d0/fork/0000010.pt').touch()
    launch.main([str(path)])
    assert [Path(p).parent.name for p in Path(calls[-1][-2]).read_text().splitlines()] == ['adamw_d0']
    launch.main([str(forks)])
    assert [Path(p).parent.name for p in Path(calls[-1][-2]).read_text().splitlines()] == ['fork_muon']

    suite(suite_dir, [{'optimizer': 'adamw', 'train.batch_size': 32}])
    with pytest.raises(SystemExit, match='different content'):
        launch.main([str(path)])
    launch.main([str(path), '--force'])
    assert yaml.safe_load((root / 'runs/s/adamw_d0/config.yaml').read_text())['train']['batch_size'] == 32
