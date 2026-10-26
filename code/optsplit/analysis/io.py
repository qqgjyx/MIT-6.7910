"""Read-only access to run directories (formats in DESIGN.md)."""

import json
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import torch

from optsplit.config import _wrap, load_yaml, root_dir, run_dir


def get_key(rec, key):
    """rec['a']['b'] for key 'a.b', or None."""
    for k in key.split('.'):
        if not isinstance(rec, dict) or k not in rec:
            return None
        rec = rec[k]
    return rec


def flatten(d, prefix=''):
    """Numeric leaves of a nested dict as {'a.b': value}."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f'{prefix}{k}.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{prefix}{k}'] = float(v)
    return out


@lru_cache(maxsize=32)
def _load_logp(path):
    return torch.load(path, map_location='cpu')['logp']


def _steps(directory, suffix):
    return sorted(int(p.stem) for p in Path(directory).glob(f'*{suffix}') if p.stem.isdigit())


class Run:
    def __init__(self, run):
        path = Path(run)
        self.path = path if path.is_absolute() else run_dir(str(run))
        try:
            self.id = str(self.path.relative_to(root_dir() / 'runs'))
        except ValueError:
            self.id = self.path.name

    def __repr__(self):
        return f'Run({self.id})'

    @cached_property
    def cfg(self):
        return _wrap(load_yaml(self.path / 'config.yaml'))

    @cached_property
    def records(self):
        """{(kind, step): record}, keeping the last record of each (kind, step)."""
        out = {}
        f = self.path / 'metrics.jsonl'
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[(r['kind'], r['step'])] = r
        return out

    def records_of(self, kind):
        return [r for (k, _), r in sorted(self.records.items(), key=lambda kv: kv[0][1]) if k == kind]

    def series(self, kind, key):
        """(steps, values) of a numeric field (dotted key) of the records of one kind."""
        pts = [(r['step'], get_key(r, key)) for r in self.records_of(kind)]
        pts = [(s, v) for s, v in pts if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return np.array([s for s, _ in pts], dtype=int), np.array([v for _, v in pts], dtype=float)

    @cached_property
    def pred_steps(self):
        return _steps(self.path / 'preds', '.pt')

    def logp(self, step):
        return _load_logp(str(self.path / 'preds' / f'{step:07d}.pt'))

    @cached_property
    def targets(self):
        return torch.load(self.path / 'preds' / 'targets.pt', map_location='cpu')['y']

    @cached_property
    def ckpt_steps(self):
        return _steps(self.path / 'ckpt', '.pt')

    def ckpt_path(self, step):
        return self.path / 'ckpt' / f'{step:07d}.pt'

    @property
    def final_step(self):
        return self.ckpt_steps[-1] if self.ckpt_steps else self.cfg.train.max_steps

    @property
    def final_pred_step(self):
        return self.pred_steps[-1]

    @cached_property
    def function_metrics(self):
        """{step: metrics dict} from function_metrics/<step>.json."""
        d = self.path / 'function_metrics'
        return {s: json.loads((d / f'{s:07d}.json').read_text()) for s in _steps(d, '.json')}

    @property
    def optimizer(self):
        return self.cfg.optimizer

    @property
    def lr(self):
        return self.cfg.optim[self.optimizer].lr

    @property
    def data_seed(self):
        return self.cfg.train.data_seed

    @property
    def init_seed(self):
        return self.cfg.train.init_seed

    @property
    def parent(self):
        return Run(self.cfg.fork.parent) if self.cfg.fork.parent else None

    @property
    def fork_step(self):
        return self.cfg.fork.step

    @property
    def done(self):
        return (self.path / 'done').exists()


def load_runs(patterns):
    """Runs whose ids ('<suite>/<name>') match any of the glob patterns, sorted by id."""
    if isinstance(patterns, str):
        patterns = [patterns]
    base = root_dir() / 'runs'
    paths = sorted({p for pat in patterns for p in base.glob(pat) if (p / 'config.yaml').exists()})
    return [Run(p) for p in paths]
