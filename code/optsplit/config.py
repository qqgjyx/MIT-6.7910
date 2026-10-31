import copy
import os
from pathlib import Path

import numpy as np
import yaml

CODE_DIR = Path(__file__).resolve().parent.parent
DEFAULTS = CODE_DIR / 'configs' / 'defaults.yaml'


class Config(dict):
    """Nested dict with attribute access."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(d):
    if isinstance(d, dict):
        return Config({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_wrap(v) for v in d]
    return d


def _unwrap(d):
    if isinstance(d, dict):
        return {k: _unwrap(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_unwrap(v) for v in d]
    return d


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(d, key, value):
    *path, last = key.split('.')
    for p in path:
        d = d.setdefault(p, {})
    d[last] = value


def parse_overrides(items):
    """['a.b=1', 'c=[1,2]'] -> {'a': {'b': 1}, 'c': [1, 2]} (values parsed as YAML)."""
    out = {}
    for item in items:
        key, value = item.split('=', 1)
        set_dotted(out, key, yaml.safe_load(value))
    return out


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path=None, overrides=()):
    """defaults <- base config at `path` (which may name its own `base:` file) <- overrides."""
    cfg = load_yaml(DEFAULTS)
    if path is not None:
        layer = load_yaml(path)
        parent = layer.pop('base', None)
        if parent is not None:
            cfg = deep_merge(cfg, _unwrap(load_config(Path(path).parent / parent)))
        cfg = deep_merge(cfg, layer)
    if isinstance(overrides, dict):
        cfg = deep_merge(cfg, overrides)
    else:
        cfg = deep_merge(cfg, parse_overrides(overrides))
    return _wrap(cfg)


def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(_unwrap(cfg), f, sort_keys=False)


def root_dir():
    return Path(os.environ.get('OPTSPLIT_ROOT', 'workdir')).expanduser().resolve()


def run_dir(run_id):
    """`run_id` is '<suite>/<name>'."""
    return root_dir() / 'runs' / run_id


def data_dir(dataset):
    return root_dir() / 'data' / dataset


def step_grid(spec, max_steps):
    """Union of a linear grid, a geometric grid, extra steps and max_steps (see DESIGN.md)."""
    steps = {0, max_steps}
    if spec.get('every'):
        steps.update(range(0, max_steps + 1, spec['every']))
    if spec.get('geometric'):
        steps.update(np.unique(np.geomspace(1, max_steps, spec['geometric']).round().astype(int)).tolist())
    steps.update(s for s in spec.get('extra', []) if s <= max_steps)
    return sorted(int(s) for s in steps)
