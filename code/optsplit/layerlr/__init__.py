"""Per-layer learning rates driven by curvature, and per-layer curvature measurement (see README.md). The core
imports this package only for a config that sets `balance.variant` (optsplit.optim.build_optimizer, optsplit.train)."""

from pathlib import Path

from ..config import _unwrap, _wrap, deep_merge, load_yaml

DEFAULTS = load_yaml(Path(__file__).with_name('defaults.yaml'))['balance']


def balance_cfg(cfg):
    """The run's `balance` section over the defaults in defaults.yaml."""
    return _wrap(deep_merge(DEFAULTS, _unwrap(cfg.get('balance') or {})))
