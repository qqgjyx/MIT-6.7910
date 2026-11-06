"""One param group per layer, each with its own rate multiplier `lr_mult`."""

from ..models import layer
from . import balance_cfg


def split_by_layer(opt, model, cfg):
    """`opt` (from optsplit.optim.build_optimizer, not yet stepped) rebuilt with each param group split by layer
    (`balance.blocks`, see optsplit.models.layer): every part keeps its group's hyperparameters, names its layer in
    `block` and starts at `lr_mult` 1."""
    split = balance_cfg(cfg).blocks
    name = {p: n for n, p in model.named_parameters()}
    groups, blocks = [], []
    for g in opt.param_groups:
        hyper = {k: v for k, v in g.items() if k not in ('params', 'base_lr')}
        for b in dict.fromkeys(layer(name[p], split) for p in g['params']):
            groups.append({**hyper, 'params': [p for p in g['params'] if layer(name[p], split) == b]})
            blocks.append(b)
    opt = type(opt)(groups)
    for g, b in zip(opt.param_groups, blocks):
        g['base_lr'], g['lr_mult'], g['block'] = g['lr'], 1.0, b
    return opt


def set_lr(opt, mult):
    """optsplit.optim.set_lr with each group's multiplier."""
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * g.get('lr_mult', 1.0) * mult
