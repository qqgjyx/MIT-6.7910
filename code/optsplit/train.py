"""Train one run: python -m optsplit.train <config.yaml> [key.sub=value ...]

The run directory is $OPTSPLIT_ROOT/runs/<suite>/<name>. If it holds latest.pt the run resumes; if
`fork.parent` is set it starts from the parent's fork checkpoint (see DESIGN.md)."""

import contextlib
import json
import math
import os
import signal
import sys
import time

import torch

from .config import load_config, run_dir, save_config, step_grid
from .data import build_data
from .measure import measure_step
from .measure.step_sharpness import StepSharpness
from .models import build_model
from .optim import Ghost, Shadow, build_optimizer, load_optimizer_state, lr_mult, set_lr
from .probe import Probe

INTERRUPTED = 3


def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def main(argv):
    cfg = load_config(argv[0], argv[1:])
    assert cfg.name, 'config needs a name'
    out = run_dir(f'{cfg.suite}/{cfg.name}')
    if (out / 'done').exists():
        print(f'{out} is done')
        return 0
    for sub in ('preds', 'ckpt', 'fork'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / 'config.yaml')

    if os.environ.get('CUDA_VISIBLE_DEVICES') and not torch.cuda.is_available():
        raise RuntimeError('a GPU was allocated but CUDA failed to initialize (bad node?)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = build_data(cfg, device)
    torch.manual_seed(cfg.train.init_seed)
    model = build_model(cfg, vocab_size=getattr(data, 'vocab_size', None),
                        num_classes=getattr(data, 'num_classes', None)).to(device)
    opt = build_optimizer(cfg.optimizer, model, cfg)
    shadow = Shadow(model, cfg)
    ghost = Ghost(model, cfg)  # open-loop copies (cfg.ghost), saved as sibling runs <name>__ghost_<rule>
    assert not (ghost.models and cfg.fork.parent), 'ghosts start at the initialization: not with forks'
    ghost_dirs = {}
    for rule in ghost.models:
        d = out.parent / f'{cfg.name}__ghost_{rule}'
        (d / 'ckpt').mkdir(parents=True, exist_ok=True)
        gcfg = load_config(out / 'config.yaml', [f'name={d.name}', f'optimizer={rule}', 'ghost=[]',
                                                 f'ghost_of={cfg.suite}/{cfg.name}'])
        save_config(gcfg, d / 'config.yaml')
        ghost_dirs[rule] = d
    rates = None
    if cfg.get('balance', {}).get('variant'):  # per-layer learning rates (optsplit/layerlr)
        from .layerlr.rates import LayerRates
        rates = LayerRates(cfg, opt, model, data)
    probe = Probe(cfg, data)
    T = cfg.train.max_steps

    start = 0
    if (out / 'latest.pt').exists():
        ck = torch.load(out / 'latest.pt', map_location=device)
        model.load_state_dict(ck['model'])
        load_optimizer_state(opt, ck['optimizer'])
        shadow.load(ck['shadow'])
        ghost.load(ck.get('ghost'))
        if rates:
            rates.load(ck['balance'])
        start = ck['step']
        print(f'resuming at step {start}')
    elif cfg.fork.parent:
        start = cfg.fork.step
        ck = torch.load(run_dir(cfg.fork.parent) / 'fork' / f'{start:07d}.pt', map_location=device)
        model.load_state_dict(ck['model'])
        load_optimizer_state(opt, ck['optimizer'][cfg.optimizer])
        shadow.load(ck['optimizer'])
        print(f'forked from {cfg.fork.parent} at step {start}')
    torch.save({'y': probe.targets()}, out / 'preds' / 'targets.pt')

    eval_steps = set(step_grid(cfg.log.eval_steps, T))
    ckpt_steps = set(step_grid(cfg.log.ckpt_steps, T)) | {0}
    fork_steps = set(cfg.log.fork_steps)
    m = cfg.measure
    measuring = m.hessian or m.gradcov or m.delta_loss
    measure_steps = set(step_grid(m.steps, T)) if measuring else set()
    tracker = StepSharpness(model, opt, cfg, data) if m.step_sharpness.steps else None

    log_file = open(out / 'metrics.jsonl', 'a')

    def log(kind, step, **rec):
        log_file.write(json.dumps(dict(kind=kind, step=step, **rec)) + '\n')
        log_file.flush()

    stop = []
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.append(True))

    def save_latest(step):
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         shadow=shadow.state_dicts(), ghost=ghost.state_dicts(), step=step,
                         balance=rates and rates.state_dict()), out / 'latest.pt')

    use_amp = device.type == 'cuda' and cfg.train.dtype == 'bfloat16'
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext
    train_loss = torch.compile(model.loss) if cfg.train.compile else model.loss
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = cfg.optim[cfg.optimizer].lr
    clip = cfg.train.grad_clip if cfg.train.grad_clip > 0 else math.inf
    last_save, t0 = time.time(), time.time()

    for step in range(start, T + 1):
        if step in eval_steps:
            logp, rec = probe.evaluate(model)
            torch.save({'logp': logp}, out / 'preds' / f'{step:07d}.pt')
            log('eval', step, **rec)
        if step in ckpt_steps and not (out / 'ckpt' / f'{step:07d}.pt').exists():
            atomic_save({'model': model.state_dict()}, out / 'ckpt' / f'{step:07d}.pt')
        for rule, d in ghost_dirs.items():
            if step in ckpt_steps and not (d / 'ckpt' / f'{step:07d}.pt').exists():
                atomic_save({'model': ghost.models[rule].state_dict()}, d / 'ckpt' / f'{step:07d}.pt')
        if step in fork_steps:
            atomic_save(dict(model=model.state_dict(),
                             optimizer={cfg.optimizer: opt.state_dict(), **shadow.state_dicts()}),
                        out / 'fork' / f'{step:07d}.pt')
        measure = step in measure_steps
        if measure:
            rec = measure_step(model, opt, cfg, data, step) if (m.hessian or m.gradcov) else {}
            rec['lr'] = base_lr * lr_mult(step, cfg)
            if m.delta_loss:
                loss_before = probe.train_loss(model)
        if step == T:
            if measure:
                log('measure', step, **rec)
            break

        x, y = data.get_batch(step)
        set_lr(opt, lr_mult(step, cfg))
        if rates:
            rates.before(step, log)
        probed = tracker is not None and tracker.before(step)
        with autocast():
            loss = train_loss(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        shadow.step()
        ghost.step(model, lr_mult(step, cfg))
        opt.zero_grad(set_to_none=True)
        if rates:
            rates.after(step, log)
        if probed:
            log('step_sharpness', step, **tracker.after(x, y, step))

        if measure:
            if m.delta_loss:
                rec['delta_loss'] = loss_before - probe.train_loss(model)
            log('measure', step, **rec)
        if step % cfg.log.train_every == 0:
            log('train', step, loss=loss.item(), lr=base_lr * lr_mult(step, cfg),
                grad_norm=grad_norm.item(), time=time.time() - t0)
            if not math.isfinite(loss.item()):
                print(f'diverged at step {step}')
                break
        if stop or time.time() - last_save > 60 * cfg.log.resume_minutes:
            save_latest(step + 1)
            last_save = time.time()
            if stop:
                print(f'interrupted, saved step {step + 1}')
                return INTERRUPTED

    for d in ghost_dirs.values():
        (d / 'done').touch()
    (out / 'done').touch()
    (out / 'latest.pt').unlink(missing_ok=True)
    print(f'done in {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
