"""Expand a suite of runs, write their configs and submit them as one Slurm array job.

Suite file:

    suite: ts10m_main                      # runs go to $OPTSPLIT_ROOT/runs/<suite>/<name>/
    base: ../base/tinystories_10m.yaml     # relative to the suite file
    overrides: {train.max_steps: 6000}     # dotted keys, applied to every run
    name: "{optimizer}_d{train.data_seed}" # default name template, {dotted.key} looked up in the
                                           # resolved config; floats formatted with :g, '/' -> '-'
    runs:                                  # each entry: product over its list-valued keys;
      - {optimizer: [sgd, adamw, muon], train.data_seed: [0, 1, 2]}   # an entry may set its own `name`

To give a key a list value inside `runs`, wrap it: `shadow: [[adamw, muon]]`.

A run that sets `fork.parent` starts from the parent's config.yaml instead of `base` (the parent must
have been launched), so forks inherit the parent's model, schedule and measurements.

Usage:
    python scripts/launch.py configs/suites/x.yaml [--partition normal|preemptable] [--gpu-type h200]
        [--per-gpu K] [--time HH:MM:SS] [--max-concurrent N] [--only REGEX] [--after JOBID] [--dry-run]
        [--local] [--force]

--after JOBID starts the array once job JOBID (e.g. the parents' array) has ended, so forks can be
submitted together with their parents.

normal = mit_normal_gpu (6 h per job, 2 GPUs per user); preemptable = mit_preemptable (48 h, 4 GPUs,
jobs can be killed). Both requeue: unfinished runs resume from latest.pt in the next job.
"""

import argparse
import itertools
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optsplit.config import (CODE_DIR, load_config, load_yaml, root_dir, run_dir,  # noqa: E402
                             save_config, set_dotted)

PARTITIONS = {
    'normal': dict(name='mit_normal_gpu', time='06:00:00', max_concurrent=2),
    'preemptable': dict(name='mit_preemptable', time='48:00:00', max_concurrent=4),
}


def format_name(template, cfg):
    def lookup(m):
        v = cfg
        for k in m.group(1).split('.'):
            v = v[k]
        if isinstance(v, float):
            v = f'{v:g}'
        return str(v).replace('/', '-')
    return re.sub(r'\{([^{}]+)\}', lookup, template)


def expand(suite_file):
    """-> (suite name, list of resolved run configs)."""
    suite = load_yaml(suite_file)
    base = Path(suite_file).parent / suite['base']
    runs = []
    for entry in suite['runs']:
        entry = dict(entry)
        template = entry.pop('name', suite.get('name'))
        if template is None:
            raise SystemExit(f'no name template for {entry}')
        values = [v if isinstance(v, list) else [v] for v in entry.values()]
        for combo in itertools.product(*values):
            over = {}
            for k, v in {**suite.get('overrides', {}), **dict(zip(entry, combo))}.items():
                set_dotted(over, k, v)
            parent = over.get('fork', {}).get('parent')
            # a fork starts from its parent's config, so it trains the same model on the same schedule
            if parent and not (run_dir(parent) / 'config.yaml').exists():
                raise SystemExit(f'launch the parent {parent} first')
            cfg = load_config(run_dir(parent) / 'config.yaml' if parent else base, over)
            cfg.suite = suite['suite']
            cfg.name = format_name(template, cfg)
            runs.append(cfg)
    names = [c.name for c in runs]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise SystemExit(f'duplicate run names: {dups}')
    return suite['suite'], runs


def main(argv=None):
    ap = argparse.ArgumentParser(description='Submit a suite of runs as a Slurm array job.')
    ap.add_argument('suite_file')
    ap.add_argument('--partition', choices=list(PARTITIONS), default='normal')
    ap.add_argument('--gpu-type', choices=['any', 'h100', 'h200', 'b200', 'l40s', 'a100'], default='any')
    ap.add_argument('--per-gpu', type=int, default=1, help='runs sharing one GPU, run concurrently')
    ap.add_argument('--time')
    ap.add_argument('--max-concurrent', type=int)
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--mem', default='64G')
    ap.add_argument('--only', help='regex on run names')
    ap.add_argument('--after', help='job id to wait for (the parents of forks); their fork checkpoints '
                    'are not checked then')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--local', action='store_true', help='run sequentially here instead of submitting')
    ap.add_argument('--force', action='store_true', help='overwrite configs that changed')
    args = ap.parse_args(argv)

    suite, runs = expand(args.suite_file)
    if args.only:
        runs = [c for c in runs if re.search(args.only, c.name)]

    todo = []
    for cfg in runs:
        d = run_dir(f'{suite}/{cfg.name}')
        if (d / 'done').exists():
            print(f'done      {cfg.name}')
            continue
        if cfg.fork.parent is not None:
            parent = run_dir(cfg.fork.parent) / 'fork' / f'{cfg.fork.step:07d}.pt'
            if not parent.exists() and not args.after:
                print(f'WARNING: skipping {cfg.name}: missing {parent}')
                continue
        path = d / 'config.yaml'
        if path.exists() and load_yaml(path) != cfg and not args.force:
            raise SystemExit(f'{path} exists with different content (use --force to overwrite)')
        todo.append((cfg, path))
        print(f'todo      {cfg.name}')
    if not todo:
        print('nothing to run')
        return
    if not args.dry_run:
        for cfg, path in todo:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_config(cfg, path)

    env = {**os.environ, 'OPTSPLIT_ROOT': str(root_dir()), 'OPTSPLIT_CODE': str(CODE_DIR)}
    if args.local:
        for _, path in todo:
            cmd = [sys.executable, '-m', 'optsplit.train', str(path)]
            print(' '.join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=CODE_DIR, env=env)
        return

    part = PARTITIONS[args.partition]
    jobs = root_dir() / 'jobs' / suite
    list_file = jobs / f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}.txt'
    n_tasks = math.ceil(len(todo) / args.per_gpu)
    gres = 'gpu:1' if args.gpu_type == 'any' else f'gpu:{args.gpu_type}:1'
    cmd = ['sbatch', f'--partition={part["name"]}', f'--time={args.time or part["time"]}',
           f'--array=0-{n_tasks - 1}%{args.max_concurrent or part["max_concurrent"]}',
           f'--gres={gres}', f'--cpus-per-task={args.cpus}', f'--mem={args.mem}',
           f'--job-name={suite}', f'--output={jobs}/logs/%A_%a.out', '--requeue',
           *([f'--dependency=afterany:{args.after}'] if args.after else []),
           str(CODE_DIR / 'slurm' / 'run_array.sbatch'), str(list_file), str(args.per_gpu)]
    print(' '.join(cmd))
    if args.dry_run:
        return
    (jobs / 'logs').mkdir(parents=True, exist_ok=True)
    list_file.write_text(''.join(f'{path}\n' for _, path in todo))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
