# optsplit: when do optimizers start to learn different functions?

Code for the Topic 3 project (`../project_plan/project_plan.pdf`). SGD, AdamW and Muon train nanoGPT
models from the same initialization and data order; we log their predictions on fixed held-out
contexts, locate the split point t*, track sharpness and the anatomy of the top Hessian eigenvector,
swap optimizers at several times, and compare function metrics. A SimpleCNN on CIFAR-10 replicates
Jastrzebski et al. (2020) as a check of the pipeline. Interfaces and file formats are in `DESIGN.md`.

## Layout

| Path | What |
|:--|:--|
| `optsplit/models/` | nanoGPT (adapted: separate q/k/v, untied head, no dropout) and SimpleCNN |
| `optsplit/optim/` | SGD, AdamW, Muon (vendored), LR schedule, shadow optimizer states |
| `optsplit/data.py`, `probe.py` | deterministic batch order; fixed probe sets for predictions and losses |
| `optsplit/train.py` | training loop: logging, checkpoints, forks, resume after preemption |
| `optsplit/measure/` | in-training: Lanczos sharpness, AdamW-preconditioned sharpness, batch sharpness, eigenvector anatomy, gradient covariance K |
| `optsplit/metrics/` | offline, on checkpoints: input-derivative metrics (1st to 3rd order), weight norms, robustness |
| `optsplit/analysis/` | function distance, split location, LMC, swaps, figures, break-even replication |
| `configs/` | `defaults.yaml`, model/data configs in `base/`, run suites in `suites/` |
| `data/` | dataset preparation scripts |
| `scripts/launch.py`, `slurm/` | submit suites as Slurm arrays on ORCD (`mit_normal_gpu` or `mit_preemptable`) |
| `tests/` | unit tests (CPU): `python -m pytest tests -q` |

## Setup on ORCD (once)

```bash
git clone https://github.com/qqgjyx/MIT-6.7910.git && cd MIT-6.7910/code
bash slurm/setup_env.sh        # conda env with torch (cu128) at $OPTSPLIT_ROOT/env
source slurm/env.sh            # in every new shell
python data/prepare_shakespeare_char.py
python data/prepare_cifar10.py
sbatch slurm/prepare_data.sbatch tinystories   # ~2 GB download + BPE tokenizer, on a CPU node
sbatch slurm/test_gpu.sbatch                   # GPU sanity check
```

`OPTSPLIT_ROOT` (data, runs, job logs) defaults to `optsplit/` in your pool storage
(`/orcd/pool/<NNN>/$USER`, 1 TB). If compute nodes have no internet access, run the TinyStories
preparation on the login node instead.

## Running

A suite is a YAML file of runs (format in `scripts/launch.py`). Submit it with

```bash
python scripts/launch.py configs/suites/<suite>.yaml [--partition normal|preemptable] [--per-gpu K] [--gpu-type h200] [--dry-run]
```

- `normal` = `mit_normal_gpu`: at most 2 GPUs at once, 6 h per job, never killed.
- `preemptable` = `mit_preemptable`: at most 4 GPUs, 48 h per job, can be killed.

Runs save `latest.pt` every 15 min and on SIGTERM. Jobs requeue themselves at the time limit or after
preemption, and the runs resume exactly where they stopped. Launching a suite again skips finished runs.
`--per-gpu 2` or `3` packs small runs (Shakespeare, 10M) on one GPU. Check progress with
`python -m optsplit.analysis.table '<suite>/*'`.

Order of work, per setting (`shakespeare`, then `ts10m`, `ts30m`):

1. `smoke.yaml`, then `smoke_forks.yaml`: a few minutes; checks the whole pipeline.
2. `<setting>_lrsweep.yaml`: pick each optimizer's learning rate with
   `python -m optsplit.analysis.table '<setting>_lrsweep/*' --sort val_loss`, and write it into
   `<setting>_main.yaml`.
3. `<setting>_main.yaml`: 3 optimizers × 5 data orders. The `d0` runs keep the other optimizers' states
   in shadow and save fork checkpoints.
4. `<setting>_forks.yaml` (after the `d0` runs are done): every ordered swap at 9 fork steps, plus
   same-optimizer forks with another data order.
5. Offline metrics: `sbatch slurm/cmd.sbatch python -m optsplit.metrics.run '<setting>_main/*_d0' '<setting>_forks/*' --steps all`.
6. Analysis (CPU is fine, except LMC):

```bash
python -m optsplit.analysis.split --runs-a 'ts10m_main/adamw_d*' --runs-b 'ts10m_main/muon_d*' --out split_adamw_muon.json
python -m optsplit.analysis.plots divergence --runs-a 'ts10m_main/adamw_d*' --runs-b 'ts10m_main/muon_d*' --split split_adamw_muon.json
python -m optsplit.analysis.plots sharpness --runs 'ts10m_main/*_d0' --split split_adamw_muon.json
python -m optsplit.analysis.plots anatomy --runs 'ts10m_main/*_d0' [--precond] [--group layer]
python -m optsplit.analysis.swap --forks 'ts10m_forks/*' --pure 'ts10m_main/*' --out swap
python -m optsplit.analysis.plots swap --swap-json swap.json --split split_adamw_muon.json
sbatch slurm/cmd.sbatch python -m optsplit.analysis.lmc --parent ts10m_main/adamw_d0 --forks 'ts10m_forks/adamw_to_*' --out lmc_adamw.json
python -m optsplit.analysis.plots lmc --lmc-json lmc_adamw.json
python -m optsplit.analysis.plots function --runs 'ts10m_main/*_d0' --keys input.hess_top.mean weights.dist_init robustness.sensitivity
```

Figures go to `$OPTSPLIT_ROOT/analysis/figures/` unless `--out` is given.

## Break-even replication (Jastrzebski et al., 2020)

The paper trains with plain SGD (no momentum) at a constant learning rate and compares learning rates
and batch sizes. It tracks the Hessian's top eigenvalue (Lanczos on 5% of CIFAR-10), the gradient
covariance K from L = 25 mini-batch gradients (top eigenvalue, smallest non-zero eigenvalue, trace,
‖g‖/‖g₅‖), the loss change over one step, and a UMAP of the test-prediction trajectories. Suites:

- `cifar_breakeven.yaml`: SimpleCNN (ReLU, as in the paper), lr 0.001/0.01/0.1 at batch 128, and
  batch 30/300 at lr 0.05.
- `shakespeare_breakeven.yaml`, `ts10m_breakeven.yaml`: the same protocol on our transformers.

Figures: `python -m optsplit.analysis.breakeven --runs 'cifar_breakeven/sgd_lr*_d0'`. The split analysis
between two learning rates (`--runs-a '..._lr0.01_d*' --runs-b '..._lr0.1_d*'`) gives the paper's
separation of trajectories in prediction space.

## Where each part of the plan lives

| Plan | Code |
|:--|:--|
| Divergence curves (equal steps and equal loss) | `analysis/distance.py`, `plots divergence` |
| Split point t* with uncertainty (noise floor, bootstrap over seeds) | `analysis/split.py` |
| Basin-level test: LMC barrier vs fork time | `analysis/lmc.py`, `plots lmc` |
| What happens at the split: λ_max, preconditioned sharpness (AdamW), batch sharpness, end of warmup, loss level | `measure/`, `plots sharpness` |
| What happens at the split: function metrics of the previous plan (input Jacobian, Hessian norm and top eigenvalue, level-set curvature, third order; weight norms; robustness and Hahn sensitivity) | `metrics/`, `plots function` |
| Swaps with warm shadow state, A-continued control | `optim.Shadow`, `fork.*` in configs, `analysis/swap.py` |
| Moonshot: anatomy of the top eigenvector by block (QK, OV, MLP per layer, embeddings, norms, head) | `measure/hessian.py:anatomy`, `plots anatomy` |

## Choices to know about

- **Hyperparameters.** AdamW uses β = (0.9, 0.95) and weight decay 0.1. Muon uses momentum 0.95 and
  no weight decay, with AdamW on embeddings, head and gains. SGD uses momentum 0.9. Clipping is off,
  because it hides edge-of-stability dynamics. Warmup is followed by cosine decay to 10%.
- **Anatomy.** For an exact eigenvector, `v_bᵀ(Hv)_b` is just `λ‖v_b‖²`. So the sharpness is split
  instead into each block's own curvature `v_bᵀH_bb v_b` plus a cross-block remainder.
- **Batch order.** It depends only on `(data_seed, step)`. Forks and resumed runs see exactly the
  batches the parent would have. The noise floor uses the same initialization with a different data
  order.
- **Tokenizer.** TinyStories uses a 4096-token BPE, so the head stays small and the stored next-token
  distributions (2048 positions × 4096 per evaluation) stay manageable.
