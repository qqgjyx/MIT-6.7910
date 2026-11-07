# Next steps (ii) and (iii) on FAS RC Cannon: log

Owner: Juntang. Branch `ii-iii-fasrc`. Newest entries at the bottom of each section. Written so a fresh agent can
resume from here.

## Where things are

- Code: this repo, `code/`. Cluster profile: `export OPTSPLIT_CLUSTER=fasrc OPTSPLIT_ROOT=<root>` then
  `source code/slurm/env.sh`.
- `OPTSPLIT_ROOT` on Cannon: lab netscratch (`<netscratch>/optsplit`; exact path in Juntang's own notes) (env, data, runs, jobs, analysis).
  Netscratch is purged after 90 days without access (FASRC policy); everything there can be regenerated from this
  repo. Lab holylabs storage was over quota on 2026-11-06 (`mkdir` failed).
- Env: uv venv at `$OPTSPLIT_ROOT/env`, Python 3.11.16, torch cu128. uv cache in `~/.cache/uv` with
  `UV_LINK_MODE=copy` (a uv cache on netscratch hung on its lock).
- Victor's ORCD runs (`/orcd/pool/014/conchell/optsplit`) are not readable from Cannon. Compare new runs with new
  runs only; the checkpoint's numbers are a reproduction gate, nothing more.

## Cannon facts (probed 2026-11-06 from a login node)

- Slurm association: account `rongma_lab` only (`sacctmgr show assoc user=$USER`), QOS normal, FairShare 0.055
  (low). `hsph_users` is a Unix group, not a Slurm account: it gives access to partition `hsph_gpu`
  (AllowGroups=hsph_users), but jobs there are still billed to `rongma_lab`.
- GPU partitions (3 days max unless noted): `gpu` (36 nodes, 4x A100 80GB, ~730 jobs pending), `gpu_h200` (22 nodes,
  ~90 pending), `gpu_requeue` (432 nodes, mixed: A100 40/80GB, H100, H200, A40, RTX A6000, RTX Pro 6000, and A100 MIG
  slices down to 1g.5gb; PreemptMode=REQUEUE; ~1400 pending), `gpu_test` (12 h, A100 MIG 3g.20gb slices only),
  `hsph_gpu` (2 nodes, 4x H200, ~14 pending). Node features `a100`, `h100`, `h200`, `a100-mig` select the GPU.
- Storage: `$HOME` 95G (15G used). `/n/netscratch` 85% full overall.

## Port (Phase 1)

`OPTSPLIT_CLUSTER=orcd|fasrc` (default orcd, so ORCD behaviour is unchanged):
- `scripts/launch.py`: `CLUSTERS` dict replaces `PARTITIONS`. fasrc partitions: normal=`gpu`,
  preemptable=`gpu_requeue`, test=`gpu_test`, hsph=`hsph_gpu`. On fasrc, `--gpu-type` becomes `--constraint`
  (default per partition: a100 / a100|h100|h200 / a100-mig / h200), so jobs never land on small MIG slices.
- `slurm/env.sh`, `slurm/setup_env.sh`: fasrc branch uses a uv venv at `$OPTSPLIT_ROOT/env` and requires
  `OPTSPLIT_ROOT` to be set (no user paths in the repo).
- `tests/test_launch.py`: fixture clears `OPTSPLIT_CLUSTER`; new `test_dry_run_fasrc`.
- Single-job sbatch files keep their ORCD headers; override on the command line (`sbatch -p gpu_test ...`).

## Log

### 2026-11-06
- Branch `ii-iii-fasrc` from `main` at e5c48e7.
- Env built (see above).
- Shakespeare text for the toy model: `$OPTSPLIT_ROOT/data/raw/input.txt` (md5 6fb458f1232090904fb40fe944165e91).
- Job 58249600 (test, CPU): `python -m pytest tests -q`: 66 passed (4 min on 16 CPUs). Then toy_model.py and
  linear_control.py to `$OPTSPLIT_ROOT/analysis/phase0/{toy,linear}` (logs `*.log` next to them).
- Job 58249627 (gpu_test, MIG 3g.20gb): `sbatch -p gpu_test --gres=gpu:1 slurm/test_gpu.sbatch`: torch 2.11.0+cu128,
  bf16 matmul 112 TFLOP/s, math-SDPA double backward ok.
- Shakespeare: `python data/prepare_shakespeare_char.py` (login node): 65 chars, 1003854 / 111540 tokens.
- Job 58249643 (gpu_test): `python scripts/launch.py configs/suites/smoke.yaml --partition test --per-gpu 6 --mem 32G`
  (gpu_test allows 2 submitted jobs per user, so runs are packed). 6/6 done. Final train/val at step 300: AdamW
  2.35/2.37, Muon 1.57/1.74, SGD 2.67/2.69 (d0) and 2.43/2.46 (d1).
- Job 58249878 (gpu_test): smoke_forks, `--per-gpu 4`: 6/6 done.
- Job 58249652 (test, CPU): `sbatch -p test slurm/prepare_data.sbatch tinystories`: 4096-token BPE (checked:
  `meta.json` vocab_size 4096, byte-level tokens), 552,521,065 train and 5,578,016 val tokens. About 4 minutes.
- linear_control.py (default args, 3.5 min CPU): matches `checkpoint/figures/linear/linear_control.json` to 1e-14
  (persistent split at loss 7.49e-4). Walkthrough section 9 reproduced.
- toy_model.py (default args, 10.5 min CPU): same lrs (SGD 0.3, AdamW 0.01, Muon 0.03); closest approach to unigram
  0.0360 / 0.0640 / 0.1817 bits (SGD / AdamW / Muon); end AdamW-Muon same 0.00105, other 0.00679. 156 of 399 numbers
  bit-identical to `checkpoint/figures/toy/toy_model.json`, the rest within 4%, all involving Muon (Newton-Schulz
  amplifies BLAS differences, Mac vs Linux). Walkthrough section 12 reproduced at the quoted precision.
- GATE A reached. Design draft: `results/ii_iii/design.md`. Waiting for Juntang's OK on account/partition and design.

### 2026-11-06, after Gate A
- Juntang: "proceed working!" Taken as OK of the Gate A recommendations: gpu_requeue on A100
  (`--partition preemptable --gpu-type a100`), account rongma_lab, design A-E, F undecided, SGD only as the 0.06
  reference in ts10m_traj.
- Phase 2 launched (from `code/`, env as above):
  - `python scripts/launch.py configs/suites/ts10m_main.yaml --only "^(adamw|muon)_" --partition preemptable --gpu-type a100 --per-gpu 2`
    -> job 58357338 (10 runs in 5 tasks). SGD skipped (lr 0.03, not needed for (ii)/(iii)).
  - `python scripts/launch.py configs/suites/ts10m_traj.yaml --partition preemptable --gpu-type a100 --per-gpu 2`
    -> job 58357346 (9 runs: SGD 0.06, AdamW, Muon x d0-d2).
- Code for (ii) (uncommitted): `optim.Ghost` + `ghost` config key (train.py), Muon `orthogonalize` option,
  `tests/test_ghost.py` (6 tests: own-rule ghost is bit-identical to the run; the ghost does not change the run;
  no-orth update norm and direction; end-to-end train writes `<name>__ghost_<rule>/`). DESIGN.md documents both.
- GPU check of the ghost (scratch suite smoke_ghost, gpu_test, bf16 autocast, 300 steps): the own-rule ghost is
  bit-identical to the run for AdamW and Muon. Caution for analysis: `<suite>/adamw_ghost_d*` also matches the ghost
  directories `adamw_ghost_d0__ghost_muon`; use explicit patterns.
- New suite `configs/suites/ts10m_ii.yaml` (A ghosts, B frozen-init ghosts, C1-C3), 18 runs. NOT launched yet:
  held until the reproduction gate passes.
- B, noise baseline: `scripts/rule_noise.py`, job 58358171 (test, CPU) -> `$OPTSPLIT_ROOT/analysis/ii_iii/rule_noise`.
  30-step check: on the low-rank+noise stream AdamW's MLP-out stable rank falls from 176 to 21 while Muon's stays at
  176; on pure noise both keep the init's spectrum.
- E: reuse ts10m_stoch_eos (`adamw_cosine`, `muon_cosine`: slope and curvature along every 10th step; analysis
  `optsplit/analysis/stochastic_eos.py`). Not launched yet.
- B noise baseline done (job 58358171, 8 min on 16 CPUs; `analysis/ii_iii/rule_noise/rule_noise.json`). One block
  matrix of each shape from the real initialization, 10k steps, the main runs' lrs and schedule, AdamW wd 0.1, Muon none.
  - Cosine of consecutive updates, pure noise, end of training: Muon 0.66 (attn) / 0.76 (MLP), AdamW 0.90 for all
    shapes. Real runs at loss 1.81 (checkpoint table): Muon 0.738, AdamW 0.883. The gap is there with no signal at
    all, so this property looks rule-driven (momentum + normalization), consistent with the swaps.
  - Pure noise: stable rank stays at the init's value for both rules (attn about 98, MLP about 174). The rules
    alone do not make W's spectrum differ.
  - Fixed rank-4 signal + equal-norm noise: both collapse to stable rank about 3.5 by step 10k, but Muon much
    more slowly (step 1000: Muon 9-10, AdamW 3.6). Orthogonalization gives the noise directions the same step as the
    signal directions, so W's spectrum stays flatter for longer. Spectral norms here are not comparable to the real
    runs (on this toy stream AdamW's sign-like update has about 4x Muon's Frobenius norm).
  - Reading: the stable-rank gap needs both the rule and low-rank structure in the gradients. Whether real gradient
    streams give Muon-like spectra under Muon's rule without feedback is what A (ghosts) tests.

### 2026-11-06, evening: move to gpu_test
- gpu_requeue arrays sat on Priority for hours (FairShare 0.08). gpu_test has free MIG slices, billing weight 0,
  QOS: 2 jobs and 8 GPUs (MIG slices), 64 CPUs per user, 12 h. New `launch.py --gpus N`: one task holds N GPUs and
  `run_array.sbatch` gives each run one of them (round robin over CUDA_VISIBLE_DEVICES, which holds MIG UUIDs).
  Checked in job 58439595: 2 slices, one per process, both work.
- Cancelled 58357346 (ts10m_traj on gpu_requeue, never started). 58357338 (ts10m_main) left pending as backup.
- Job 58439753 (gpu_test, 3 slices, 2 runs each):
  `python scripts/launch.py configs/suites/ts10m_traj.yaml --only "^(adamw|muon)_" --partition test --gpus 3 --per-gpu 2 --cpus 24 --mem 96G`
- Job 58439755 (gpu_test, 4 slices, 2 runs each), started before the gate passes to save time (discard if it fails):
  `python scripts/launch.py configs/suites/ts10m_ii.yaml --only "^(adamw|muon)_ghost_d|^(adamw_wd0|muon_noorth)_d0$" --partition test --gpus 4 --per-gpu 2 --cpus 32 --mem 128G`
- All (ii)/(iii) comparisons are now on A100 MIG 3g.20gb slices. Do not mix with full-A100 runs.
- First gpu_test submissions (58439753, 58439755, 58440522) failed with CUDA OOM: two runs per 20 GB slice do not
  fit, and AdamW alone OOMed in batch_sharpness (fp32 double backward on 128 x 256 tokens, ~17 GB) from
  fragmentation. Fix: `slurm/env.sh` (fasrc) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; one run per
  slice (`--gpus N --per-gpu 1`). 58440521 requeued itself on scancel and restarted with the fix.
- Job 58440521 (gpu_test, 6 slices): ts10m_traj AdamW/Muon d0-d2 all done at 18:54 (AdamW ~26 min, Muon ~16 min).
- Job 58444564 (gpu_test, 2 slices): ts10m_ii adamw_ghost_d0, muon_ghost_d0.
- Cancelled 58357338 (ts10m_main on gpu_requeue, never started): the gate uses ts10m_traj.
- REPRODUCTION GATE, validation part (eval every 250 steps, `kind=eval` records of metrics.jsonl):
  final val AdamW 1.339 / 1.341 / 1.345, Muon 1.308 / 1.309 / 1.310 (d0/d1/d2); Muon first reaches AdamW's final
  val at step 7750 / 7500 / 7250. Checkpoint: 1.344, 1.309, step 7500. Passes.
- Job 58451653 (gpu_test, 6 slices): `metrics.trajectory --steps all` on the six ts10m_traj runs, script
  `$OPTSPLIT_ROOT/jobs/ii_iii/traj_metrics.sbatch` (one run per slice). For stable rank / active units at loss 1.81.

### 2026-11-06, ORCD set up as a second cluster
- Juntang got ORCD access. Jobs are issued from Cannon over an SSH ControlMaster connection that Juntang
  opens (Duo once per 12 h); the SSH setup is kept outside the repo.
- ORCD: account `mit_general` must be given explicitly (`export SBATCH_ACCOUNT=mit_general`). mit_normal_gpu: QOS
  2 GPUs per user and an estimated start days away; mit_preemptable starts at once. Storage
  `/orcd/pool/<NNN>/$USER/optsplit` (1 TB). Victor's `/orcd/pool/014/conchell/optsplit`: permission denied.
- `~/dev/MIT-7910` on ORCD (clean clone at e5c48e7): branch `ii-iii-fasrc` with this working tree rsynced
  (identical `git status`). Env: `bash slurm/setup_env.sh` (conda, torch 2.11.0+cu128). Data: prepared TinyStories
  and Shakespeare rsynced from Cannon (md5 of val.bin and meta.json checked).
- Job 28147133 (mit_preemptable, H200): test_gpu ok, bf16 matmul 781 TFLOP/s, double backward ok.
- ORCD runs form their own comparison set (H200): never mix them with Cannon MIG runs.

### 2026-11-06, gate passed; first ghost result
- Job 58451653 (trajectory metrics, 3 min on 6 slices), then on a CPU node:
  `python -m optsplit.analysis.rank_metrics 'ts10m_traj/adamw_d*' 'ts10m_traj/muon_d*' --out $OPTSPLIT_ROOT/analysis/ii_iii/traj`
  and `python optsplit/analysis/metric_ratios.py <out>/metrics.json A 1.81`. At training loss 1.803: stable rank
  AdamW 32.7 +- 0.2, Muon 86.3 +- 0.5; active MLP units 0.120 / 0.222; sigma1/sigma_median 8.07 / 2.68; lambda_max
  6.14 / 14.3; cosine of consecutive steps 0.886 / 0.738. All 30 rows agree with checkpoint Table 3 within the
  data-order ranges (batch sharpness noisier: 0.82 / 3.37 against 0.97 / 3.13). REPRODUCTION GATE PASSED.
- A, d0 (job 58444564): new `optsplit/analysis/ghost.py`, on a CPU node:
  `python -m optsplit.analysis.ghost ts10m_ii/adamw_ghost_d0:ts10m_traj/muon_d0 ts10m_ii/muon_ghost_d0:ts10m_traj/adamw_d0 --out $OPTSPLIT_ROOT/analysis/ii_iii/ghost`
  Block matrices, mean over 36, at step 1000 / 10000:
  - Muon ghost on AdamW's gradients: stable rank 98.6 / 87.2 (AdamW run 31.2 / 41.4, Muon run 85.8 / 75.3);
    sigma1/sigma_median 2.61 / 2.77 (AdamW 8.11 / 5.80, Muon 2.69 / 2.90); update cosine 0.70 (AdamW 0.88-0.90,
    Muon 0.69-0.71).
  - AdamW ghost on Muon's gradients: stable rank 12.2 / 12.8; sigma1/sigma_median 16.2 / 16.0; update cosine
    0.87-0.90.
  - Spectral norm does not follow the rule: Muon ghost 3.96 / 10.88, AdamW ghost 7.84 / 7.75, both above their
    closed-loop counterparts (Muon 3.18 / 7.46, AdamW 4.68 / 5.91).
  - Reading: stable rank, sigma1/sigma_median and update cosine are set by the rule applied to either optimizer's
    real gradient stream (rule-driven). The ghosts are more extreme than the closed-loop runs, so the feedback
    pulls both runs toward each other. One data order so far; d1, d2 running.
  - Not shown: whether the rule's spectra matter for the loss (ghosts are not trained models). That is (iii).
- Launched on gpu_test (4 slices each): 58457614 (ghosts d1, d2), 58457619 (adamw_wd0_d0, muon_wd_d0,
  muon_noorth_d0, frozen_ghost_d0).

### 2026-11-06, 21:30: A on three data orders, first C runs
- All of 58457614 / 58457619 done by about 19:45. ghost.py on all pairs (same command, d0-d2 plus
  `ts10m_ii/frozen_ghost_d0:ts10m_traj/muon_d0`). Ranges over d0-d2, step 10000:
  - AdamW run: stable rank 39.9-41.4, sigma1/sigma_median 5.80-6.19, update cosine 0.89-0.90.
  - Muon ghost on AdamW's gradients: 87.0-87.2, 2.77-2.80, 0.70. Muon run: 75.3-75.5, 2.89-2.90, 0.71.
  - AdamW ghost on Muon's gradients: 12.6-13.0, 15.6-16.0, 0.89. Same picture at step 1000.
  - Runs carrying a ghost are identical to the ts10m_traj runs (same final val, same spectra), as expected.
  - Frozen init (B, real gradients, no learning): gradients barely change (Muon ghost update cosine 0.98), so the
    steps add up in one direction: stable rank 85 but sigma1/sigma_median 15 and spectral norm 75. Not a useful
    "default spectrum"; drop it from the write-up.
- C, d0, final val (AdamW d0 1.339, Muon d0 1.308): AdamW wd 0 1.346; Muon wd 0.02 1.297 (reaches AdamW's final
  at 7750, like Muon); Muon without orthogonalization at lr 0.01 3.729 (fails). Its lr check (0.003, 0.03) and
  d1/d2 of C launched: jobs 58497631 (adamw_wd0, muon_wd d1-d2), 58497647 (muon_noorth d1-d2, lr 0.003 / 0.03).

### 2026-11-06, 22:30: C done; plan for (iii) revised
- C, all done. Final val: adamw_wd0 1.346 / 1.353 / 1.350; muon_wd 1.297 / 1.298 / 1.304 (d0/d1/d2);
  muon_noorth lr 0.01: 3.729 / 2.943 / 1.389 (two of three fail), lr 0.003: 1.454, lr 0.03: 4.078 (d0).
- Juntang's point: C3 is a knockout. It shows orthogonalization is needed for Muon's speed, not why, and it does
  not test whether the Step-2 properties cause the speed. Revised (iii):
  1. Timescale in the swaps (ts10m_forks_dense): after a swap the update cosine changes at once, the spectra over
     hundreds to thousands of steps. Does the loss rate change at once (per-step update) or with the spectra?
  2. Dose-response: `optim.muon.sv_power=p`, update U S^p V^T from an exact SVD, rescaled to an orthogonal
     update's Frobenius norm (p=0 exact orthogonalization, p=1 = muon_noorth). New test
     `test_muon_sv_power`. Suite `configs/suites/ts10m_svp.yaml`: p 0, 0.25, 0.5, 0.75 x lr 0.003, 0.01, d0.
  3. F (attention rebalancing by the exact symmetry) if time allows.
  E unchanged.
- Launched:
  - Cannon gpu_test: ts10m_svp, jobs 58519353 (lr 0.01) and 58519359 (lr 0.003), 4 slices each.
  - ORCD (mit_preemptable, H200, `export SBATCH_ACCOUNT=mit_general`), its own comparison set:
    28163938 ts10m_main adamw_d0, muon_d0 (swap parents, with shadows and fork checkpoints);
    28163942 ts10m_parent_late (after 28163938); 28163943 ts10m_forks_dense, 24 swaps, --per-gpu 4 (after 28163942);
    28163949 ts10m_stoch_eos adamw_cosine, muon_cosine (E).

## Ledger (paper side <-> experiment side; append only, tagged and dated)

### R1, exp-side ball-kick to the paper side (appended 2026-11-06, agent EXP-CANNON)

Full status and numbers: `results/ii_iii/experiment_response_R1.md`. Headline: (ii) done on d0-d2 (ghosts: the
spectra and update cosine follow the rule on either optimizer's gradients; C: weight decay is not the reason, C3
shows orthogonalization is necessary). Draft `results/next_step_ii_iii.tex` has sections 1-2; sections 3-4 are on
the paper side. (iii) running (dose-response on Cannon, swaps and E on ORCD).
