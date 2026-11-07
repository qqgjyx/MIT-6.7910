# Experiment-side response, R1 (2026-11-06)

> Experiment-side agent (Cannon). Cross-linked from `results/ii_iii/LOG.md`, entry "R1, exp-side ball-kick to the
> paper side". Paper side: the agent on Juntang's laptop, editing `results/next_step_ii_iii.tex`.

## Headline

The experiments for (ii) are done and stable over data orders 0 to 2; (iii) is running. The draft
`results/next_step_ii_iii.tex` has the title block, Summary, Setup, section 1 (reproduction) and section 2 (the
rules without learning). Sections 3 (ghosts) and 4 (C ablations) are to be written from the numbers below. Two
framing decisions are already made with Juntang: (a) section 3 is a 2 x 2 design (rule x gradient stream), not a
"rule or training" dichotomy; (b) the C3 knockout only shows orthogonalization is necessary, not why.

## Per-item status

| item | status | output | notes |
|:--|:--|:--|:--|
| 1. Reproduction | DONE | in the tex (Table 1) | gate passed |
| 2. Rules without learning (B) | DONE | `results/ii_iii/figures/rule_noise.pdf`, in the tex | |
| 3. Ghosts (A), d0-d2 | DONE, not yet written | `results/ii_iii/figures/ghost.pdf` | numbers below |
| 4. C ablations, final loss | DONE, not yet written | numbers below | spectra of C runs not yet computed |
| 5. (iii) dose-response, `sv_power` p in {0, .25, .5, .75} | IN-PROGRESS | Cannon jobs 58519353, 58519359 | about 23:30 |
| 6. (iii) swap timescale | IN-PROGRESS | ORCD jobs 28163938, 28163942, 28163943 | H200, its own comparison set |
| 7. (iii) E, per-step loss decomposition | IN-PROGRESS | ORCD job 28163949 | |
| 8. F, attention rebalancing | DEFERRED | | only if time allows |

## Numbers for the paper side (with provenance)

Paths on Cannon, `$OPTSPLIT_ROOT` (lab netscratch).

**Section 3, ghosts.** Source: `$OPTSPLIT_ROOT/analysis/ii_iii/ghost/ghost.json` (`optsplit/analysis/ghost.py`).
Mean over the 36 block matrices; range over data orders 0 to 2.

| | step 1000: stable rank / sigma1/sigma_median | step 10000: stable rank / sigma1/sigma_median / update cosine |
|:--|:--|:--|
| AdamW run | 30.9-31.2 / 8.11-8.20 | 39.9-41.4 / 5.80-6.19 / 0.89-0.90 |
| Muon rule on AdamW's gradients | 98.4-98.6 / 2.61-2.62 | 87.0-87.2 / 2.77-2.80 / 0.70 |
| Muon run | 85.1-85.8 / 2.69-2.70 | 75.3-75.5 / 2.89-2.90 / 0.71 |
| AdamW rule on Muon's gradients | 12.1-12.4 / 16.2 | 12.6-13.0 / 15.6-16.0 / 0.89 |

The 2 x 2 at step 10000 (stable rank): rows = rule, columns = gradient stream.

| | AdamW's gradients | Muon's gradients |
|:--|--:|--:|
| Muon's rule | 87 (ghost) | 75 (run) |
| AdamW's rule | 40 (run) | 13 (ghost) |

Spectral norm does not follow the rule: Muon ghost 3.96 (step 1000) / 10.88 (step 10000), AdamW ghost 7.84 / 7.75,
runs: Muon 3.18 / 7.46, AdamW 4.68 / 5.91. Runs carrying a ghost are identical to runs without one (same final
validation loss and spectra), and a ghost of the run's own rule is bit-identical to the run (`tests/test_ghost.py`,
and checked on GPU).

**Section 4, C ablations, final validation loss.** Source: `kind=eval` records at step 10000 of
`$OPTSPLIT_ROOT/runs/ts10m_ii/<run>/metrics.jsonl`; baselines from `$OPTSPLIT_ROOT/runs/ts10m_traj/`.

| run | d0 | d1 | d2 |
|:--|--:|--:|--:|
| AdamW (baseline) | 1.339 | 1.341 | 1.345 |
| AdamW, weight decay 0 | 1.346 | 1.353 | 1.350 |
| Muon (baseline) | 1.308 | 1.309 | 1.310 |
| Muon, weight decay 0.02 (same shrink per step as AdamW) | 1.297 | 1.298 | 1.304 |
| Muon without orthogonalization, lr 0.01 | 3.729 | 2.943 | 1.389 |
| same, lr 0.003 | 1.454 | | |
| same, lr 0.03 | 4.078 | | |

Muon with weight decay reaches AdamW's final validation loss at step 7750 on d0, as plain Muon does.

## Surfaces for the paper side

- A frozen-initialization ghost (B with real gradients) was run and dropped: at fixed weights the gradient barely
  changes (update cosine 0.98), so the steps pile up in one direction. Do not report it.
- The cosine in section 2 is "0.66 to 0.76" for Muon because it depends on the matrix shape (attention 0.66, MLP
  0.75 to 0.76).
- No em-dashes anywhere; plain, concrete style of `checkpoint/` (Juntang's rule).
- Cannon's TeX lacks `cleveref`; the draft uses `\ref`. On the laptop, switching to `\cref` like the checkpoint is
  fine.

## Action items now on the paper side

1. Write section 3 (ghosts): the test (a copy updated by the other rule on the run's gradients, no feedback), the
   2 x 2 table, `figures/ghost.pdf`, and what it does not show (the ghost is not a trained model: no loss, Hessian
   or activations for it).
2. Write section 4 (C): weight decay is not the reason; C3 shows orthogonalization is necessary, not why.
3. Leave a section 5 placeholder for (iii): dose-response, swap timescale, E.
4. Keep the Summary in step with sections 3 and 4.

## Prompt for the paper-side agent (copy this when picking up the ball)

> You are the paper-side agent for the MIT 6.7910 project "How AdamW and Muon Train a Small Language Model"
> (Juntang Wang and Victor Conchello), repo `MIT-6.7910`, branch `ii-iii-fasrc`. Pull the branch, then read
> `results/ii_iii/experiment_response_R1.md` end-to-end before any compile; also read `checkpoint/checkpoint.pdf`
> (the earlier report) and `results/ii_iii/LOG.md` (the experiment log; the ledger).
>
> - You own `results/next_step_ii_iii.tex`. Its figures are in `results/ii_iii/figures/` and are produced on the
>   experiment side; do not regenerate or edit them. Never edit `code/` or `results/ii_iii/LOG.md` above the
>   ledger entries.
> - Do the "Action items" in order, one section at a time, and show Juntang each section before the next (he
>   prefers small steps).
> - Every number you write must come from this response file or an earlier section of the tex. A number you need
>   and do not have is BLOCKED: list it for the experiment side, do not estimate it.
> - Style: the plain, concrete style of `checkpoint/checkpoint.tex`; no em-dashes anywhere.
> - Compile locally (pdflatex twice). Do not commit, push or open a PR without Juntang's explicit approval.
> - When done, append a one-line ledger entry at the end of `results/ii_iii/LOG.md`:
>   `### R1, paper side: <what was integrated> (appended <date>, agent PAPER-LAPTOP)`. For new asks of the
>   experiment side, write `results/ii_iii/paper_ballkick_R2.md` (Template A of `juntang-paper-coordination`) and
>   cross-link it from the ledger.
