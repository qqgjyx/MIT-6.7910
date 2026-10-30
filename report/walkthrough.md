# Walkthrough: when do SGD, AdamW and Muon split?

**[planned]** = in the project plan. **[extra]** = added after seeing the results.
Status: TinyStories-30M swaps/LMC, function metrics and the long Shakespeare runs still running.

## TL;DR

![](figures/overview.png)

*Distance between optimizers at equal training loss ÷ distance between two data orders (red: split
threshold). Training goes left → right.*

- **No "same path, then split".** At equal training loss the optimizers differ most at the very start
  (10–200× the data-order noise), then meet again around the bigram stage and stay within ~1× of the noise.
- **Swapping AdamW for Muon changes the final function no more than reshuffling the data** (TinyStories).
- Most "separation in time" is one optimizer (or lr) being faster. Same for the break-even replication.
- **Edge of stability doesn't explain any of it.** AdamW sits at its threshold whether or not the
  optimizers differ; SGD only around the end of warmup.
- **Swaps:** whoever trains last decides the function (if it's fast enough). But in **weight space** the
  optimizer leaves a mark: same function, different basin.
- The detector does find a split where theory says there is one (linear model), but only near interpolation.

## 1. [planned] Learning-rate sweeps

| | SGD | AdamW | Muon |
|:--|:--|:--|:--|
| Shakespeare (min val) | 0.2 → 2.44 | 3e-4 → 1.54 | 0.04 → 1.56 |
| TinyStories 10M (final val) | 0.1 → 1.91 | 2e-3 → 1.34 | 0.01 → 1.30 |
| TinyStories 30M (final val) | 0.3 → 1.34 | 2e-3 → 1.20 | 0.01 → 1.18 |

- Muon < AdamW < SGD everywhere. SGD's best rate was unstable on other data orders, so it uses one grid
  step lower (0.05 / 0.03 / 0.1).
- Shakespeare overfits (no dropout). Muon memorizes fastest (train 0.19, val 2.86 at the end).

## 2. [planned] Divergence curves and the split

Right panel = distance between optimizers / distance between data orders, **at equal training loss**.

![](figures/ts30m/divergence_adamw_muon.png)

- TinyStories 30M, AdamW vs Muon: up to ~100× the floor at loss 8 → 6, ≈ 1× from loss 4 on. **No split.**
- Same in all 9 pairs (3 settings × 3 pairs): big early gap, back to ≈ 1×. Only exception: Shakespeare
  Muon vs SGD, slightly above the floor from loss 2.8 (where SGD stalls).

![](figures/shakespeare/divergence_muon_sgd.png)

**[extra] Why they meet:** all optimizers pass through the **bigram model** (Shakespeare loss 2.47,
TinyStories ≈ 3.8). AdamW and SGD also pass the unigram model; Muon skips it.

![](figures/shakespeare/ngram.png)

**[extra] Optimizer vs data order, at equal steps** (blue: other optimizer, same data; greys: same
optimizer, other data):

![](figures/ts30m/decompose_js.png)

- AdamW vs Muon at the end of 30M: all ≈ 0.05 bits. Optimizer ≈ data order.
- vs SGD: 0.08 vs 0.06 — SGD ends elsewhere, but it is also slower (loss 1.37 vs 1.20).

## 3. [planned] What coincides with the changes: edge of stability, warmup, anatomy

![](figures/ts30m/sharpness.png)

- AdamW: preconditioned sharpness on its threshold for most of training (adaptive EoS).
- SGD: on its threshold only around the end of warmup.
- The early episodes end within warmup (steps 9–85), the same way for AdamW–Muon (no SGD involved).
  → No EoS event lines up with them. Best description: a *stage of learning* (bigram), not a step.

**[planned] Anatomy of the sharp direction:**

![](figures/ts30m/anatomy.png)

- Differs by optimizer: Muon → value/output matrices (TinyStories) or embeddings (Shakespeare); AdamW → head
  or norms.
- 70–85% of the top eigenvalue comes from **curvature between blocks**, not inside one.
- Changes of the dominant block happen long after the optimizers have re-met. No link to the split.

**[extra] Muon's embeddings at the edge of stability (Shakespeare):** Muon leaves embeddings/head/norms to
an Adam. Sharpness of that block over its threshold:

![](figures/shakespeare/aux_eos_matched.png)

- Muon's block goes 2.5× over at steps 300–450, right before Muon memorizes. AdamW's stays ≤ 0.1×.
- **[extra] Control:** Muon with that Adam matched to AdamW (lr 3e-4). It still memorizes, and its block
  still crosses (1.1×, later). So Muon's hidden-matrix updates make the embeddings sharp; the lr only
  changes when.
- On 30M it's AdamW's block that overshoots (2.1×) — it has the larger lr there.

## 4. [planned] Swaps: train with A until step t, then switch to B (TinyStories 10M)

Fork's distance to each pure run, in units of the noise floor (solid: to the new optimizer, dashed: old).

![](figures/ts10m/swap.png)

- AdamW ↔ Muon: within the floor of **both** pure runs at every switch step. Nothing to tell apart.
- SGD → AdamW/Muon: lands on the new optimizer even after 4000 SGD steps. SGD leaves no trace.
- AdamW/Muon → SGD: late switches stay near the old optimizer — SGD is too slow to undo it.

**[planned] Linear mode connectivity** (loss barrier, parent's final weights vs fork's final weights):

![](figures/ts10m/lmc.png)

- Same optimizer, other data: one basin if forked after step ~1000 (10% of training).
- Other optimizer: needs a later fork (~2500–4000). SGD → AdamW: **never** (barrier 3–6 nats), although
  that fork computes AdamW's function. **Same function, different basin.**

## 5. [planned] Break-even replication (CIFAR-10, plain SGD)

![](figures/breakeven/cifar/curves.png)

- Replicates: larger lr → lower sharpness (≈ 2/η) and better-conditioned gradient noise; runs share a path,
  then the large lr leaves first.
- **But** at equal loss, lr 0.01 vs 0.1 are only 1.0–1.75× the floor. The "separation" is speed.

**[extra] Transfer and batch size:**
- TinyStories 10M, lr 0.03/0.1/0.3: sharpness and noise effects replicate (not the noise conditioning). At
  equal loss: 150× early, 1× from loss 4 — the optimizer story again.
- Shakespeare: SGD without warmup blows up for ~10 steps; no split between rates.
- CIFAR batch 30 vs 300: smaller batch → lower sharpness (12 vs 50); at equal loss median 1.2×, no split.

## 6. [extra] Robustness of "no split"

Re-ran with top-1 disagreement, margin 0.25/1.0, run length 2/5, matching tolerance 1%/5%.

- TinyStories (both sizes) and Shakespeare AdamW–Muon: no split under **any** variant.
- Shakespeare Muon–SGD: gone under 3 of 7 variants → weak.

## 7. [extra] Second initialization

Same runs from another init (2 data orders). Distance to the same optimizer from **another init** (green)
vs **another data order** (grey) vs other optimizers (dashed), at equal steps:

![](figures/shakespeare/initseed_js.png)

- After ~100 steps, changing the init ≈ changing the data order. The data-order floor is a fair noise floor.
- TinyStories 10M, end of training: other init 0.054, other data 0.062, AdamW↔Muon 0.058 bits. **Init, data
  order and optimizer all move the function by the same amount.**
- The split analysis replicates almost exactly: AdamW–Muon episodes 4.18–2.86 and 2.41–2.02 (init 0:
  4.23–2.77, 2.30–2.12), no split; Muon–SGD again slightly above the floor from loss 2.80 (init 0: 2.82).
  TinyStories 10M: same episodes, no split.

## 8. [extra] Rare vs frequent tokens

![](figures/ts10m/tokens.png)

- SGD's loss gap is mostly on **rare** tokens (10M: +1.16 nats rarest fifth vs +0.11 most frequent), as
  Kunstner et al. (2024) predict.
- At equal loss: ≈ 1× the floor in every bin. SGD is slower on rare tokens, not different.

## 9. [extra] Positive control: a case where theory says they *should* split

Logistic regression on separable data: GD → ℓ2 max-margin, Adam → ℓ∞ max-margin.

![](figures/linear/linear_control.png)

- Same detector: within the floor for most of training, then a **persistent split** below loss ~1e-3. The
  method can find a split — but it comes near interpolation, which the language models never reach.
