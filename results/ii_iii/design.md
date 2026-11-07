# Design for next steps (ii) and (iii) (Gate A draft, for Juntang's OK)

All runs: TinyStories-10M, the checkpoint's settings (AdamW 2e-3, Muon 0.01), 10,000 steps, data orders d0-d2 unless
said otherwise, on one GPU type (A100). New runs are compared only with new runs.

Properties under test (Step 2, AdamW vs Muon at training loss 1.81): stable rank, spectral norm and
sigma1/sigma_median of the block matrices W; cosine of consecutive updates; lambda_max(H) and tr H; fraction of
active MLP units.

## (ii) Rule or training?

A property is *rule-driven* if Muon's update rule produces it whatever gradients it is fed; *emergent* if it needs
the closed loop where Muon's own updates shape the gradients it later sees.

**A. Open-loop ghost (main control).** During an AdamW run, keep a second copy of the 36 block matrices (the
"ghost") and update it with Muon's rule (own Nesterov momentum, NS5, Muon's lr schedule) on the AdamW run's
gradients. The ghost never affects the gradients it gets: the same gradient stream goes through Muon's rule with the
feedback removed. Symmetric: an AdamW ghost (with wd 0.1, own moments) inside a Muon run. Sanity check: a ghost of
the run's own rule reproduces the run's weights exactly.
- Ghost Muon on AdamW gradients ends with Muon-like spectra (stable rank about 86, low sigma1/sigma_median): the
  spectra follow from the rule applied to this kind of gradient stream. AdamW-like spectra: they need feedback, so
  they are emergent.
- Only weight properties are well defined for a ghost (its model is not trained, so its loss, Hessian and
  activations mean little). Curvature and MLP sparsity are covered by C and D.
- Cost: a hook like `optim.Shadow`, but with a real learning rate on a parameter copy, plus ghost checkpoints at the
  ts10m_traj steps; 2 parents x 3 orders = 6 runs, about 2 GPU-h.

**B. No-signal baseline (cheap, CPU).** The rule fed i.i.d. Gaussian gradients of the same shapes and norms, and
gradients at the frozen initialization on fresh batches (real structure, no learning). This gives each rule's
"default" spectrum. If AdamW also gives a flat spectrum on noise, AdamW's low stable rank comes from the structure
of real gradients, not from its rule.

**C. Which part of Muon's rule.** Closed-loop runs that change one component at a time:
- C1 AdamW without weight decay (Muon has none, so weight decay is a confound for spectral norm and stable rank);
- C2 Muon with decoupled weight decay 0.1;
- C3 Muon without orthogonalization: the same Nesterov momentum, each matrix's update rescaled to the Frobenius
  norm of an orthogonal update. This separates orthogonalization from step normalization. It is a new optimizer,
  so it gets a 3-point lr check on d0.

A property that flips with a component is driven by that component. Cost: 9 runs + 3 lr runs, about 4 GPU-h, plus
offline trajectory metrics.

## (iii) Do the differences explain Muon's speed?

**D. Dissociation across variants (no extra runs).** For AdamW, Muon, C1-C3 (and SGD as a reference), plot speed
(steps to reach losses 2.0, 1.6 and 1.4) against each property at equal loss. A property that moves while speed
does not (or the reverse) is not the reason Muon is faster. Only consistent co-variation keeps it as a candidate.
With 5-6 variants, this can only reject candidates.

**E. Where each step's progress comes from (cheap).** Loss drop per step is about
-eta g.u + (1/2) eta^2 u^T H u. The existing `measure.delta_loss` and `measure.step_sharpness` log both terms. At
equal loss: does Muon gain more first-order progress, or pay less curvature along its own step even though its
lambda_max is higher? This links Muon's higher sharpness to its speed directly. Turn on `step_sharpness` in the
reproduced d0-d2 runs (a few extra GPU-min each).

**F. Optional, interventional.** Attention has exact symmetries per head (Wq A, Wk A^-T; A Wv, Wo A^-1). At step
1000, rebalance AdamW's attention matrices to Muon-like spectra without changing the function, then continue with
AdamW. If it now trains faster, the attention spectrum is causal. MLP has no such symmetry, so this only tests
attention. About 1 GPU-h, but more code.

## Budget (A100, rough: 15-25 GPU-min per 10k-step run)

| Item | Runs | GPU-h |
|:--|--:|--:|
| Reproduction: ts10m_main (AdamW, Muon d0-d4; SGD d0 for the swap parent) | 11 | 4 |
| Reproduction: ts10m_traj (AdamW, Muon d0-d2) + trajectory metrics | 6 | 4 |
| Swap figure: ts10m_parent_late + ts10m_forks_dense + metrics | 26 | 8 |
| A ghost | 6 | 2 |
| C ablations (+ lr check) | 12 | 5 |
| E step sharpness (inside reproduction) | 0 | 1 |
| F (optional) | 3 | 1 |
| Total | | about 25 |

Full ts10m_forks (54 runs, about 12 GPU-h) is not needed for (ii)/(iii). Suggest skipping it unless you want the LMC
figures reproduced.

## Open points for you

- SGD: ts10m_main uses lr 0.03, while ts10m_traj and the checkpoint tables use 0.06. Proposal: only AdamW and Muon in
  the reproduction, with SGD 0.06 in ts10m_traj as the reference.
- F: in or out?
