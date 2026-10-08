# Decisions in `project_plan.tex` (topic 3: optimizer implicit bias)

Each entry: what the decision is about, the options, what we chose and why.

## 1. Theory or experiments
- **About:** the assignment allows proofs, experiments or both.
- **Options:** (a) experiments only; (b) a new theorem; (c) experiments plus a check against known theory.
- **Chosen:** (c). The professor's three requirements (model bank, two metrics, swap) are experimental. On linearly separable data the limits are known (ℓ2 for GD, ℓ∞ for Adam, spectral for Muon), so the swap has an exact prediction there, which validates the pipeline. A new theorem for nonlinear networks is unrealistic in two months.

## 2. Tasks
- **About:** where the model bank is built.
- **Options:** linear separable data; Vasudeva's Gaussian setting with a two-layer network; MNIST or CIFAR-10 with an MLP or CNN; a small transformer language model (TinyStories); spurious-correlation benchmarks (Waterbirds, colored MNIST); ReLU or smooth activations.
- **Chosen:** three tasks, all with GELU:
  - **(0) linear separable data**, a theory check;
  - **(A) a small GELU CNN on CIFAR-10;**
  - **(B) a small GELU transformer language model on TinyStories.**

  Each task is run at 3 model sizes (input dimension for task 0, CNN width, roughly 1M/3M/10M parameters for the language model), and at each size each optimizer gets 3 learning rates × 3 seeds: 81 runs per task. Learning rates and seeds keep optimizer differences from being learning-rate differences and let them be compared with the seed-to-seed spread; sizes show whether a difference holds as models grow, and give more models per comparison.
- **Why:**
  - Only one synthetic task: the linear check is enough for validation; the other two are realistic.
  - The language model is the setting Muon was designed for and the one most relevant today. Input derivatives are taken with respect to the input embeddings, so the metrics stay defined despite discrete tokens.
  - CIFAR-10 comes first because it is cheaper, so first results are ready for the Oct 30 checkpoint.
  - Vasudeva's Gaussian setting was dropped: it came with a theoretical prediction, but a second synthetic task adds little over task 0.
  - Smooth activations: with ReLU the input Hessian is zero almost everywhere, so second- and third-order metrics would be meaningless; ReLU is also rare in current models.

## 3. What "matched loss" means
- **About:** the condition under which models count as risk-equivalent.
- **Options:** stop at fixed training-loss targets (e.g. before and after interpolation); train to convergence; same test loss or accuracy; same number of steps; same compute.
- **Chosen:** train every run to convergence (training-loss plateau), and report final training and test loss with every metric.
- **Why:** simpler than loss targets, and the converged model is the one the optimizer actually returns. On CIFAR-10 all runs reach near-zero training loss, so losses are effectively matched. The language model does not interpolate, so its final losses may differ between optimizers; reporting them lets us check whether metric differences just track loss.

## 4. Function metrics
- **About:** how complexity, curvature and robustness are measured.
- **Options:** region counts; input-derivative metrics (Jacobian, Hessian, higher order); weight norms; margin in input space; representation similarity (CKA); accuracy and robustness.
- **Chosen:** two main sets, plus accuracy:
  - **Input derivatives** of f, the loss at the true label as a function of the input (for the language model: next-token loss as a function of the input embeddings), at test points; mean and standard deviation over the test set:
    - Jacobian Frobenius norm (Novak 2018);
    - Hessian Frobenius norm by Hutchinson, and top eigenvalue λ by power iteration (CURE, Moosavi-Dezfooli 2019); level-set curvature ‖PHP‖_F/‖∇f‖ = √(E_z‖PH(Pz)‖²)/‖∇f‖, with P projecting orthogonally to ∇f (how fast the gradient direction turns; a projected variant of Srinivas 2022's normalized curvature; the curl was rejected since gradient fields are curl-free);
    - **third order, novel to the best of our knowledge** (no input-space precedent found), all built on T[u,u,·] = ∇ₓ(uᵀH(x)u) with u fixed:
      - ‖T‖_σ = max_{‖u‖=1} |T[u,u,u]|, estimated by shifted tensor power iteration from a few random starts;
      - T[v,v,v], the third derivative along the sharpest direction, with the sign of v fixed by vᵀ∇f ≥ 0;
      - λ/‖∇ₓλ‖ (∇ₓλ = T[v,v,·] for simple λ), the distance over which, to first order, the top curvature doubles or vanishes.
  - **Weight norms:** ℓ0, ℓ1, ℓ2, ℓ3, ℓ∞ of all weights; per layer spectral norm, nuclear norm and stable rank; path norm, spectral complexity and distance from initialization (the last three among the measures compared in Jiang 2020).
  - **Accuracy and robustness:** test accuracy; for images, accuracy under Gaussian input noise; for the language model, Hahn et al. 2021 sensitivity (how often the prediction changes when one token is replaced), chosen over embedding noise because it is discrete, the standard simplicity measure for transformers, and avoids the embedding scale differing across optimizers.
- **Why:**
  - Input derivatives describe the function itself, independent of parametrization, and go from slope to curvature to change of curvature.
  - Norms are what linear theory predicts each optimizer keeps small, so they test those predictions directly; using several avoids picking the one that favours a given optimizer.
  - Region counts were dropped: they need piecewise-linear networks and say little about smooth functions.

## 5. Swap design
- **About:** the causal intervention.
- **Options:** one pair or all pairs; one swap time or several; how to start B: cold state with a learning-rate re-warm, or B's state accumulated in shadow during A's run; controls: none, or continue with the same optimizer.
- **Chosen:** all 6 ordered pairs, swapped once, after convergence: from A's converged model, continued with B until it converges again. B's momentum and second-moment buffers are accumulated in shadow from A's gradients during A's run (without applying B's updates), so B starts warm. One control: continue with A for the same number of steps. The result is the fraction of the A-versus-B gap a swap closes.
- **Why:**
  - Swapping after convergence asks the cleanest question: is A's solution stable under B, or does B move it toward B's own solutions? It keeps the design to one swap per pair. Swapping before convergence was rejected as less clean (the result mixes finishing training with B's bias). Expected asymmetry: SGD barely moves from a converged point (near-zero gradient), while Adam and Muon normalize their steps and keep moving.
  - Shadow state removes the cold start (e.g. Adam's second-moment estimate is wrong for its first steps), the main reason for a re-warm, and with it the need for a re-warm control. It costs only memory. The step size still changes at the swap, which the A-continued control addresses.

## Assumptions to check
- **Deadlines:** initial checkpoint Oct 30 (first results needed), project discussion in the first two weeks of November, presentation Dec 1–4, final paper Dec 11. Experiments are planned to finish by Nov 30.
- **Split of work:** Juntang leads CIFAR-10, and Víctor leads the linear check and the language model. This is a placeholder until you agree with Juntang.

## Revision after the TA meeting (Oct 8)
The plan above was rewritten around *when* optimizers diverge, following the TA's suggestions. Decisions 2-5 are superseded where they conflict with the points below.

### 6. Main question
- **About:** what the project measures.
- **Options:** (a) end-point comparison at matched loss plus a swap after convergence (the Oct 5 plan); (b) the full trajectory: do SGD, AdamW and Muon first follow the same path in function space and diverge later, and where.
- **Chosen:** (b). The TA pointed to a vision result where runs separate in prediction space after an early phase (we cite Jastrzębski et al. 2020, the break-even point; **Víctor: confirm this is the paper he meant**). The end-point comparison survives as the last step of the swap.
- **Why:** "where do they split, and what happens there" is a sharper question than "are the end points different", and it connects directly to edge-of-stability events, which is the instructors' line.

### 7. Setting
- **Chosen:** nanoGPT (the TA's suggestion) on character-level Shakespeare first, then TinyStories at ~10M and ~30M parameters. Muon on hidden matrices with AdamW on embeddings and head, as in the Muon reference implementation. A small CIFAR-10 CNN only as a check that the pipeline reproduces the vision result.
- **Dropped:** the linear separable task and the three-size sweep (cost, and the 2-page limit); the third-order input-derivative metrics move to "if time allows".

### 8. Locating the split
- **Chosen:** Jensen-Shannon divergence and top-1 agreement of next-token predictions on a fixed held-out set, aligned both by step and by training loss; noise floor from the same optimizer with a different data order; t* = first loss-matched point above the floor by a fixed margin for several checkpoints, bootstrapped over seeds. Second test: linear mode connectivity (Frankle et al. 2020) between forks as a function of fork time.

### 9. Swap timing
- **Chosen:** swaps at several times (before, at and after t*) instead of once after convergence; shadow state kept. Prediction: before t* the fork ends at B's function, after t* it stays near A's.

### 10. Moonshot: layer anatomy of EoS
- **Chosen:** split the top Hessian eigenvector by parameter block (embeddings, QK, OV and MLP per layer, norms, head) and record each block's share of the direction and of the sharpness, over training and by optimizer (AdamW: preconditioned Hessian). Side project, shared, after Nov 15.

### Split of work (proposal, Víctor to confirm)
Juntang: divergence curves and sharpness tracking. Víctor: swaps and function metrics. Moonshot shared.
