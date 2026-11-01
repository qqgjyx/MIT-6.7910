# layerlr: per-layer learning rates driven by curvature (dropped research line)

Per-layer rate multipliers that even out each layer's share of the sharp direction (`balance.py`: v1 variants A/B,
Muon on its spectral-norm sharpness, v2 `R` on the curvature along each layer's step, `F` fixed multipliers), and
per-layer curvature: layer x layer heatmaps of the top eigenvector (`curvature_layers.py`) and each layer's own top
eigenvalue by block Lanczos (`layer_lanczos.py`). Report: "Curvature-balanced per-layer learning rates" (§5.8,
Appendix E) and the heatmaps in "Anatomy of the sharp direction" (§5.7).

Results: for SGD it matches the re-tuned rate, as a self-tuning one (B1 beats SGD at 0.03 by about as much as the
re-tuned 0.06 does); AdamW and Muon gain nothing. Each layer self-stabilizes at its own edge (R_a settles near 1 whatever its rate), so curvature
cannot allocate the rate across layers.

Run: a config with `balance.variant` (defaults in `defaults.yaml`) makes `build_optimizer` and `train.py` load this
package; nothing else in the core imports it. Suites in `configs/suites/layerlr/` (`ts10m_balanced`,
`ts10m_balanced_v2`, `ts30m_balanced`, `shakespeare_balanced`, `ts10m_fixedmult`, `ts10m_fixedmult_ramp`), e.g.
`python scripts/launch.py configs/suites/layerlr/ts10m_balanced.yaml`; summaries and figures with
`python -m optsplit.layerlr.balanced`, `.curvature_layers` and `.layer_lanczos` (usage in their docstrings).
Tests: `python -m pytest tests/layerlr -q`.
