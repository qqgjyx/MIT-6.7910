# MIT-6.7910

MIT 9.520 / 6.7910 Statistical Learning Theory, Fall 2026. Group: Juntang Wang and Víctor Conchello.

## Oct 2 literature reviews

`litreview/` holds one document per project: an annotated literature review, three key papers, and a
page of implications.

| Project | Source | Status |
|:--|:--|:--|
| Why does late-stage learning-rate annealing help? | `topic1_annealing.tex` | done |
| Does an EoS mechanism explain E3M4 vs E2M5? | `topic2_e3m4_e2m5.tex` | done |
| Optimizer-controlled implicit bias and functional complexity | `topic3_optimizer_bias.tex`| done |

Build from `litreview/` with `latexmk -pdf <file>.tex`. Shared preamble in `preamble.tex`;
bibliography in `refs.bib` (titles and authors from arXiv, venues checked against proceedings).
