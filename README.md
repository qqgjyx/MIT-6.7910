# MIT-6.7910

MIT 9.520 / 6.7910 Statistical Learning Theory, Fall 2026. Group: Juntang Wang and Víctor Conchello.

## Literature reviews (Oct 2)

`litreview/` holds one document per project: an annotated literature review, three key papers, and a
page of implications.

| Project | Source |
|:--|:--|
| Why does late-stage learning-rate annealing help? | `topic1_annealing.tex` |
| Does an EoS mechanism explain E3M4 vs E2M5? | `topic2_e3m4_e2m5.tex` |
| Optimizer-controlled implicit bias and functional complexity | `topic3_optimizer_bias.tex` |

## Project plan (Oct 9)

`project_plan/project_plan.tex`: when do SGD, AdamW and Muon start to learn different functions
(nanoGPT), plus a moonshot on which layers carry the sharpness at the edge of stability.
`project_plan/decisions.md` logs each design choice and why.

## Topic 3 project code

`code/` holds the experiments of `project_plan/project_plan.pdf` (setup and usage in `code/README.md`).

## Building

Run `latexmk -pdf <file>.tex` inside the document's folder. Topics 1 and 3 and the plan share
`litreview/preamble.tex` and `litreview/refs.bib` (titles and authors from arXiv, venues checked
against proceedings); the plan adds `project_plan/plan_refs.bib`. Topic 2 is self-contained, with
its own bibliography. The preamble uses biblatex's bibtex backend because biber is broken on our
build machine.
