# Diagrams

Figures in these docs come in three kinds. Use the right tool for each; do not
force one tool to do another's job.

## 1. Geometry and linear algebra → TikZ, pre-compiled to SVG

Anything that makes a claim about *directions, projections, or angles* in
activation space. The reward-direction projection, preference-pair geometry,
multi-objective reward angles. These need to be precise, so they are drawn in
TikZ and compiled to SVG.

- Sources live in `tikz/*.tex`, one standalone document per figure.
- Build with `./build_figures.sh` (needs a local LaTeX toolchain plus one of
  `pdf2svg` / `dvisvgm` / `inkscape`). Output lands in
  `../content/assets/figures/*.svg`.
- The SVGs are committed, so the site builds in CI with no LaTeX dependency.
- Embed with `![The reward readout](assets/figures/reward-projection.svg)`.

`tikz/reward-projection.tex` is the worked example: the hero figure the whole
library rests on. Start there.

## 2. Process, architecture, decision flow → Mermaid (inline)

Anything that is a *flow* rather than a geometry: the RLHF pipeline, the
activation-patching swap, the observational-vs-causal decision map, the library
architecture. Write these inline as Mermaid fenced blocks; Material renders them
in the browser, so there is nothing to pre-build.

    ```mermaid
    flowchart LR
      A[chosen / rejected pair] --> B[RewardModel]
      B --> C{observational or causal?}
    ```

## 3. Empirical results → matplotlib, from real runs

Anything that reports *numbers the library actually produced*: reward-lens
curves, attribution waterfalls and heatmaps, the attribution-vs-patching
scatter, hacking effect-size bars, concept dose-response. Generate these from
real analysis runs, export to `../content/assets/figures/`, and commit them.
`make_empirical_figures.py` is the worked example: it reads recorded results and
draws them, so no number on a figure is typed in. Never hand-draw an empirical
plot, and never invent the numbers on it. If you cannot regenerate a figure,
leave a marked placeholder instead of faking it.

## Prior art worth a look

Anthropic's Jacobian Lens renders a legible layer × position grid where every
cell is readable and a summary row shows the model's real output
(<https://github.com/anthropics/jacobian-lens>). That legibility bar is the one
to hit for the attribution and patching heatmaps here. A future interactive
version of those grids is a good placeholder to leave in the docs.
