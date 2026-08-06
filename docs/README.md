# reward-lens docs

The documentation site, built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
Self-contained in this folder.

## Layout

```
docs/
  mkdocs.yml              # site config: theme, plugins, nav
  requirements-docs.txt   # build deps (no torch needed)
  hooks.py                # build hooks: injects the generated pages
  gen_catalogue.py        # renders the catalogue + refusals from the registry
  open-quantities.txt     # lint rule two's ratchet
  claims-baseline.txt     # the claims checker's ratchet
  content/                # docs_dir: every hand-written page lives here
    index.md              # Home / Why reward-lens
    getting-started/  tutorials/  how-to/  models-and-signals/
    instruments/  training-loops/  concepts/  discipline/  theory/
    reference/  contributing/
    caveats.md  sciences.md  cli.md  migration.md  not-in-3-0.md
    assets/               # css, js (mathjax), figures/
  diagrams/               # figure sources + pipeline (not served)
    tikz/                 # TikZ sources -> SVG
    build_figures.sh      # compile TikZ into content/assets/figures/
```

## Build and preview

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements-docs.txt
pip install -e .                # the build imports reward_lens; see below

# live preview at http://127.0.0.1:8000
mkdocs serve -f docs/mkdocs.yml

# one-off build into docs/_site
mkdocs build --strict -f docs/mkdocs.yml
```

The API reference is generated from `src/` by griffe via static analysis, so you
do **not** need `torch`/`transformers` installed to build the docs. The base
install is required, though: `docs/hooks.py` imports `reward_lens` and renders
the instrument catalogue, the quantity registry and the refusal reference from
the live registry rather than from a transcription. Optional extras are not
needed and only make the open-research-target list shorter and more accurate.

## The two ratchets

Both are lists that may shrink and must not grow, and both fail CI when they do.

`docs/claims-baseline.txt` holds numbers in prose with no evidence id, recorded
before the claims checker existed. A number in a new page is either tagged with
the Evidence id it came from or labelled illustrative in the same sentence.

`docs/open-quantities.txt` is the second lint rule: a `Quantity` with no
estimator fails the docs build, naming it an open research target rather than a
bug. Record a deliberate one with

```bash
python docs/gen_catalogue.py --write-ledger    # from a base install, not one with extras
python docs/gen_catalogue.py --check           # what the docs build runs
```

Write the ledger from a base install. A richer environment imports more estimator
modules and so sees fewer open quantities, and a ledger recorded there would fire
spuriously on any leaner build.

## Figures

```bash
cd docs
./diagrams/build_figures.sh     # needs a local LaTeX toolchain + pdf2svg
```

See [`diagrams/README.md`](diagrams/README.md) for the three figure kinds (TikZ
geometry, inline Mermaid, matplotlib-from-real-runs) and when to use each.

## Deploy

Documentation publishes to GitHub Pages automatically. The
[`Deploy Docs to GitHub Pages`](../.github/workflows/docs.yml) workflow builds
the site with MkDocs on every push to the default branch and serves the built
HTML directly, so Pages never falls back to rendering a README.

One-time repository setup: **Settings → Pages → Build and deployment → Source →
GitHub Actions**. After that, every push to `main` rebuilds and redeploys.

A ReadTheDocs build also works (point it at `docs/mkdocs.yml`). `site_url` in
`mkdocs.yml` assumes GitHub Pages; change it to match your host.
