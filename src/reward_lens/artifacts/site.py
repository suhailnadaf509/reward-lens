"""Static site export: cards, the scoreboard, and the Atlas leaderboard (M13).

The delivery layer's public face is the existing MkDocs Material site, and whoever publishes the
benchmarks owns the field's citation graph, so the site export is a first-class artifact rather than
a documentation afterthought. ``build_site`` writes markdown pages that MkDocs renders: an overview,
the theorem scoreboard, the Atlas population and its leaderboard, and one RM card per model present
in the store. Every page is a view over stored Evidence assembled through the existing generators
(``build_card``, ``Scoreboard``, ``Atlas.leaderboard``); this module computes no numbers of its own
and adds none to the store. It writes markdown rather than the generators' HTML because the MkDocs
theme owns the page chrome, and an uncalibrated index stays visibly flagged in the markdown exactly
as it is on the HTML card (gate 1 at the render layer).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from reward_lens.artifacts.atlas import Atlas, Leaderboard
from reward_lens.artifacts.card import Card, build_card
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.studies.scoreboard import Scoreboard


def _slug(text: str) -> str:
    """A filesystem- and URL-safe slug for a fingerprint or model name."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")
    return s or "unnamed"


def _signals_in_store(store: EvidenceStore) -> list[str]:
    """The distinct model fingerprints any Evidence in the store is about."""
    found: set[str] = set()
    for ev in store.find():
        for fp in ev.subject.signals:
            found.add(fp)
    return sorted(found)


def _card_to_markdown(card: Card) -> str:
    """Render a Card to markdown, keeping the unvalidated flagging visible (gate 1)."""
    lines = [
        f"# RM Card: `{card.signal}`",
        "",
        (
            f"Total metered GPU-seconds: {card.total_gpu_seconds:.2f}. "
            f"{len(card.entries)} measurements, {len(card.unvalidated)} uncalibrated."
        ),
        "",
    ]
    if card.unvalidated:
        obs = ", ".join(sorted({e.observable for e in card.unvalidated}))
        lines += [
            '!!! warning "Explicit gaps"',
            f"    Uncalibrated, not yet validated against an answer key: {obs}.",
            "",
        ]
    lines += [
        "| Observable | Value | CI | n_eff | Trust | Gauge | Calibrated |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in card.entries:
        ci = ""
        if e.ci[0] is not None and e.ci[1] is not None:
            ci = f"[{e.ci[0]:.3g}, {e.ci[1]:.3g}]"
        ness = f"{e.n_effective:.1f}" if e.n_effective is not None else ""
        obs_label = e.observable if e.validated else f"**unvalidated** {e.observable}"
        lines.append(
            f"| {obs_label} | {e.value_repr} | {ci} | {ness} | {e.trust} | {e.gauge} | "
            f"{'yes' if e.calibrated else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _leaderboard_to_markdown(lb: Leaderboard) -> str:
    """Render the Atlas leaderboard to a markdown table, flagging uncalibrated cells."""
    lines = [
        "## Leaderboard",
        "",
        (
            f"{len(lb.models)} models, {len(lb.observables)} observables, "
            f"{len(lb.present)} measured cells, {len(lb.flagged)} uncalibrated. "
            f"Total metered GPU-seconds: {lb.total_gpu_seconds:.2f}."
        ),
        "",
    ]
    if not lb.observables:
        lines += [
            "_No observables measured for this population yet. The leaderboard populates as the "
            "battery runs and Evidence lands in the store._",
            "",
        ]
        return "\n".join(lines)
    header = "| Model | " + " | ".join(lb.observables) + " |"
    sep = "|---|" + "|".join(["---"] * len(lb.observables)) + "|"
    lines += [header, sep]
    for m in lb.models:
        cells = []
        for obs in lb.observables:
            c = lb.cells.get((m, obs))
            if c is None:
                cells.append("_no evidence_")
            elif c.validated:
                cells.append(c.value_repr)
            else:
                cells.append(f"{c.value_repr} _(unvalidated)_")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _population_to_markdown(atlas: Atlas) -> str:
    """Render the Atlas population registry with declared lineage."""
    lines = [
        "## Population",
        "",
        (
            "The standard population, with declared lineage. Fingerprints are card-claimed until a "
            "weight-verified sweep replaces them (the provenance tier states which)."
        ),
        "",
        "| Model | Paradigm | Base model | Release | Provenance tier |",
        "|---|---|---|---|---|",
    ]
    for e in atlas.entries:
        lines.append(
            f"| {e.name} | {e.paradigm} | {e.lineage.base_model} | "
            f"{e.lineage.release_date or 'n/a'} | {e.lineage.provenance_tier} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_site(
    store: EvidenceStore | None = None,
    out_dir: str | Path = "docs/content/atlas",
    *,
    signals: Iterable[str] | None = None,
    atlas: Atlas | None = None,
    observables: Iterable[str] | None = None,
    scoreboard: Scoreboard | None = None,
) -> dict[str, Path]:
    """Write the static site pages from the store, returning a map of page key to path.

    Writes ``index.md`` (overview and links), ``scoreboard.md`` (the theorem scoreboard), ``atlas.md``
    (the population registry and its leaderboard), and ``cards/<slug>.md`` for each model the store
    holds Evidence about. ``signals`` overrides the auto-discovered model set; ``atlas`` overrides the
    default standard population (pass one whose fingerprints match your store to populate the
    leaderboard); ``observables`` fixes the leaderboard columns; ``scoreboard`` supplies a persisted
    scoreboard rather than the default rows. Directories are created as needed. The pages are markdown
    for the MkDocs Material theme and every number in them is a stored measurement, not a fresh one.
    """
    store = store if store is not None else default_store()
    atlas = atlas if atlas is not None else Atlas.standard()
    scoreboard = scoreboard if scoreboard is not None else Scoreboard()
    out = Path(out_dir)
    cards_dir = out / "cards"
    out.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # Cards, one per model present in the store.
    fps = list(signals) if signals is not None else _signals_in_store(store)
    card_links: list[str] = []
    for fp in fps:
        card = build_card(fp, store)
        page = cards_dir / f"{_slug(fp)}.md"
        page.write_text(_card_to_markdown(card), encoding="utf-8")
        written[f"card:{fp}"] = page
        card_links.append(f"- [`{fp}`](cards/{_slug(fp)}.md)")

    # Scoreboard.
    sb_page = out / "scoreboard.md"
    sb_page.write_text(
        "# Theorem scoreboard\n\n"
        "Standing theorems and candidate laws with their status and adjudicating evidence. "
        "Refutations render as prominently as confirmations (I4).\n\n"
        + scoreboard.render_markdown()
        + "\n",
        encoding="utf-8",
    )
    written["scoreboard"] = sb_page

    # Atlas: population + leaderboard.
    lb = atlas.leaderboard(store=store, observables=observables)
    atlas_page = out / "atlas.md"
    atlas_page.write_text(
        "# Atlas\n\n"
        "The reward-model population and its leaderboard, a view over the evidence store.\n\n"
        + _population_to_markdown(atlas)
        + "\n"
        + _leaderboard_to_markdown(lb)
        + "\n",
        encoding="utf-8",
    )
    written["atlas"] = atlas_page

    # Index.
    index_lines = [
        "# reward-lens artifacts",
        "",
        "Cards, the theorem scoreboard, and the Atlas leaderboard, all assembled from the evidence "
        "store. Nothing on these pages is computed here; every number cites the Evidence it came "
        "from, which is what lets a card and a paper agree (I5).",
        "",
        "- [Atlas](atlas.md): the model population and its leaderboard.",
        "- [Theorem scoreboard](scoreboard.md): standing theorems and candidate laws.",
        "",
        "## RM cards",
        "",
    ]
    index_lines += (
        card_links if card_links else ["_No model cards yet: the store holds no Evidence._"]
    )
    index_lines.append("")
    index_page = out / "index.md"
    index_page.write_text("\n".join(index_lines), encoding="utf-8")
    written["index"] = index_page

    return written


__all__ = ["build_site"]
