"""The RM Card generator.

A card is a view over the evidence store: it assembles every Evidence about a signal into a JSON
and an HTML document and computes nothing new (I5). The design's hard requirement is the one this
module enforces at the render layer: every uncalibrated index renders in a visually distinct
"unvalidated" style, and there is no code path that styles an uncalibrated number as validated
(gate 1 enforcement at the render layer). The card also names its explicit gaps: the observables
present about this signal that carry no calibration are listed as unvalidated so a reader sees what
the card does not yet stand behind.

The card renders whatever Evidence exists about a signal, grouped by observable. As the battery and
the index library populate the store, the same generator produces a richer card with no change
here; that is what "artifacts never compute" buys.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from reward_lens.core.store import EvidenceStore, default_store


@dataclass
class CardEntry:
    """One measurement on a card, with its trust, gauge, and calibration surfaced."""

    observable: str
    version: str
    value_repr: str
    trust: str
    gauge: str
    calibrated: bool
    ci: tuple[float | None, float | None]
    n_effective: float | None
    readout: str | None
    evidence_id: str

    @property
    def validated(self) -> bool:
        """A card entry is validated only if it carries a calibration reference (gate 1)."""
        return self.calibrated


@dataclass
class Card:
    """An RM Card: the assembled evidence about one signal."""

    signal: str
    entries: list[CardEntry] = field(default_factory=list)
    total_gpu_seconds: float = 0.0

    @property
    def unvalidated(self) -> list[CardEntry]:
        """The explicit gaps: entries with no calibration, rendered distinctly (gate 1)."""
        return [e for e in self.entries if not e.validated]

    def to_json(self) -> str:
        return json.dumps(
            {
                "signal": self.signal,
                "total_gpu_seconds": self.total_gpu_seconds,
                "entries": [asdict(e) for e in self.entries],
                "unvalidated_observables": sorted({e.observable for e in self.unvalidated}),
            },
            indent=2,
        )

    def to_html(self) -> str:
        return _render_html(self)


def _value_repr(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_value_repr(v)}" for k, v in list(value.items())[:4]) + "}"
    return type(value).__name__


def build_card(signal_fp: str, store: EvidenceStore | None = None) -> Card:
    """Assemble a Card from every latest Evidence about a signal.

    Queries the store for the most recent Evidence per observable about this signal fingerprint,
    reads each one's value, uncertainty, trust, gauge, and calibration status, and returns a Card.
    Uncalibrated entries are not filtered out; they are carried and flagged, because a card that
    hid its gaps would be a marketing document.
    """
    store = store if store is not None else default_store()
    found = store.find(signal=signal_fp, latest=True)
    entries: list[CardEntry] = []
    total_gpu = 0.0
    for ev in found:
        total_gpu += ev.provenance.cost.gpu_seconds
        entries.append(
            CardEntry(
                observable=ev.observable,
                version=ev.observable_version,
                value_repr=_value_repr(ev.value),
                trust=ev.trust.name,
                gauge=ev.gauge.value,
                calibrated=ev.is_calibrated,
                ci=(ev.uncertainty.ci_low, ev.uncertainty.ci_high),
                n_effective=ev.uncertainty.n_effective,
                readout=ev.subject.readout,
                evidence_id=ev.id,
            )
        )
    entries.sort(key=lambda e: (not e.validated, e.observable))
    return Card(signal=signal_fp, entries=entries, total_gpu_seconds=total_gpu)


_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; }
.validated { }
.unvalidated { background: #fff4f4; color: #7a2b2b; }
.unvalidated td:first-child::before { content: "unvalidated "; font-weight: 700; text-transform: uppercase; font-size: 0.7em; letter-spacing: 0.05em; }
.badge { font-size: 0.75em; padding: 1px 6px; border-radius: 3px; background: #eee; }
.trust-EXPLORATORY { background: #f0e6e6; }
.trust-CALIBRATED { background: #e6f0e6; }
.trust-REGISTERED { background: #e6e9f0; }
.trust-ADJUDICATED { background: #d6ecd6; }
"""


def _render_html(card: Card) -> str:
    rows = []
    for e in card.entries:
        cls = "validated" if e.validated else "unvalidated"
        ci = ""
        if e.ci[0] is not None and e.ci[1] is not None:
            ci = f"[{e.ci[0]:.3g}, {e.ci[1]:.3g}]"
        ness = f"{e.n_effective:.1f}" if e.n_effective is not None else ""
        rows.append(
            f'<tr class="{cls}">'
            f"<td>{html.escape(e.observable)}</td>"
            f"<td>{html.escape(e.value_repr)}</td>"
            f"<td>{html.escape(ci)}</td>"
            f"<td>{html.escape(ness)}</td>"
            f'<td><span class="badge trust-{e.trust}">{e.trust}</span></td>'
            f"<td>{html.escape(e.gauge)}</td>"
            f"<td>{'yes' if e.calibrated else 'no'}</td>"
            f"</tr>"
        )
    gap_note = ""
    if card.unvalidated:
        obs = ", ".join(sorted({e.observable for e in card.unvalidated}))
        gap_note = (
            f'<p class="unvalidated" style="padding:8px">Explicit gaps (uncalibrated, '
            f"not yet validated against an answer key): {html.escape(obs)}.</p>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RM Card: {html.escape(card.signal)}</title>
<style>{_CSS}</style></head>
<body>
<h1>RM Card</h1>
<p>Signal <code>{html.escape(card.signal)}</code>. Total metered GPU-seconds:
{card.total_gpu_seconds:.2f}. {len(card.entries)} measurements,
{len(card.unvalidated)} uncalibrated.</p>
{gap_note}
<table>
<thead><tr><th>Observable</th><th>Value</th><th>CI</th><th>n_eff</th><th>Trust</th>
<th>Gauge</th><th>Calibrated</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>"""


__all__ = ["CardEntry", "Card", "build_card"]
