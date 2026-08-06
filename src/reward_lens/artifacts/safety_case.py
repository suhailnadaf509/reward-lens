"""The safety case: an ADJUDICATED-only assembly over the store (M13).

A safety case is the strongest artifact the library emits: the claim "safe to optimize against this
reward model for at most K nats under monitor M, with certified erasure of bias set B", plus the
honesty certificate that the monitored regime is incentive-compatible. It is a view over stored
Evidence like every other artifact (I5), but with the highest bar the gates express. Where a card
renders an uncalibrated index in an unvalidated style and moves on, a safety case *refuses*: a claim
this consequential may not rest on a number that lacks a scorecard or a preregistration. The
assembler loads each component Evidence and requires it to carry both calibration (gate 1) and
registration (gate 3); any component that lacks either makes the assembler raise, so an unvalidated
safety case cannot be built, let alone committed to the store or a paper.

The four components and where they come from in the science layer:

- ``k_nats`` budget: the KL budget in nats at which optimization turns catastrophic, from reward
  thermodynamics (S3, scoreboard T9). This is the K in "at most K nats".
- ``monitor``: the validated runtime monitor with its lead-time and obfuscation grades, from the
  policy-grader coupling and robustness work (S13/S14). This is the M in "under monitor M".
- ``erasure`` certificate: the post-hoc certificate that bias set B was erased at the chosen surface
  (LEACE affine erasure, S6/S12). This is the B in "certified erasure of bias set B".
- ``honesty`` certificate: the "honesty-incentive-compatible on domain D under evidence schema E"
  guarantee from the recursive-reward-skepticism and unraveling work (S16, scoreboard T14).

Nothing here recomputes K, re-grades the monitor, or re-verifies the certificate. It reads the four
Evidence records, checks their trust, extracts the claim fields, and renders the assembled case with
every number traceable to the Evidence id it came from.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Any

from reward_lens.core.errors import RewardLensError
from reward_lens.core.evidence import Evidence
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.core.types import TrustLevel

# The component roles, in the order they appear in the assembled claim.
ROLE_K_NATS = "k_nats_budget"
ROLE_MONITOR = "monitor"
ROLE_ERASURE = "erasure_certificate"
ROLE_HONESTY = "honesty_certificate"
_ROLES = (ROLE_K_NATS, ROLE_MONITOR, ROLE_ERASURE, ROLE_HONESTY)


class SafetyCaseRefusal(RewardLensError):
    """Raised when a safety-case component is not ADJUDICATED-grade.

    A safety case is refused, not degraded, when any component lacks calibration (gate 1) or
    registration (gate 3). The message names the component, the failing gate, and the Evidence id so
    the operator knows exactly which measurement must be calibrated or preregistered before the case
    can stand.
    """


@dataclass(frozen=True)
class SafetyComponent:
    """One evidence-backed component of a safety case, with its gate status surfaced."""

    role: str
    evidence_id: str
    observable: str
    value_repr: str
    trust: str
    calibrated: bool
    registered: bool

    @property
    def adjudicated(self) -> bool:
        """A component clears the bar only if it is both calibrated and registered."""
        return self.calibrated and self.registered


@dataclass
class SafetyCase:
    """An assembled safety case.

    Holds the subject reward model, the extracted claim parameters (K, the monitor summary, the bias
    set B, the honesty domain and schema), and the four components with their Evidence ids. It renders
    to markdown, HTML, and JSON; every rendered number cites the component it came from, so the case
    is as traceable as a card and refutable against the store by the claims checker.
    """

    signal: str
    k_nats: float
    bias_set: list[str]
    monitor_summary: str
    honesty_domain: str
    honesty_schema: str
    components: list[SafetyComponent] = field(default_factory=list)

    @property
    def claim(self) -> str:
        """The one-sentence safety claim this case stands behind."""
        b = ", ".join(self.bias_set) if self.bias_set else "(none named)"
        return (
            f"Safe to optimize against {self.signal} for at most {self.k_nats:g} nats under monitor "
            f"'{self.monitor_summary}', with certified erasure of bias set {{{b}}}, and honesty-"
            f"incentive-compatible on domain '{self.honesty_domain}' under evidence schema "
            f"'{self.honesty_schema}'."
        )

    @property
    def adjudicated(self) -> bool:
        """True only if every component cleared calibration and registration.

        The assembler refuses before a non-adjudicated case can be constructed, so this is True for
        any ``SafetyCase`` that exists; it is exposed so a renderer or a test can assert the invariant
        rather than trust it.
        """
        return all(c.adjudicated for c in self.components)

    def to_json(self) -> str:
        return json.dumps(
            {
                "signal": self.signal,
                "claim": self.claim,
                "k_nats": self.k_nats,
                "bias_set": self.bias_set,
                "monitor_summary": self.monitor_summary,
                "honesty_domain": self.honesty_domain,
                "honesty_schema": self.honesty_schema,
                "adjudicated": self.adjudicated,
                "components": [
                    {
                        "role": c.role,
                        "evidence_id": c.evidence_id,
                        "observable": c.observable,
                        "value": c.value_repr,
                        "trust": c.trust,
                        "calibrated": c.calibrated,
                        "registered": c.registered,
                    }
                    for c in self.components
                ],
            },
            indent=2,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Safety case: {self.signal}",
            "",
            f"> {self.claim}",
            "",
            "Every component below is ADJUDICATED: calibrated against an organism answer key (gate 1) "
            "and produced under a frozen study (gate 3). A component that lacked either would have "
            "made the assembler refuse, so this case exists only because all four cleared the bar.",
            "",
            "| Component | Observable | Value | Trust | Calibrated | Registered | Evidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in self.components:
            lines.append(
                f"| {c.role} | {c.observable} | {c.value_repr} | {c.trust} | "
                f"{'yes' if c.calibrated else 'no'} | {'yes' if c.registered else 'no'} | "
                f"`{c.evidence_id}` |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = []
        for c in self.components:
            rows.append(
                "<tr>"
                f"<td>{html.escape(c.role)}</td>"
                f"<td>{html.escape(c.observable)}</td>"
                f"<td>{html.escape(c.value_repr)}</td>"
                f"<td>{html.escape(c.trust)}</td>"
                f"<td>{'yes' if c.calibrated else 'no'}</td>"
                f"<td>{'yes' if c.registered else 'no'}</td>"
                f"<td><code>{html.escape(c.evidence_id)}</code></td>"
                "</tr>"
            )
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Safety case: {html.escape(self.signal)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; }}
blockquote {{ border-left: 4px solid #2b6a2b; padding-left: 1rem; color: #234; }}
</style></head>
<body>
<h1>Safety case</h1>
<blockquote>{html.escape(self.claim)}</blockquote>
<p>Every component is ADJUDICATED: calibrated (gate 1) and registered (gate 3). A component that
lacked either would have made the assembler refuse.</p>
<table>
<thead><tr><th>Component</th><th>Observable</th><th>Value</th><th>Trust</th><th>Calibrated</th>
<th>Registered</th><th>Evidence</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>"""


def _value_repr(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_value_repr(v)}" for k, v in list(value.items())[:3]) + "}"
    return type(value).__name__


def _require_adjudicated(role: str, ev: Evidence[Any]) -> SafetyComponent:
    """Check one component's gates and build its ``SafetyComponent``, or refuse.

    Calibration (gate 1) is read from whether the Evidence carries a calibration reference;
    registration (gate 3) is read from whether its trust reached REGISTERED, which the study runner
    sets on Evidence produced under a frozen study. Either gate failing raises ``SafetyCaseRefusal``
    with the role, the missing gate, and the Evidence id. Both passing yields a component that reports
    both facts so the render can show the reader the case stands on validated ground.
    """
    calibrated = ev.is_calibrated
    registered = ev.trust >= TrustLevel.REGISTERED
    if not calibrated:
        raise SafetyCaseRefusal(
            f"refusing to assemble the safety case: the {role} component (Evidence {ev.id}, "
            f"observable {ev.observable}) is not calibrated. A safety case is ADJUDICATED-only; an "
            "uncalibrated component has no scorecard behind it (gate 1) and the assembler will not "
            "let it stand behind a safety claim."
        )
    if not registered:
        raise SafetyCaseRefusal(
            f"refusing to assemble the safety case: the {role} component (Evidence {ev.id}, "
            f"observable {ev.observable}) is not registered (trust {ev.trust.name} < REGISTERED). A "
            "safety case is ADJUDICATED-only; a component that was not produced under a frozen study "
            "carries no preregistered prediction (gate 3) and cannot back a safety claim."
        )
    return SafetyComponent(
        role=role,
        evidence_id=ev.id,
        observable=ev.observable,
        value_repr=_value_repr(ev.value),
        trust=ev.trust.name,
        calibrated=calibrated,
        registered=registered,
    )


def _extract_float(ev: Evidence[Any], field_name: str, default: float) -> float:
    """Read a float claim field from an Evidence value, tolerating scalar or dict payloads."""
    val = ev.value
    if isinstance(val, dict) and field_name in val:
        val = val[field_name]
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_str_list(ev: Evidence[Any], field_name: str) -> list[str]:
    """Read a list-of-strings claim field (the erased bias set) from an Evidence value."""
    val = ev.value
    if isinstance(val, dict) and field_name in val:
        val = val[field_name]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [val]
    return []


def _extract_str(ev: Evidence[Any], field_name: str, default: str) -> str:
    val = ev.value
    if isinstance(val, dict) and field_name in val:
        return str(val[field_name])
    return default


def assemble_safety_case(
    signal: str,
    *,
    k_nats_evidence: str,
    monitor_evidence: str,
    erasure_evidence: str,
    honesty_evidence: str,
    store: EvidenceStore | None = None,
    bias_set: list[str] | None = None,
) -> SafetyCase:
    """Assemble a safety case from four component Evidence ids, refusing unless all are ADJUDICATED.

    Loads the K-nats budget (S3), the monitor (S13/S14), the erasure certificate (S6/S12), and the
    honesty certificate (S16) by id, requires each to carry calibration (gate 1) and registration
    (gate 3), and extracts the claim parameters from the Evidence values. ``bias_set`` names the set B
    the case claims erased; if omitted, it is read from the erasure Evidence's ``bias_set`` field. Any
    missing id raises ``KeyError`` from the store; any un-calibrated or un-registered component raises
    ``SafetyCaseRefusal``. Nothing is recomputed: K, the monitor grade, and the certificate are read
    straight from their Evidence.
    """
    store = store if store is not None else default_store()

    k_ev = store.get(k_nats_evidence)
    monitor_ev = store.get(monitor_evidence)
    erasure_ev = store.get(erasure_evidence)
    honesty_ev = store.get(honesty_evidence)

    components = [
        _require_adjudicated(ROLE_K_NATS, k_ev),
        _require_adjudicated(ROLE_MONITOR, monitor_ev),
        _require_adjudicated(ROLE_ERASURE, erasure_ev),
        _require_adjudicated(ROLE_HONESTY, honesty_ev),
    ]

    k_nats = _extract_float(k_ev, "k_nats", default=0.0)
    monitor_summary = _extract_str(monitor_ev, "monitor", default=monitor_ev.observable)
    resolved_bias = bias_set if bias_set is not None else _extract_str_list(erasure_ev, "bias_set")
    honesty_domain = _extract_str(honesty_ev, "domain", default="unspecified")
    honesty_schema = _extract_str(honesty_ev, "evidence_schema", default="unspecified")

    return SafetyCase(
        signal=signal,
        k_nats=k_nats,
        bias_set=resolved_bias,
        monitor_summary=monitor_summary,
        honesty_domain=honesty_domain,
        honesty_schema=honesty_schema,
        components=components,
    )


__all__ = [
    "SafetyCaseRefusal",
    "SafetyComponent",
    "SafetyCase",
    "assemble_safety_case",
    "ROLE_K_NATS",
    "ROLE_MONITOR",
    "ROLE_ERASURE",
    "ROLE_HONESTY",
]
