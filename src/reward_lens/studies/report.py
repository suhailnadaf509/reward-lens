"""Study reports, rendered from Evidence.

A report is a view over the study's frozen spec and the Evidence it produced; it computes nothing
new. It shows the registered predictions next to their outcomes, so a reader sees exactly what was
predicted before the run and how it came out. A study whose kill criterion fired renders a
first-class negative-result report, not a hidden failure: the design treats a fired kill criterion
as equally publishable, and the report says so plainly at the top.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.studies.freeze import FrozenStudy
from reward_lens.studies.spec import StudyResult


def render_report(frozen: FrozenStudy, result: StudyResult, store: EvidenceStore) -> str:
    """Render a markdown study report from the frozen spec, the result, and the store."""
    spec = frozen.spec
    lines: list[str] = []
    lines.append(f"# {spec.title}")
    lines.append("")
    lines.append(
        f"Study `{frozen.study_id}` ({spec.science}). Frozen {frozen.frozen_at} at git "
        f"`{frozen.git_sha}`."
    )
    lines.append("")

    if result.killed:
        lines.append(
            "> Negative result. A kill criterion fired: "
            f"{', '.join(result.killed_by)}. This is a registered, publishable outcome, "
            "not a failure."
        )
        lines.append("")

    if result.voids:
        lines.append(
            f"> **VOID.** {len(result.voids)} registered "
            f"{'metric was' if len(result.voids) == 1 else 'metrics were'} not computed, so this "
            "study was not read. Nothing below is a finding about its hypotheses. Each void names "
            "what to repair."
        )
        lines.append("")

    lines.append("## Registered hypotheses and outcomes")
    lines.append("")
    lines.append("| Hypothesis | Registered prediction | Metric value | Outcome |")
    lines.append("|---|---|---|---|")
    for h in spec.hypotheses:
        p = h.prediction
        pred = f"{p.metric} {p.comparator} {p.threshold}"
        value = result.metrics.get(p.metric)
        vstr = f"{value:.4g}" if isinstance(value, (int, float)) else "n/a"
        outcome = result.outcomes.get(h.id, "void")
        badge = {"confirmed": "CONFIRMED", "refuted": "REFUTED", "void": "VOID"}[outcome]
        lines.append(f"| {h.statement} | `{pred}` | {vstr} | {badge} |")
    lines.append("")

    if spec.kill_criteria:
        lines.append("## Kill criteria")
        lines.append("")
        for k in spec.kill_criteria:
            v = result.metrics.get(k.metric)
            # A criterion that could not be evaluated is not a criterion that passed. Saying
            # "not fired" for a missing metric is the failure this vocabulary exists to prevent.
            state = {"fired": "FIRED", "passed": "not fired", "void": "VOID, not evaluated"}[
                result.kill_outcomes.get(k.id, "void")
            ]
            vstr = f"{v:.4g}" if isinstance(v, (int, float)) else "n/a"
            lines.append(
                f"- `{k.metric} {k.comparator} {k.threshold}` ({state}, value {vstr}): {k.description}"
            )
        lines.append("")

    if result.voids:
        lines.append("## Voids")
        lines.append("")
        lines.append("| What | Reason | Arc | Detail | Remedy |")
        lines.append("|---|---|---|---|---|")
        for vid, void in result.voids.items():
            arc = f"`{void.arc}`" if void.arc else "not declared"
            lines.append(
                f"| {vid} | `{void.reason.value}` | {arc} | {void.detail} | {void.remedy} |"
            )
        lines.append("")

    lines.append("## Adjudicating evidence")
    lines.append("")
    total_gpu = 0.0
    if result.evidence:
        lines.append("| Evidence | Observable | Value | Trust | Gauge | Calibrated |")
        lines.append("|---|---|---|---|---|---|")
        for eid in result.evidence:
            try:
                ev = store.get(eid)
            except KeyError:
                lines.append(f"| {eid} | (missing) | | | | |")
                continue
            total_gpu += ev.provenance.cost.gpu_seconds
            val = ev.value
            vstr = f"{val:.4g}" if isinstance(val, (int, float)) else type(val).__name__
            lines.append(
                f"| `{ev.id[:16]}...` | {ev.observable} | {vstr} | {ev.trust.name} | "
                f"{ev.gauge.value} | {'yes' if ev.is_calibrated else 'no'} |"
            )
        lines.append("")
    lines.append(
        f"Total metered GPU-seconds: {total_gpu:.2f}. {len(result.evidence)} evidence records."
    )
    if result.summary:
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(result.summary)
    return "\n".join(lines)


__all__ = ["render_report"]
