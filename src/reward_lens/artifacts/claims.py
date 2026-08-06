"""The manuscript claims checker (R-anti-self-deception).

This is the structural fix for the PAPER_DISCREPANCIES failure class: v1's paper numbers disagreed
with the CSVs (stale appendix tables, transposed rows, invented SNR values) and nobody could tell
which was authoritative. Here the evidence store is the single source of truth, and a document may
not claim a number the store does not contain. A claim is a value tagged with the Evidence id it
came from; the checker loads that Evidence, extracts the comparable value, and verifies the claim
within a tolerance. A tag pointing at an id the store does not have, or a value that disagrees with
the stored one, is a failure. It runs in CI over the repo's own docs.

Claim syntax, chosen to be readable in prose and unambiguous to parse:

    [[claim value=-0.171 ev=ev:ab12... field=per_model_mean_rho.Skywork tol=0.01]]

``value`` is the number as written in the prose; ``ev`` is the Evidence id; ``field`` (optional) is
a dotted path into the Evidence value when it is a dict or dataclass (omit it when the value is a
scalar); ``tol`` (optional) overrides the default absolute tolerance. The checker also verifies bare
``ev:...`` references resolve, so a citation to a nonexistent measurement is caught even without a
value.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reward_lens.core.store import EvidenceStore, default_store

_CLAIM_RE = re.compile(r"\[\[claim\s+(?P<body>[^\]]+)\]\]")
_KV_RE = re.compile(r"(\w+)=(\S+)")
_BARE_EV_RE = re.compile(r"(?<![\w:])(ev:[0-9a-f]{8,})")


@dataclass
class ClaimResult:
    """The verification outcome for one claim."""

    claimed: float | None
    evidence_id: str
    field: str | None
    actual: Any
    ok: bool
    message: str


@dataclass
class ClaimReport:
    """The full result of checking a document."""

    results: list[ClaimResult] = field(default_factory=list)
    unresolved_refs: list[str] = field(default_factory=list)
    unbound: list["UnboundNumber"] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results) and not self.unresolved_refs and not self.unbound

    @property
    def n_failures(self) -> int:
        return (
            sum(1 for r in self.results if not r.ok) + len(self.unresolved_refs) + len(self.unbound)
        )

    def render(self) -> str:
        lines = [
            f"Claims checked: {len(self.results)}. "
            f"Unbound numbers: {len(self.unbound)}. "
            f"Failures: {self.n_failures}."
        ]
        for r in self.results:
            mark = "ok" if r.ok else "FAIL"
            lines.append(f"  [{mark}] {r.evidence_id} {r.field or ''}: {r.message}")
        for ref in self.unresolved_refs:
            lines.append(f"  [FAIL] {ref}: referenced but not in the store")
        for u in self.unbound:
            lines.append(f"  [FAIL] unbound number {u}")
        return "\n".join(lines)


def _extract_field(value: Any, dotted: str | None) -> Any:
    if dotted is None:
        return value
    cur = value
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def check_text(
    text: str, store: EvidenceStore | None = None, default_tol: float = 1e-6
) -> ClaimReport:
    """Check every tagged claim in ``text`` against the store, returning a ClaimReport.

    A claim fails if its Evidence id is not in the store, if the named field cannot be extracted, or
    if the claimed value differs from the stored value by more than the tolerance. Bare ``ev:...``
    references that do not resolve are collected separately so a dangling citation is also caught.
    """
    store = store if store is not None else default_store()
    report = ClaimReport()
    tagged_ids: set[str] = set()

    for m in _CLAIM_RE.finditer(text):
        kv = dict(_KV_RE.findall(m.group("body")))
        ev_id = kv.get("ev", "")
        tagged_ids.add(ev_id)
        fld = kv.get("field")
        # A malformed tag is a failed claim, not a crash. A checker whose job is to catch bad
        # numbers must not be stoppable by one: the docs carry at least one template tag with a
        # placeholder value, and until now that took the whole run down with a ValueError.
        try:
            claimed = float(kv["value"]) if "value" in kv else None
            tol = float(kv.get("tol", default_tol))
        except ValueError as exc:
            report.results.append(
                ClaimResult(None, ev_id, fld, None, False, f"malformed claim tag ({exc})")
            )
            continue
        if ev_id not in store:
            report.results.append(
                ClaimResult(claimed, ev_id, fld, None, False, "evidence id not in the store")
            )
            continue
        ev = store.get(ev_id)
        try:
            actual = _extract_field(ev.value, fld)
        except (KeyError, AttributeError) as exc:
            report.results.append(
                ClaimResult(claimed, ev_id, fld, None, False, f"field '{fld}' not found ({exc})")
            )
            continue
        if claimed is None:
            report.results.append(ClaimResult(None, ev_id, fld, actual, True, "reference resolves"))
            continue
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            report.results.append(
                ClaimResult(claimed, ev_id, fld, actual, False, "stored value is not numeric")
            )
            continue
        diff = abs(actual_f - claimed)
        ok = diff <= tol
        msg = (
            f"claimed {claimed:g}, stored {actual_f:g}, |diff|={diff:g} <= tol {tol:g}"
            if ok
            else f"claimed {claimed:g} but stored {actual_f:g} (|diff|={diff:g} > tol {tol:g})"
        )
        report.results.append(ClaimResult(claimed, ev_id, fld, actual_f, ok, msg))

    # Bare ev: references not already covered by a claim tag: verify they resolve.
    for m in _BARE_EV_RE.finditer(text):
        ref = m.group(1)
        if ref in tagged_ids:
            continue
        if ref not in store:
            report.unresolved_refs.append(ref)

    return report


def check_files(paths: list[str | Path], store: EvidenceStore | None = None) -> ClaimReport:
    """Check a set of documents, aggregating into one report (the CI entry point)."""
    combined = ClaimReport()
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        rep = check_text(text, store)
        combined.results.extend(rep.results)
        combined.unresolved_refs.extend(rep.unresolved_refs)
        combined.unbound.extend(replace(u, path=str(p)) for u in find_unbound_numbers(text))
    return combined


# ---------------------------------------------------------------------------
# Unbound numbers
# ---------------------------------------------------------------------------
#
# A tagged claim that disagrees with the store is caught above. The commoner and more damaging
# failure is a number that was never tagged at all: it looks like a measurement, it reads like a
# measurement, and it came from a draft. During the preparation of this project's own build
# specification a summarising fetch fabricated plausible numbers on at least one occasion, which is
# the reason this half exists.
#
# The rule is the one the project writes to: a number in prose is either traceable to an evidence
# id or explicitly labelled illustrative in the same sentence. Anything else is unbound.

#: Decimals only. Integers in prose are overwhelmingly counts, years, section numbers and step
#: indices, and flagging them produces noise that trains people to ignore the checker. A decimal
#: point is the signature of a measurement.
# The trailing guard rejects a word character or a further decimal group, so "0.5.1" and "0.5x"
# are not matched, but a plain full stop is fine: a number at the end of a sentence is the most
# common position a measurement appears in, and an earlier form of this pattern silently skipped
# every one of them.
_UNBOUND_NUM_RE = re.compile(r"(?<![\w.$/])(\d{1,4}\.\d+)\s*(%|pp|nats|bits)?(?!\w|\.\d)")

#: Words that make a number honest without an evidence id. They must appear in the same sentence,
#: not merely somewhere on the page, or a single disclaimer at the top would launder the document.
_ILLUSTRATIVE = (
    "illustrative",
    "for example",
    "e.g.",
    "hypothetical",
    "placeholder",
    "worked example",
    "suppose",
    "imagine",
)

# "say" is an illustrative marker only in the idiom "say 0.5", where it introduces a made-up value.
# As a plain substring it was `"say "`, which exempted **every number in any sentence containing the
# ordinary verb**: "what this does not say is that the mass is 0.214" passed the gate with a bare
# 0.214 in it. Found by the X7 write-up, whose own section heading tripped it.
#
# This is the gate `docs/content/findings.md` runs against **with no baseline at all**, so a hole
# here is not backlog, it is the published artifact's only check. Narrowed to require a number right
# after the word, optionally through a comma or an article, which is what the idiom looks like.
_ILLUSTRATIVE_SAY_RE = re.compile(r"\bsay,?\s+(?:about\s+|roughly\s+|around\s+)?[-+]?\d")

_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")
# Narrow on purpose: an unprefixed two-component decimal is a measurement far more often than a
# version, so only a "v" prefix or a third component makes it a version.
_VERSION_RE = re.compile(r"\bv\d+\.\d+(\.\d+)*\w*\b|\b\d+\.\d+\.\d+\w*\b")

# A release of this library named in prose without the "v": "the 2.0 API", "a 1.0 script",
# "porting an existing 1.0 workflow". `_VERSION_RE` deliberately does not mask a bare `1.0`,
# because `1.0` is far more often a perfect AUC or a unit cosine than a release name, and this
# repository's docs contain both. What separates them is grammar rather than the digits: a release
# name sits in a noun phrase, after a determiner or before a noun, while a measurement sits in a
# predicate, after "at", "of", "near", or a metric's name. So this matches the phrase, not the
# number, and the fractional part must be exactly one zero.
#
# Measured over `docs/content` with the CI exclusions: 195 unbound numbers before, 172 after. Every
# one of the 23 it removes is a release reference and the check was by hand, line by line. It does
# not reach the four that carry no grammatical signal at all ("a real bug in 1.0.", "figures from
# 1.0, recompute them", "So in 2.0 a measurement", "Once your scripts run on 2.0"); those stay in
# the baseline, which is where a false positive with no rule behind it belongs.
_RELEASE_WORDS = (
    "API|APIs|library|libraries|toolkit|toolkits|workflow|workflows|script|scripts|"
    "call|calls|path|paths|names|version|release|releases|equivalent|measurement|"
    "compatibility|codebase|module|modules|package|primitive|primitives|era"
)
_RELEASE_RE = re.compile(
    r"\b(?:the|a|an|its|this|that|every|existing|preserved|legacy|old|original|classic)\s+"
    r"\d{1,2}\.0\b"
    rf"|\b\d{{1,2}}\.0\s+(?:{_RELEASE_WORDS})\b"
    r"|\b\d{1,2}\.0\s+(?:does|spoke|refuses|refused|supersedes|superseded|different)\b",
    re.I,
)

#: Cross-references. In a manuscript, "section 3.1" and "F1" are addresses rather than
#: measurements, and a long write-up is full of them.
_XREF_RE = re.compile(r"(?:§|\bsections?\s+|\bpart\s+|\bappendix\s+)\s*\d+(\.\d+)*", re.I)

#: Licence identifiers. "Apache-2.0" and "MPL-2.0" are names, not measurements.
_LICENCE_RE = re.compile(r"\b[A-Z][A-Za-z]*(-[A-Za-z]+)*-\d+\.\d+\b")


@dataclass
class UnboundNumber:
    """A number in prose with no evidence id and no illustrative label."""

    value: str
    line: int
    context: str
    path: str = ""

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.path else f"line {self.line}"
        return f"{where}: {self.value} in {self.context!r}"


def _blank_spans(text: str, pattern: re.Pattern[str]) -> str:
    """Replace every match with spaces, preserving offsets so line numbers stay correct."""
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def find_unbound_numbers(text: str) -> list[UnboundNumber]:
    """Find decimal numbers in prose that are neither claim-tagged nor labelled illustrative.

    Code, inline code, link targets and version strings are excluded, because a number inside them
    is not a claim about the world. Everything else that looks like a measurement has to say where
    it came from.
    """
    masked = _blank_spans(text, _FENCE_RE)
    masked = _blank_spans(masked, _INLINE_CODE_RE)
    masked = _blank_spans(masked, _LINK_TARGET_RE)
    masked = _blank_spans(masked, _CLAIM_RE)
    masked = _blank_spans(masked, _VERSION_RE)
    masked = _blank_spans(masked, _RELEASE_RE)
    masked = _blank_spans(masked, _XREF_RE)
    masked = _blank_spans(masked, _LICENCE_RE)

    out: list[UnboundNumber] = []
    for m in _UNBOUND_NUM_RE.finditer(masked):
        start = m.start()
        # The sentence around it, read from the ORIGINAL text so an illustrative marker is visible
        # even when it sits inside inline code. Markdown prose ends sentences at a newline at least
        # as often as at a full stop, so both bound the window; without the newline bound a single
        # "illustrative" anywhere in the file would launder every number after it.
        left = max(
            text.rfind(". ", 0, start),
            text.rfind(".\n", 0, start),
            text.rfind("\n", 0, start),
        )
        right = min(
            (i for i in (text.find(". ", m.end()), text.find("\n", m.end())) if i >= 0),
            default=len(text),
        )
        sentence = text[(left + 1 if left >= 0 else 0) : right]
        low = sentence.lower()
        if any(word in low for word in _ILLUSTRATIVE):
            continue
        if _ILLUSTRATIVE_SAY_RE.search(low):
            continue
        if "ev:" in sentence:
            continue
        out.append(
            UnboundNumber(
                value=m.group(0).strip(),
                line=text.count("\n", 0, start) + 1,
                context=" ".join(sentence.split())[:120],
            )
        )
    return out


def baseline_key(u: "UnboundNumber") -> str:
    """The identity of a known-unbound number, for the ratchet.

    Keyed on the file, the value and the surrounding sentence rather than the line number, because
    a line number shifts whenever anything above it is edited and a baseline that drifts is a
    baseline nobody trusts. Editing the sentence does re-fire the check, which is correct: a
    rewritten sentence is a new claim.
    """
    # Relative to the working directory, so a baseline written by `reward-lens-claims docs/` and
    # one written from an absolute path are the same file. Without this the ratchet silently
    # exempts nothing whenever the invocation changes shape.
    try:
        where = os.path.relpath(u.path)
    except ValueError:  # different drive on Windows
        where = u.path
    return f"{Path(where).as_posix()}\t{u.value}\t{u.context}"


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def main(argv: list[str] | None = None) -> int:
    """The CI entry point. Exits nonzero on any failed claim, dangling ref, or unbound number.

    Documented as the CI entry point since 2.0 and never wired into a workflow, which is how a
    guard against fabricated numbers spent two releases not running.

    The docs carry a backlog of numbers written before this check existed. Failing on all of them
    would make the gate permanently red, and a permanently red gate is one people learn to ignore,
    which is worse than no gate. So `--baseline` exempts a recorded set and fails on anything new:
    the backlog is visible, it can only shrink, and the check is live from the first commit.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="reward-lens-claims", description=__doc__)
    parser.add_argument("paths", nargs="+", help="files or directories of markdown to check")
    parser.add_argument(
        "--no-unbound",
        action="store_true",
        help="check only tagged claims and evidence references, not unbound numbers",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="a file of already-known unbound numbers to exempt; anything new still fails",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help=(
            "skip files whose path contains SUBSTRING. For pages that teach the claim syntax: "
            "their example tags are illustrations, not claims, and a checker that cannot tell "
            "the difference would make its own documentation unpublishable."
        ),
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline file from what is found now, and exit 0",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        files.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])
    if args.exclude:
        files = [f for f in files if not any(s in str(f) for s in args.exclude)]

    report = check_files(files)  # type: ignore[arg-type]

    if args.write_baseline:
        if args.baseline is None:
            parser.error("--write-baseline needs --baseline")
        keys = sorted(baseline_key(u) for u in report.unbound)
        args.baseline.write_text(
            "# Numbers in prose with no evidence id, recorded before the check existed.\n"
            "# This list may shrink and must not grow. Bind one and delete its line.\n"
            + "\n".join(keys)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(keys)} baselined unbound numbers to {args.baseline}")
        return 0

    known = load_baseline(args.baseline) if args.baseline else set()
    if known:
        n_before = len(report.unbound)
        report.unbound = [u for u in report.unbound if baseline_key(u) not in known]
        print(f"baseline exempted {n_before - len(report.unbound)} of {n_before} unbound numbers.")

    print(report.render())
    if args.no_unbound:
        return 0 if (all(r.ok for r in report.results) and not report.unresolved_refs) else 1
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the CI job and the acceptance test
    raise SystemExit(main())


__all__ = [
    "ClaimResult",
    "ClaimReport",
    "UnboundNumber",
    "check_text",
    "check_files",
    "find_unbound_numbers",
    "main",
]
