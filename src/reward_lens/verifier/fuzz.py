"""D5, `verifier.false_positive_rate` and `verifier.fp_catalogue`: where the verifier accepts a
wrong answer.

The measured pathologies that make this worth building are other people's, and they are large.
Buggy-verifier false-positive rates of **0.832 for math, 0.869 for JSON tool calls and 0.557 for
code unit tests**, against **0.000 for strict references**; `math-verify` at **FPR 1.000 on
partial-answer cases**. Those are published findings, cited here rather than reproduced: nothing in
this module has measured them. What this module does is give you the same number for *your*
verifier, and the catalogue of inputs behind it.

Four rungs, cheapest first.

**Rung 0, replay known exploit families.** The failure modes are not novel per grader. Substring
containment, a partial answer that happens to contain the gold, an empty answer against an empty
gold, a boxed answer with extra content, tolerance abuse on a numeric compare: these recur, so
replaying them is the first thing to try and it costs a handful of calls.

**Rung 1, `hypothesis`-driven search against a stricter reference.** A false positive is a case the
grader accepts and the reference rejects, which is a predicate, and a predicate is what property-
based search consumes. Two searches run, because they answer different questions. A volume pass
draws many candidates and classifies every one of them, which is what fills the *catalogue* and,
just as importantly, what supplies the denominator: the rate is `FP / (FP + TN)` over the cases the
reference rejects, so a draw both verifiers rejected is a true negative and has to be counted.
Then `find` returns the **shrunk** minimal hit as a Python object, which is what a bug report is.
Both searches are handed an explicit `Random`, because a reproducer that shows up on one run and
not the next is not a reproducer.

**Rung 2, coverage-guided fuzzing with `atheris`.** Not reachable in this environment: `atheris` is
not a declared dependency of the `verifier` extra and is not installed. The entry point exists and
raises a typed error naming the extra rather than pretending the rung ran, and the catalogue
records `coverage_guided_available=False` with the reason, so a reading at rung 1 cannot be
mistaken for a reading at rung 2.

**Rung 3, symbolic execution, routed by layer.** `crosshair` is not pointed at the grader. It is
pointed at the *layers of the grader it can reason about*: the parse, the normalisation and the
threshold. It is close to useless on a sympy equivalence check (the reasoning happens inside a
library it would have to symbolically execute) and inapplicable to anything that shells out to a
sandbox, so those layers are routed away with the reason recorded rather than run and reported as
clean. The distinction between `CONFIRMED` and `CANNOT_CONFIRM` is preserved end to end, because
crosshair exits 0 for both and only prints the difference under `--report_all`. Exit 0 does not
mean verified.

**Dual use.** A false-positive catalogue for a verifier is an exploit list for that verifier, and
it is exactly as useful to somebody optimising against the grader as to the person fixing it. So
`FPCatalogue.sensitive` defaults to True, `redacted()` returns the counts without the reproducers,
and `for_publication()` raises unless a `DisclosureDecision` has been recorded. That is a property
of the evidence row rather than a line in a document.

**Kill condition: if the fuzzer finds nothing on a verifier already known to be leaky, the search
is too weak.** That is what the random-mutation baseline is for: if property-based search does not
beat random character edits, it has not earned its calls.
"""

from __future__ import annotations

import enum
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from random import Random
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.errors import RewardLensError
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.extras import ExtraRequiredError, require_extra
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import CostModel
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.verifier import SENSITIVE_NOTE, SENSITIVE_SUBJECT_EXTRA, Rollout, VerifierUnderTest
from reward_lens.verifier.metamorphic import resolve_grader, score

# ---------------------------------------------------------------------------
# The stricter reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrictReference:
    """The oracle a false positive is defined against: what *should* have been rejected.

    ``basis`` is what makes it strict, in one sentence a reader can dispute: "exact equality after
    Unicode NFKC normalisation and whitespace stripping, with no substring matching and no numeric
    tolerance". An empty basis is refused with `REFERENCE_UNCERTIFIED`, because an FPR quoted
    against a reference whose strictness nobody stated is a number about two unknown things.

    The reference is not assumed correct. It is assumed *stricter*, which is a weaker and checkable
    claim: every case it accepts, the grader under test should also accept.
    `FPCatalogue.reference_disagreements` counts the cases where that failed, and a nonzero count
    means the two are not ordered and the FPR is not the quantity it claims to be.
    """

    name: str
    decide: Callable[..., bool]
    basis: str = ""

    def accepts(self, rollout: Rollout) -> bool:
        return bool(self.decide(**dict(rollout.inputs)))


# ---------------------------------------------------------------------------
# Rung 0: known exploit families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExploitFamily:
    """A named, citable way graders are known to be wrong, and how to generate instances.

    ``citation`` is not decoration. Every family here is a published failure mode, and a catalogue
    that mixes published families with somebody's guesses without saying which is which cannot be
    read.
    """

    name: str
    description: str
    generate: Callable[[Rollout, Random], Iterable[Rollout]]
    citation: str = ""


def _variant(rollout: Rollout, on: str, text: str, tag: str) -> Rollout:
    return Rollout(
        id=f"{rollout.id}#{tag}",
        inputs={**dict(rollout.inputs), on: text},
        score=None,
    )


def answer_exploit_families(
    on: str = "response", gold_key: str = "gold"
) -> tuple[ExploitFamily, ...]:
    """The families a free-text answer verifier is known to fall to.

    Each one is a shape rather than a string, so it instantiates against whatever gold the seed
    rollout carries. None of these is novel; that is the point of a rung-0 replay.
    """

    def gold_of(rollout: Rollout) -> str:
        value = rollout.inputs.get(gold_key)
        return value if isinstance(value, str) else ""

    def partial_answer(rollout: Rollout, rng: Random) -> Iterable[Rollout]:
        gold = gold_of(rollout)
        if not gold:
            return ()
        return (
            _variant(rollout, on, f"{gold} or {gold}1", "partial_or"),
            _variant(
                rollout, on, f"I am not sure, maybe {gold}, maybe something else", "partial_hedge"
            ),
            _variant(rollout, on, f"not {gold}", "partial_negated"),
        )

    def substring_containment(rollout: Rollout, rng: Random) -> Iterable[Rollout]:
        gold = gold_of(rollout)
        if not gold:
            return ()
        return (
            _variant(rollout, on, gold + "0", "contain_suffix"),
            _variant(rollout, on, "1" + gold, "contain_prefix"),
            _variant(rollout, on, f"{gold}{gold}", "contain_doubled"),
        )

    def empty_answer(rollout: Rollout, rng: Random) -> Iterable[Rollout]:
        return (
            _variant(rollout, on, "", "empty"),
            _variant(rollout, on, "   ", "whitespace_only"),
            _variant(rollout, on, "\\boxed{}", "empty_box"),
        )

    def prompt_echo(rollout: Rollout, rng: Random) -> Iterable[Rollout]:
        gold = gold_of(rollout)
        prompt = rollout.inputs.get("prompt")
        if not gold or not isinstance(prompt, str):
            return ()
        return (_variant(rollout, on, f"{prompt} {gold}", "prompt_echo"),)

    def numeric_tolerance(rollout: Rollout, rng: Random) -> Iterable[Rollout]:
        gold = gold_of(rollout)
        try:
            value = float(gold)
        except ValueError:
            return ()
        return (
            _variant(rollout, on, repr(value + 1e-7), "tolerance_epsilon"),
            _variant(rollout, on, repr(value * 10), "tolerance_decade"),
        )

    return (
        ExploitFamily(
            name="partial_answer",
            description="the gold appears inside a hedged or negated answer",
            generate=partial_answer,
            citation="math-verify measured at FPR 1.000 on partial-answer cases",
        ),
        ExploitFamily(
            name="substring_containment",
            description="the gold is a substring of a different answer",
            generate=substring_containment,
            citation="the containment bug behind the 0.832 math false-positive rate",
        ),
        ExploitFamily(
            name="empty_answer",
            description="an empty or whitespace answer, including an empty \\boxed{}",
            generate=empty_answer,
            citation="the empty-extraction path in the buggy-verifier family",
        ),
        ExploitFamily(
            name="prompt_echo",
            description="the response restates the prompt, which contains the answer",
            generate=prompt_echo,
            citation="SWE-Bench+ found 33.04% solution leakage of this shape",
        ),
        ExploitFamily(
            name="numeric_tolerance",
            description="a number close to but not equal to the gold, or off by a decade",
            generate=numeric_tolerance,
            citation="tolerance abuse on a float compare, in the code unit-test family",
        ),
    )


# ---------------------------------------------------------------------------
# Rung 3: symbolic execution, routed by layer
# ---------------------------------------------------------------------------


class LayerKind(enum.Enum):
    """What a layer of the grader is made of, which is what decides whether a solver can read it."""

    PARSE = enum.auto()
    NORMALISE = enum.auto()
    THRESHOLD = enum.auto()
    EQUIVALENCE = enum.auto()
    EXECUTION = enum.auto()


#: The layer kinds `crosshair` can actually reason about, and why each of the other two is out.
SYMBOLIC_TRACTABLE: frozenset[LayerKind] = frozenset(
    {LayerKind.PARSE, LayerKind.NORMALISE, LayerKind.THRESHOLD}
)

_INTRACTABLE_REASON: dict[LayerKind, str] = {
    LayerKind.EQUIVALENCE: (
        "the decision happens inside a computer-algebra library. Symbolically executing sympy "
        "means symbolically executing its own term rewriting, which does not terminate usefully; "
        "crosshair would return CANNOT_CONFIRM after burning the timeout, and CANNOT_CONFIRM read "
        "as a pass is exactly the failure this routing exists to prevent"
    ),
    LayerKind.EXECUTION: (
        "the layer shells out to a subprocess or a container. There is no Python-level path for a "
        "solver to explore, so a clean result would be a statement about the wrapper rather than "
        "about the harness"
    ),
}


@dataclass(frozen=True)
class GraderLayer:
    """One layer of a grader, with what a symbolic checker would be pointed at.

    ``target`` is what `crosshair check` takes: a file path, a `path.py:<line>`, or a fully
    qualified function name. Absent means the layer exists in the description but has no separately
    checkable entry point, which is itself worth reporting: a grader whose parse and threshold are
    inlined into one function cannot be checked a layer at a time.
    """

    name: str
    kind: LayerKind
    target: str | None = None
    note: str = ""


@register_payload
@dataclass(frozen=True)
class LayerRoute:
    """Where a layer was sent, and why. The record that the routing happened at all."""

    layer: str
    kind: str
    tool: str
    applicable: bool
    reason: str


def route_symbolic(layers: Sequence[GraderLayer]) -> tuple[LayerRoute, ...]:
    """Decide, per layer, whether a symbolic checker applies. No layer is run here.

    This is the whole content of "route by layer rather than point it at everything". Pointing
    crosshair at a grader containing a sympy call or a Docker exec gets a `CANNOT_CONFIRM` for the
    entire grader, and a `CANNOT_CONFIRM` at whole-grader granularity is indistinguishable from a
    clean result to anyone reading an exit code.
    """
    routes: list[LayerRoute] = []
    for layer in layers:
        if layer.kind not in SYMBOLIC_TRACTABLE:
            routes.append(
                LayerRoute(
                    layer=layer.name,
                    kind=layer.kind.name,
                    tool="",
                    applicable=False,
                    reason=_INTRACTABLE_REASON[layer.kind],
                )
            )
        elif not layer.target:
            routes.append(
                LayerRoute(
                    layer=layer.name,
                    kind=layer.kind.name,
                    tool="",
                    applicable=False,
                    reason=(
                        "the layer has no separately addressable entry point, so there is nothing "
                        "to point a checker at. Extract it into its own function to make it "
                        "checkable"
                    ),
                )
            )
        else:
            routes.append(
                LayerRoute(
                    layer=layer.name,
                    kind=layer.kind.name,
                    tool="crosshair",
                    applicable=True,
                    reason="pure Python over values a solver can model",
                )
            )
    return tuple(routes)


SymbolicStatus = Literal["refuted", "confirmed", "cannot_confirm", "no_contract", "error"]


@register_payload
@dataclass(frozen=True)
class SymbolicFinding:
    """What a symbolic checker said about one layer, with the three outcomes kept apart.

    ``confirmed`` means the postcondition held over *all* paths. ``cannot_confirm`` means no
    counterexample was found on the paths that were attempted, which is not a proof. crosshair
    exits 0 for both and prints the difference only under `--report_all`, so a caller reading exit
    codes cannot tell them apart. Keeping them separate here is the point of running it at all.
    """

    layer: str
    target: str
    status: SymbolicStatus
    messages: tuple[str, ...]
    exit_code: int
    seconds: float

    @property
    def is_proof(self) -> bool:
        return self.status == "confirmed"


def crosshair_available() -> bool:
    """Whether the `crosshair` executable or module is reachable in this interpreter."""
    if shutil.which("crosshair"):
        return True
    import importlib.util

    return importlib.util.find_spec("crosshair") is not None


def run_crosshair(
    layer: GraderLayer,
    *,
    per_condition_timeout: float = 3.0,
    timeout: float = 60.0,
) -> SymbolicFinding:
    """Run `crosshair check --report_all` on one layer and classify what came back.

    `--report_all` is not optional. Without it both `Confirmed over all paths.` and `Not
    confirmed.` are suppressed and the command exits 0 with no output, which is how a
    `CANNOT_CONFIRM` gets silently read as a proof.
    """
    require_extra("verifier", subsystem="D5 rung 3 (symbolic execution)")
    if not layer.target:
        return SymbolicFinding(
            layer=layer.name,
            target="",
            status="no_contract",
            messages=("the layer has no addressable target",),
            exit_code=-1,
            seconds=0.0,
        )
    cmd = [
        sys.executable,
        "-m",
        "crosshair",
        "check",
        "--report_all",
        "--per_condition_timeout",
        str(per_condition_timeout),
        layer.target,
    ]
    started = time.perf_counter()
    proc = subprocess.run(  # noqa: S603 - the argument vector is built here, not by a caller
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    elapsed = time.perf_counter() - started
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    status: SymbolicStatus
    if any(": error: " in ln for ln in lines):
        status = "refuted"
    elif any("Confirmed over all paths." in ln for ln in lines):
        status = "confirmed"
    elif any("Not confirmed." in ln for ln in lines):
        status = "cannot_confirm"
    elif proc.returncode == 2:
        status = "error"
    else:
        # Exit 0 with nothing to say means no contract was found to check. Reporting that as
        # "confirmed" would be the single most dangerous mistranslation available here.
        status = "no_contract"
    return SymbolicFinding(
        layer=layer.name,
        target=layer.target,
        status=status,
        messages=tuple(lines) or tuple(ln.strip() for ln in proc.stderr.splitlines() if ln.strip()),
        exit_code=proc.returncode,
        seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Rung 2: coverage-guided fuzzing
# ---------------------------------------------------------------------------

#: Why rung 2 does not run here. `atheris` is Google's coverage-guided fuzzer for Python and it is
#: not in the `verifier` extra's dependency list, so it is not installable through this package.
ATHERIS_GAP = (
    "atheris is not a declared dependency of the `verifier` extra and is not installed, so rung 2 "
    "did not run. A reading produced without it is a rung-1 reading and says so."
)


def atheris_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("atheris") is not None
    except (ImportError, ValueError):
        return False


def coverage_guided_search(*_args: Any, **_kwargs: Any) -> None:
    """Rung 2. Raises a typed error naming the extra rather than degrading to rung 1 silently."""
    if not atheris_available():
        raise ExtraRequiredError(
            "D5 rung 2 (coverage-guided fuzzing) needs `atheris`, which is not installed and is "
            "not declared in the `verifier` extra. Rung 1 is the highest rung reachable here, and "
            "the reading records that rather than presenting itself as rung 2."
        )
    raise NotImplementedError(  # pragma: no cover - unreachable until atheris is declared
        "atheris is importable but the rung-2 harness is not built: it needs a byte-string "
        "decoder from the fuzzer's input to a Rollout, which is grader-specific."
    )


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


class DisclosureRequired(RewardLensError):
    """A sensitive payload was asked for its publishable form with no decision recorded."""


@register_payload
@dataclass(frozen=True)
class DisclosureDecision:
    """Somebody decided this catalogue may leave the building, and signed it.

    The fields are the ones an audit needs: who, why, when, and how far. A boolean flag with no
    author is not a decision, it is a default that someone flipped.
    """

    decided_by: str
    reason: str
    scope: Literal["internal", "published"] = "internal"
    decided_at: str = ""

    def __post_init__(self) -> None:
        if not self.decided_by.strip() or not self.reason.strip():
            raise ValueError(
                "a disclosure decision needs a person and a reason. Publishing an exploit "
                "catalogue is a judgement about who is helped more, and an unsigned one cannot be "
                "reviewed."
            )
        if not self.decided_at:
            object.__setattr__(self, "decided_at", datetime.now(timezone.utc).isoformat())


@register_payload
@dataclass(frozen=True)
class FalsePositive:
    """One input the grader accepts and the stricter reference rejects."""

    family: str
    rung: int
    inputs: Mapping[str, Any]
    grader_score: float
    reference_accepts: bool
    seed: int = 0
    shrunk: bool = False
    source: str = "census"

    def rerun(
        self,
        grader: Callable[..., float] | VerifierUnderTest,
        reference: StrictReference,
    ) -> tuple[float, bool]:
        """Call both again on the recorded inputs.

        A catalogue entry that will not rerun is a note, not evidence.
        """
        fn, _, _ = resolve_grader(grader)
        kwargs = dict(self.inputs)
        return float(fn(**kwargs)), bool(reference.decide(**kwargs))

    def render(self) -> str:
        tag = " [shrunk]" if self.shrunk else ""
        return (
            f"{self.family} (rung {self.rung}, {self.source}){tag}\n"
            f"    grader {self.grader_score:.6g}, reference rejects\n"
            f"    {dict(self.inputs)!r}"
        )


@register_payload
@dataclass(frozen=True)
class FPCatalogue:
    """The reading: a false-positive rate and the inputs behind it. **Sensitive by default.**

    ``false_positive_rate`` is `FP / (FP + TN)`: the fraction of the cases the strict reference
    *rejects* that the grader nonetheless accepts. That is the definition the published 0.832,
    0.869 and 0.557 figures use, and it is not the fraction of all trials, which would move with
    the mix of the candidate pool rather than with the grader.

    ``reference_disagreements`` counts the other direction: cases the reference accepts and the
    grader rejects. The reference is assumed *stricter*, not correct, and a nonzero count here
    means the two verifiers are not ordered, so an FPR against this reference is not the quantity
    it claims to be. It is reported rather than silently tolerated.
    """

    grader: str
    reference: str
    reference_basis: str
    trials: int
    reference_rejects: int
    false_positives: int
    false_positive_rate: float
    reference_disagreements: int
    by_family: Mapping[str, int]
    entries: tuple[FalsePositive, ...]
    baseline_random_mutation_fpr: float
    baseline_random_mutation_hits: int
    symbolic_routes: tuple[LayerRoute, ...] = ()
    symbolic_findings: tuple[SymbolicFinding, ...] = ()
    coverage_guided_available: bool = False
    coverage_guided_gap: str = ATHERIS_GAP
    rung: int = 1
    sensitive: bool = True
    sensitive_note: str = SENSITIVE_NOTE
    disclosure: DisclosureDecision | None = None
    withheld: int = 0

    # -- dual use ----------------------------------------------------------

    def redacted(self) -> "FPCatalogue":
        """The counts without the reproducers. What a rendered card gets by default.

        Everything that makes the number auditable survives: how many trials, how many the
        reference rejected, how many the grader let through, and which families they came from. The
        inputs themselves do not, because those are the exploit.
        """
        return replace(
            self,
            entries=(),
            withheld=len(self.entries),
            sensitive=False,
            symbolic_findings=tuple(
                replace(f, messages=("<withheld: contains a counterexample>",))
                for f in self.symbolic_findings
                if f.status == "refuted"
            )
            + tuple(f for f in self.symbolic_findings if f.status != "refuted"),
        )

    def for_publication(self) -> "FPCatalogue":
        """The unredacted catalogue, and only with a decision recorded."""
        if self.disclosure is None:
            raise DisclosureRequired(
                f"{type(self).__name__} for {self.grader!r} carries {len(self.entries)} "
                f"reproducible ways to make the grader wrong and no recorded decision to publish "
                f"them. Attach a DisclosureDecision naming who decided and why, or call "
                f"`redacted()` for the counts without the inputs."
            )
        return self

    def with_disclosure(self, decision: DisclosureDecision) -> "FPCatalogue":
        return replace(self, disclosure=decision)

    # -- presentation ------------------------------------------------------

    @property
    def beats_baseline(self) -> bool:
        """Whether the search found more than random character edits did. The kill condition."""
        return self.false_positive_rate > self.baseline_random_mutation_fpr

    def render(self, *, include_entries: bool = False) -> str:
        lines = [
            f"false-positive catalogue for {self.grader} against {self.reference}",
            f"    reference basis: {self.reference_basis}",
            f"    {self.false_positives} false positives over {self.reference_rejects} inputs the "
            f"reference rejects: FPR {self.false_positive_rate:.3f}",
            f"    {self.trials} candidates tried at rung {self.rung}",
            f"    baseline (random character mutation): FPR "
            f"{self.baseline_random_mutation_fpr:.3f} from "
            f"{self.baseline_random_mutation_hits} hits",
        ]
        for family, count in sorted(self.by_family.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {family:<28} {count:>6}")
        if self.reference_disagreements:
            lines.append(
                f"    WARNING: the reference accepted {self.reference_disagreements} cases the "
                f"grader rejected, so the two are not ordered and this FPR is not well defined"
            )
        for route in self.symbolic_routes:
            state = f"-> {route.tool}" if route.applicable else "not applicable"
            lines.append(f"    layer {route.layer:<20} {route.kind:<12} {state}")
        for finding in self.symbolic_findings:
            lines.append(f"    {finding.layer:<20} crosshair: {finding.status}")
        if not self.coverage_guided_available:
            lines.append(f"    rung 2 not run: {self.coverage_guided_gap}")
        if self.entries and not include_entries:
            lines.append(f"    {len(self.entries)} reproducers withheld. {self.sensitive_note}")
        elif include_entries:
            for entry in self.entries:
                lines.append("    " + entry.render().replace("\n", "\n    "))
        if self.withheld:
            lines.append(f"    {self.withheld} reproducers withheld from this view")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The search space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """How rung 1 draws a candidate the grader might wrongly accept.

    ``build`` is handed the `hypothesis.strategies` module and returns a strategy over input
    mappings, which is the whole of the coupling to hypothesis. Keeping it a callable rather than a
    strategy object means importing this module does not import hypothesis, which matters because
    the core install does not have it.
    """

    name: str
    build: Callable[[Any], Any]


def mutation_space(
    seeds: Sequence[Rollout],
    *,
    on: str = "response",
    name: str = "seeded_text_mutation",
) -> SearchSpace:
    """Draw a seed rollout and rewrite one string input with generated text.

    Seeded rather than free-form because a grader's accept region is a thin set: drawing arbitrary
    Unicode finds nothing, and finding nothing on a leaky verifier is this instrument's kill
    condition rather than a clean bill of health. Starting from real accepted inputs and perturbing
    them puts the search where the boundary is.
    """
    if not seeds:
        raise ValueError("a seeded search space needs at least one seed rollout")

    def build(st: Any) -> Any:
        alphabet = st.characters(min_codepoint=32, max_codepoint=126, blacklist_categories=("Cs",))
        text = st.text(alphabet=alphabet, max_size=12)
        index = st.sampled_from(range(len(seeds)))

        def affix(i: int, prefix: str, suffix: str, drop: int, keep: int) -> dict[str, Any]:
            """Wrap, truncate and un-prefix. The structured half of the space."""
            body = str(seeds[i].inputs.get(on, ""))[drop:][: keep or None]
            return {**dict(seeds[i].inputs), on: prefix + body + suffix}

        def edit(i: int, position: int, char: str, kind: int) -> dict[str, Any]:
            """One character inserted, deleted or substituted at a drawn position.

            The unstructured half, and it is here because the declared baseline is exactly this
            operator applied at random. A search space that cannot reach where random byte edits
            reach is misdesigned rather than interestingly conservative, and the baseline is then
            measuring the gap in the space instead of the value of the search. With both halves
            present the comparison is about hits per draw, which is the question the baseline is
            supposed to answer.
            """
            body = str(seeds[i].inputs.get(on, ""))
            if not body:
                return {**dict(seeds[i].inputs), on: char}
            at = position % len(body)
            if kind == 0:
                mutated = body[:at] + char + body[at:]
            elif kind == 1:
                mutated = body[:at] + body[at + 1 :]
            else:
                mutated = body[:at] + char + body[at + 1 :]
            return {**dict(seeds[i].inputs), on: mutated}

        return st.one_of(
            st.builds(
                affix,
                index,
                text,
                text,
                # Deleting a prefix matters as much as truncating a suffix: a normaliser that
                # rewrites `.5` to `0.5` is only reachable by removing a leading character.
                st.integers(min_value=0, max_value=4),
                st.integers(min_value=0, max_value=64),
            ),
            st.builds(
                edit,
                index,
                st.integers(min_value=0, max_value=255),
                st.characters(min_codepoint=32, max_codepoint=126, blacklist_categories=("Cs",)),
                st.integers(min_value=0, max_value=2),
            ),
        )

    return SearchSpace(name=name, build=build)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: D5's envelope. A false positive that does not reproduce is not a false positive, so the grader
#: has to replay to the same verdict between the moment the search found it and the moment somebody
#: reads the catalogue.
D5_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
)


class FalsePositiveFuzzing(BaseObservable):
    """D5. Search for inputs the grader accepts and a stricter reference rejects.

    **Kill condition: if the fuzzer finds nothing on a verifier already known to be leaky, the
    search is too weak.** The random-mutation baseline is the operational form of that test, and
    `FPCatalogue.beats_baseline` is where it is answered.
    """

    name = "verifier.false_positive_rate"
    version = "1.0"
    quantity = "verifier.false_positive_rate"
    capabilities = Capability.SCORES
    requires = {Component.GRADER: Access.QUERY, Component.GOLD: Access.QUERY}
    substrates = frozenset({Substrate.PROGRAM, Substrate.PROCEDURAL, Substrate.COMPOSITE})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = D5_ENVELOPE
    invariance = "trivial"
    invariance_relation = Relation("invariant")
    baselines = ("random_mutation",)
    rung = 1
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "property-based testing (Claessen and Hughes 2000) against a differential oracle"
    deviations = (
        "the oracle is a stricter reference verifier rather than a specification, so a finding is "
        "a disagreement between two implementations and the reference is assumed stricter, not "
        "correct",
        "rung 2 is declared and not reachable: atheris is not a dependency of the verifier extra, "
        "and the reading records the gap rather than presenting rung 1 as rung 2",
        "rung 3 is routed by layer, so a clean symbolic result covers the parse, normalisation and "
        "threshold layers only and says nothing about a sympy or sandbox layer",
    )

    def __init__(
        self,
        grader: Callable[..., float] | VerifierUnderTest,
        reference: StrictReference,
        seeds: Sequence[Rollout],
        *,
        families: Sequence[ExploitFamily] | None = None,
        space: SearchSpace | None = None,
        layers: Sequence[GraderLayer] = (),
        accept_threshold: float = 0.5,
        max_examples: int = 300,
        max_entries: int = 1000,
        seed: int = 0,
        run_symbolic: bool = False,
        per_condition_timeout: float = 3.0,
    ) -> None:
        self.fn, self.subject, self.grader_name = resolve_grader(grader)
        self.reference = reference
        self.seeds = list(seeds)
        self.families = tuple(families if families is not None else answer_exploit_families())
        self.space = space
        self.layers = tuple(layers)
        self.accept_threshold = accept_threshold
        self.max_examples = max_examples
        self.max_entries = max_entries
        self.seed = seed
        self.run_symbolic = run_symbolic
        self.per_condition_timeout = per_condition_timeout

    def accepts(self, rollout: Rollout) -> tuple[bool, float]:
        value = score(self.fn, rollout)
        return value >= self.accept_threshold, value

    # -- preflight ---------------------------------------------------------

    def preflight(self, ctx: Context) -> PreflightResult:
        base = super().preflight(ctx)
        if not base.ok:
            return base
        family_calls = 2 * len(self.seeds) * sum(3 for _ in self.families)
        search_calls = 2 * self.max_examples if self.space is not None else 0
        calls = family_calls + search_calls + 2 * self.max_examples
        return PreflightResult(
            instrument=self.name,
            ok=True,
            rung=self.rung,
            cost=CostModel(
                calls=calls,
                note=(
                    f"about {calls:,} paired calls to the grader and the reference. No GPU. "
                    f"Rung 2 is not reachable here: {ATHERIS_GAP}"
                ),
            ),
            regime=base.regime,
            unchecked=base.unchecked,
            notes=(
                f"{len(self.families)} exploit families at rung 0"
                + (f", {self.max_examples} draws at rung 1" if self.space else ""),
            ),
        )

    # -- the measurement ---------------------------------------------------

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx if ctx is not None else Context(signal=self.subject, readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        if not self.reference.basis.strip():
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.REFERENCE_UNCERTIFIED,
                detail=(
                    f"the reference {self.reference.name!r} states no basis for its strictness, so "
                    f"a false-positive rate against it would be a number about two unknown things."
                ),
                remedy=(
                    "set `StrictReference.basis` to one sentence saying what makes it strict, for "
                    "example 'exact equality after Unicode NFKC normalisation and whitespace "
                    "stripping, with no substring match and no numeric tolerance'. If you cannot "
                    "write that sentence, the reference is not known to be stricter and the FPR is "
                    "not defined."
                ),
            )
        if not self.seeds:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no seed rollouts were supplied, so there is nothing to perturb",
                remedy=(
                    "supply at least one rollout the grader accepts. Both rung 0 and rung 1 work "
                    "outward from a real accepted input; searching from nothing finds nothing, and "
                    "finding nothing is this instrument's kill condition rather than a clean bill "
                    "of health."
                ),
            )
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        rng = Random(self.seed)
        entries: list[FalsePositive] = []
        by_family: dict[str, int] = {}
        trials = 0
        reference_rejects = 0
        disagreements = 0
        seen: set[tuple[tuple[str, str], ...]] = set()

        def consider(
            candidate: Rollout, family: str, rung: int, *, shrunk: bool, source: str
        ) -> bool:
            nonlocal trials, reference_rejects, disagreements
            key = tuple(sorted((k, repr(v)) for k, v in candidate.inputs.items()))
            if key in seen:
                return False
            seen.add(key)
            trials += 1
            accepted, value = self.accepts(candidate)
            ref = self.reference.accepts(candidate)
            if not ref:
                reference_rejects += 1
                if accepted:
                    by_family[family] = by_family.get(family, 0) + 1
                    if len(entries) < self.max_entries:
                        entries.append(
                            FalsePositive(
                                family=family,
                                rung=rung,
                                inputs=dict(candidate.inputs),
                                grader_score=value,
                                reference_accepts=False,
                                seed=self.seed,
                                shrunk=shrunk,
                                source=source,
                            )
                        )
                    return True
            elif not accepted:
                disagreements += 1
            return False

        # -- the corpus itself, before anything is generated ----------------
        # The seeds are real graded items, so a disagreement on one is worth more than a
        # disagreement on anything synthesised from it. This pass is also what makes the ordering
        # assumption checkable at all: the reference is assumed *stricter*, meaning everything it
        # accepts the grader accepts too, and the only place to test that is on inputs the grader
        # was actually built for.
        for rollout in self.seeds:
            consider(rollout, "corpus_replay", 0, shrunk=False, source="corpus")

        # -- rung 0: replay the known families -----------------------------
        for family in self.families:
            for rollout in self.seeds:
                for candidate in family.generate(rollout, rng):
                    consider(candidate, family.name, 0, shrunk=False, source="exploit_replay")

        # -- rung 1: property-based search ---------------------------------
        rung = 0
        if self.space is not None:
            rung = 1
            for inputs in self._search_volume(self.space):
                consider(
                    Rollout(id=f"search:{len(seen)}", inputs=inputs),
                    self.space.name,
                    1,
                    shrunk=False,
                    source="hypothesis.given",
                )
            shrunk_inputs = self._shrink(self.space)
            if shrunk_inputs is not None:
                # `_shrink` only ever returns a genuine false positive, so a False here means the
                # volume pass had already drawn these exact inputs. The shrunk example is the one
                # a grader author debugs, so in that case the existing entry is re-tagged rather
                # than duplicated: one row, correctly labelled as the minimal one.
                recorded = consider(
                    Rollout(id="shrunk", inputs=shrunk_inputs),
                    self.space.name,
                    1,
                    shrunk=True,
                    source="hypothesis.find",
                )
                if not recorded:
                    target = dict(shrunk_inputs)
                    for k, entry in enumerate(entries):
                        if dict(entry.inputs) == target:
                            entries[k] = replace(
                                entry, shrunk=True, source="hypothesis.find", rung=1
                            )
                            break

        # -- the declared baseline -----------------------------------------
        base_hits, base_trials = self._random_mutation_baseline(
            trials=max(trials, 1), rng=Random(self.seed + 1)
        )

        # -- rung 3: symbolic execution, routed --------------------------------
        routes = route_symbolic(self.layers)
        findings: tuple[SymbolicFinding, ...] = ()
        if self.run_symbolic and crosshair_available():
            findings = tuple(
                run_crosshair(layer, per_condition_timeout=self.per_condition_timeout)
                for layer, route in zip(self.layers, routes)
                if route.applicable
            )
            if findings:
                rung = 3

        fpr = len(entries) / reference_rejects if reference_rejects else float("nan")
        catalogue = FPCatalogue(
            grader=self.grader_name,
            reference=self.reference.name,
            reference_basis=self.reference.basis,
            trials=trials,
            reference_rejects=reference_rejects,
            false_positives=len(entries),
            false_positive_rate=fpr,
            reference_disagreements=disagreements,
            by_family=dict(by_family),
            entries=tuple(entries),
            baseline_random_mutation_fpr=(base_hits / base_trials if base_trials else float("nan")),
            baseline_random_mutation_hits=base_hits,
            symbolic_routes=routes,
            symbolic_findings=findings,
            coverage_guided_available=atheris_available(),
            rung=rung,
        )
        return ctx.emit(
            catalogue,
            uncertainty=Uncertainty(
                n=trials,
                method="differential search against a stricter reference",
            ),
            subject_extra=dict(SENSITIVE_SUBJECT_EXTRA),
        )

    # -- rung 1 internals --------------------------------------------------

    def _search_volume(self, space: SearchSpace) -> list[Mapping[str, Any]]:
        """Draw `max_examples` candidates and return **every one of them**, hit or not.

        `find` returns one example. A catalogue needs many, so the volume pass drives generation
        with `@given` restricted to the generate phase and collects what it drew. That is
        generation without shrinking, which is the division of labour: this pass produces the
        catalogue, `_shrink` produces the bug report.

        Returning every draw rather than only the hits is not tidiness, it is the denominator. The
        false-positive rate is `FP / (FP + TN)` over the cases the reference rejects, so a draw the
        reference rejected and the grader also rejected is a true negative and belongs in the
        count. Filtering here would drop every true negative rung 1 produced and inflate the rate
        towards one.
        """
        require_extra("verifier", subsystem="D5 rung 1 (property-based search)")
        from hypothesis import HealthCheck, given, settings
        from hypothesis import Phase as HypothesisPhase
        from hypothesis import strategies as st

        drawn: list[Mapping[str, Any]] = []

        @settings(
            max_examples=self.max_examples,
            database=None,
            deadline=None,
            derandomize=True,
            phases=[HypothesisPhase.generate],
            suppress_health_check=list(HealthCheck),
        )
        @given(space.build(st))
        def probe(inputs: Mapping[str, Any]) -> None:
            drawn.append(dict(inputs))

        probe()
        return drawn

    def _shrink(self, space: SearchSpace) -> Mapping[str, Any] | None:
        """The smallest candidate the grader accepts and the reference rejects, via `find`.

        `find` returns the shrunk falsifying example as a Python object, here the input mapping
        itself. Returns None when the search finds nothing inside `max_examples` draws, which is
        the honest answer and is not the same as the grader being clean.
        """
        require_extra("verifier", subsystem="D5 rung 1 (property-based search)")
        from hypothesis import HealthCheck, find, settings
        from hypothesis import strategies as st
        from hypothesis.errors import NoSuchExample, Unsatisfiable

        def is_false_positive(inputs: Mapping[str, Any]) -> bool:
            candidate = Rollout(id="candidate", inputs=inputs)
            accepted, _ = self.accepts(candidate)
            return accepted and not self.reference.accepts(candidate)

        try:
            return find(
                space.build(st),
                is_false_positive,
                # `find` draws from global entropy unless it is handed a generator, which makes
                # the shrunk example appear on one run and not the next. A reproducer that is only
                # sometimes there is not a reproducer, so the seed is passed explicitly.
                random=Random(self.seed),
                settings=settings(
                    max_examples=self.max_examples,
                    database=None,
                    deadline=None,
                    suppress_health_check=list(HealthCheck),
                ),
            )
        except (NoSuchExample, Unsatisfiable):
            return None

    # -- the baseline ------------------------------------------------------

    def _random_mutation_baseline(self, *, trials: int, rng: Random) -> tuple[int, int]:
        """The declared baseline: random character edits, same number of attempts.

        The comparison that matters. Property-based search costs more than mangling bytes, and if
        it does not find more false positives than mangling bytes does, it has not earned the
        difference.
        """
        on_keys = [k for k, v in self.seeds[0].inputs.items() if isinstance(v, str)]
        if not on_keys:
            return 0, 0
        key = on_keys[0]
        hits = 0
        done = 0
        for _ in range(trials):
            seed_rollout = self.seeds[rng.randrange(len(self.seeds))]
            text = str(seed_rollout.inputs.get(key, ""))
            mutated = _random_edit(text, rng)
            candidate = Rollout(id="baseline", inputs={**dict(seed_rollout.inputs), key: mutated})
            done += 1
            accepted, _ = self.accepts(candidate)
            if accepted and not self.reference.accepts(candidate):
                hits += 1
        return hits, done


def _random_edit(text: str, rng: Random) -> str:
    """One random character insertion, deletion or substitution. The dumbest possible mutator."""
    if not text:
        return chr(rng.randrange(32, 127))
    kind = rng.randrange(3)
    i = rng.randrange(len(text))
    ch = chr(rng.randrange(32, 127))
    if kind == 0:
        return text[:i] + ch + text[i:]
    if kind == 1:
        return text[:i] + text[i + 1 :]
    return text[:i] + ch + text[i + 1 :]


def false_positive_fuzzing(
    grader: Callable[..., float] | VerifierUnderTest,
    reference: StrictReference,
    seeds: Sequence[Rollout],
    *,
    ctx: Context | None = None,
    **kwargs: Any,
) -> Reading:
    """Run D5 and return the Reading. The one-call form, for a card renderer."""
    return FalsePositiveFuzzing(grader, reference, seeds, **kwargs).estimate(ctx)


__all__ = [
    "ATHERIS_GAP",
    "D5_ENVELOPE",
    "SYMBOLIC_TRACTABLE",
    "DisclosureDecision",
    "DisclosureRequired",
    "ExploitFamily",
    "FPCatalogue",
    "FalsePositive",
    "FalsePositiveFuzzing",
    "GraderLayer",
    "LayerKind",
    "LayerRoute",
    "SearchSpace",
    "StrictReference",
    "SymbolicFinding",
    "SymbolicStatus",
    "answer_exploit_families",
    "atheris_available",
    "coverage_guided_search",
    "crosshair_available",
    "false_positive_fuzzing",
    "mutation_space",
    "route_symbolic",
    "run_crosshair",
]
