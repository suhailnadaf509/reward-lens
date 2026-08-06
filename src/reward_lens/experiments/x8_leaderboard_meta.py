"""X8: pooling the 2026 re-analyses that found published leaderboard orderings unresolved.

Six groups published re-analyses of public model leaderboards during 2026. They used five different
statistical traditions and worked on six different benchmarks, they arrived independently, none of
them is from a frontier lab, and every one of them concluded that a material fraction of the
orderings the field quotes does not survive its own preferred inferential procedure. Nobody has
pooled them and no leaderboard operator has responded to any of them.

This module pools them, and the interesting part is not the pooled number.

**What the analysis is actually for.** Each of the six answers a question about the leaderboard it
looked at. None of them answers the question a reader wants answered, which is about a leaderboard
nobody has re-analysed yet: if I go and check the next one, what fraction of its orderings should I
expect to fail? That is a prediction interval, not a confidence interval, and the six papers between
them do not contain one. Getting it is the contribution.

**The self-reference, stated up front because it is the strongest thing here and not a weakness to
bury.** Six studies is few. Four of the six turn out to report a numerator and a denominator that
can be pooled at all, so the realised k is smaller still. At that k, tau2 is estimated with three
degrees of freedom and its confidence interval is enormous, and the prediction interval uses
t(k-2) = t(2) = 4.30 rather than z = 1.96. A meta-analysis of a handful of studies that came back
with a tight interval and a confident headline would be committing, in miniature, exactly the error
the six papers are collectively about: quoting a point estimate as though the sample behind it were
larger than it is. So the wide interval this produces is not a disappointing result. It is the
result, and it is the same result the six papers report, one level up.

**Extraction discipline.** Every effect size here is a numerator and a denominator quoted from a
specific line of `field-scan-2026/13-EVAL-SCIENCE.md`, and `verify_quotes` re-reads that file at
those line numbers and checks the quoted text is present before anything is computed. A study whose
finding is stated without a denominator is refused rather than imputed, and two of the six are.
The refusals are reported in the write-up with the same prominence as the inclusions, because "two
of the six papers complaining about reporting standards do not report a denominator" is itself a
finding about this literature.

The dossier is a read-only reference held outside this repository, so there is no path the library
can assume. Point ``REWARD_LENS_EVAL_DOSSIER`` at it, or pass ``--dossier``; without it the module
imports fine and every entry point refuses by name rather than reading a file that is not there.

Run it:

    REWARD_LENS_EVAL_DOSSIER=/path/to/13-EVAL-SCIENCE.md \
        python -m reward_lens.experiments.x8_leaderboard_meta

which verifies the quotes, freezes the spec, runs the analysis, and prints the write-up section.
The freeze happens before the analysis in the same process and the ordering is not adjustable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reward_lens.stats import meta
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec, SubjectQuery

# ---------------------------------------------------------------------------
# The evidence base
# ---------------------------------------------------------------------------

#: The environment variable that supplies the dossier's location.
DOSSIER_ENV_VAR = "REWARD_LENS_EVAL_DOSSIER"

#: The file the effect sizes are quoted from.
DOSSIER_NAME = "13-EVAL-SCIENCE.md"

#: The dossier the effect sizes come from, or None when it has not been located. It is a read-only
#: reference held outside this repository, so there is no path an installed copy of the library
#: could guess: set `DOSSIER_ENV_VAR` to point at it, or pass a path to the functions below. There
#: is deliberately no fallback, because a default pointing at a file that is not there would turn a
#: missing evidence base into a confusing error somewhere further down.
_dossier_env = os.environ.get(DOSSIER_ENV_VAR)
DOSSIER: Path | None = Path(_dossier_env) if _dossier_env else None

#: The dossier as it stood when these line numbers were read. Recorded rather than enforced: a
#: mismatch means the file has been edited and the line numbers need rechecking, which
#: `verify_quotes` will detect anyway by failing to find the quoted text.
DOSSIER_SHA256 = "0dcbcf05d420d2099170ee7338dfa7e2ff305f88d0fe286aec495a9112d99c42"

#: Where the table itself lives, for a reader who wants the source.
DOSSIER_SECTION = "c.7.3, lines 1047-1072"


class DossierNotConfigured(RuntimeError):
    """Raised when the analysis is asked to run and no path to the evidence base is available."""


def resolve_dossier(dossier: Path | None = None) -> Path:
    """The dossier path, or a refusal that names the variable which supplies it.

    Every number in this module is a quotation, so there is nothing to compute when the file the
    quotations come from has not been located. Failing here, by name, is better than defaulting to
    a path that happens to exist on one machine.
    """
    resolved = dossier if dossier is not None else DOSSIER
    if resolved is None:
        raise DossierNotConfigured(
            f"no path to the evidence base. Set {DOSSIER_ENV_VAR} to the location of "
            f"{DOSSIER_NAME}, or pass an explicit path. Every effect size in this analysis is a "
            f"quotation from that file, checked against it before anything is computed, so the "
            f"analysis does not run without it."
        )
    return Path(resolved)


@dataclass(frozen=True)
class Extraction:
    """One of the six re-analyses, with the line its numbers were read off.

    ``unresolved`` and ``total`` are the numerator and denominator of the estimand. Both are None
    when the dossier states the finding without a count, which is not a gap in the extraction but a
    property of the source, and `status` records which.

    ``sources`` is a tuple of (line number, quoted substring) pairs, one for every load-bearing
    number, and all of them are checked against the file before the analysis runs. Substrings rather
    than whole lines because the dossier is markdown with emphasis markers inside the table cells,
    and more than one pair where the numerator and the denominator are on different lines, because
    a denominator taken on trust is exactly the kind of number that turns out later to have been
    remembered rather than read.
    """

    key: str
    arxiv: str
    authors: str
    date: str
    benchmark: str
    tradition: str
    estimand: str
    unresolved: int | None
    total: int | None
    sources: tuple[tuple[int, str], ...]
    status: str  # "extracted" or "no-denominator"
    note: str = ""

    @property
    def line(self) -> int:
        """The line the headline finding is on; the rest are corroboration."""
        return self.sources[0][0]

    @property
    def quote(self) -> str:
        return self.sources[0][1]

    @property
    def proportion(self) -> float | None:
        if self.unresolved is None or not self.total:
            return None
        return self.unresolved / self.total

    @property
    def extracted(self) -> bool:
        return self.status == "extracted"

    def cite(self) -> str:
        lines = ", ".join(str(n) for n, _ in self.sources)
        return f"arXiv {self.arxiv} ({self.authors}, {self.date}), dossier line(s) {lines}"


#: The six, in the order the dossier's table gives them.
#:
#: Rows 3 and 6 carry no denominator. Row 3 says "most pairwise comparisons among 12 models", which
#: bounds the fraction above 0.5 but gives no count, and turning "most" into a number would be
#: inventing one. Row 6 says "all observed NF4-FP16 deltas fall below the implied MDE", which is a
#: proportion of 1.0 over a denominator the dossier never states. Both are carried here in full so
#: the write-up can report them as refusals rather than quietly dropping them, and so that anyone
#: who fetches the two papers can complete the extraction by filling in two integers.
SIX: tuple[Extraction, ...] = (
    Extraction(
        key="chandrahas",
        arxiv="2607.04429",
        authors="Chandrahas",
        date="2026-07-05",
        benchmark="MMLU, 9 models",
        tradition="paired permutation with multiple-comparison correction",
        estimand="adjacent leaderboard-rank gaps that are not significant after correction",
        unresolved=3,
        total=8,
        sources=(
            (1054, "3 of the 8 adjacent leaderboard-rank gaps are not statistically significant"),
            (1054, "correcting for the 36 pairwise comparisons the ranking implies"),
        ),
        status="extracted",
        note=(
            "Correction is over the 36 pairwise comparisons the ranking of nine models implies. "
            "The denominator of 8 is the adjacent-rank gaps, which is the leaderboard-ordering "
            "estimand rather than the all-pairs one."
        ),
    ),
    Extraction(
        key="kotawala",
        arxiv="2605.30315",
        authors="Kotawala",
        date="2026-05-28",
        benchmark="MMLU-Pro top-10",
        tradition="inverted paired power tests, resolution ratio q = N/N*",
        estimand="adjacent-rank pairs unresolved at (alpha, 1-beta) = (0.05, 0.8)",
        unresolved=4,
        total=9,
        sources=(
            (1055, "4 of 9 MMLU-Pro top-10 adjacent-rank pairs are unresolved"),
            (1055, "11 of 40 Open LLM Leaderboard v1 pairwise comparisons"),
            (1055, "The MMLU-Pro count rises to 6/9 under real subject-level clustering"),
        ),
        status="extracted",
        note=(
            "This paper reports three counts on the same line: 11 of 40 all-pairs comparisons on "
            "Open LLM Leaderboard v1, 4 of 9 adjacent-rank pairs on MMLU-Pro, and 6 of 9 for the "
            "same pairs under subject-level clustering. The adjacent-rank count is taken under the "
            "inclusion rule; the other two are pre-registered sensitivity analyses."
        ),
    ),
    Extraction(
        key="mandujano_reyes",
        arxiv="2607.25257",
        authors="Mandujano Reyes",
        date="2026-07-28",
        benchmark="a standard LLM benchmark leaderboard, 12 models",
        tradition="Bayesian posterior over IRT ability (Laplace-PSN-IRT)",
        estimand="pairwise comparisons not statistically distinguishable",
        unresolved=None,
        total=None,
        sources=(
            (1056, "most pairwise comparisons among 12 models"),
            (1056, "are not statistically distinguishable despite differing point-estimate ranks"),
        ),
        status="no-denominator",
        note=(
            "'Most' bounds the fraction above 0.5 and gives no count. Twelve models imply 66 "
            "pairwise comparisons and 11 adjacent-rank ones, but which of those 'most' refers to "
            "is not stated, so both the numerator and the denominator would be inferred. Excluded "
            "from the pool; entered at the bound 6 of 11 in one clearly labelled sensitivity run."
        ),
    ),
    Extraction(
        key="chacon_sartori",
        arxiv="2605.27789",
        authors="Chacon Sartori & Garcia",
        date="2026-05-27",
        benchmark="multi-hop RAG, 400 questions",
        tradition="cluster-aware inference with Bonferroni",
        estimand="apparently-significant comparisons that do not survive",
        unresolved=3,
        total=4,
        sources=(
            (1041, "4 of 4 significant → 1 of 4 significant"),
            (
                1038,
                "four semantic-baseline comparisons look significant; cluster-aware inference "
                "leaves only one",
            ),
        ),
        status="extracted",
        note=(
            "Read as 4 minus 1: a binomial test made all four semantic-baseline comparisons look "
            "significant and cluster-aware inference left one Bonferroni-significant result "
            "(dossier line 1038), so three of four did not survive. The comparisons are between "
            "retrieval methods rather than between leaderboard rows, which is the widest departure "
            "from the common estimand in the set and is recorded as such."
        ),
    ),
    Extraction(
        key="dlugosz",
        arxiv="2605.28700",
        authors="Dlugosz, Oliveira & Diaz-Rodriguez",
        date="2026-05-27",
        benchmark="GSM-Symbolic, 20 open-weight models",
        tradition="generalised linear mixed model, per-question random effects",
        estimand="models whose reported performance change is not significant under a GLMM",
        unresolved=10,
        total=20,
        sources=(
            (724, "only half exhibit statistically"),
            (723, "Re-evaluating **20 open-weight models using Generalised Linear"),
            (733, "~5 of 20 models left with an unexplained"),
        ),
        status="extracted",
        note=(
            "'Only half' of 20 is read as 10, which is the dossier's own arithmetic at line 733. "
            "This is the one approximate numerator in the set and the sensitivity analysis moves "
            "it to 9 and to 11. The paper's further finding, that roughly half the remaining "
            "significant cases are explained by an integer-magnitude confound, would put the count "
            "at about 15 of 20; that is a confound rather than an inference failure, so it is a "
            "sensitivity run and not the primary."
        ),
    ),
    Extraction(
        key="zhuang",
        arxiv="2605.28873",
        authors="Zhuang, Li & Fan",
        date="2026-05-25",
        benchmark="4-bit quantization suites (NF4 against FP16)",
        tradition="paired-binary minimum detectable effect bound",
        estimand="observed deltas falling below the implied MDE",
        unresolved=None,
        total=None,
        sources=(
            (855, "all observed NF4-FP16 deltas fall below the implied MDE"),
            (1059, "all observed NF4-FP16 deltas fall below the implied MDE"),
        ),
        status="no-denominator",
        note=(
            "'All' fixes the proportion at 1.0 and leaves the denominator unstated anywhere in the "
            "dossier. A proportion of 1.0 over an unknown n carries no weight, and inventing an n "
            "would set the weight this study receives. Excluded, and completable by anyone who "
            "fetches the paper and counts the deltas in its table."
        ),
    ),
)

#: The framing claims, quoted from the dossier and verified alongside the effect sizes.
#:
#: These are counted by the dossier, not by this module. Deriving "five traditions" from the six
#: strings in the table above would give six, because two of the six papers invert a power
#: calculation and I described them differently; the dossier groups them and its grouping is the
#: cited one. Anything a write-up asserts about the shape of the set comes from here.
FRAMING_SOURCES: tuple[tuple[int, str], ...] = (
    (1061, "Six groups. Five different statistical traditions"),
    (1062, "Six different benchmarks (MMLU, Open LLM Leaderboard v1, MMLU-Pro, a"),
    (1063, "Every single one finds that a material"),
    (1064, "Not one is from a frontier"),
    (1065, "Not one has been cited by a leaderboard operator, and not one leaderboard has changed"),
    (1066, "as of 2026-07-30 I found no announcement to the contrary"),
    (1069, "this is a coordinated, unclaimed finding lying on the ground"),
)

#: Counts taken from FRAMING_SOURCES rather than recomputed. Five and six, per lines 1061-1062.
N_TRADITIONS = 5
N_BENCHMARKS = 6

#: The limit on the "nobody has responded" claim, in the dossier's own words at section c.11. It
#: travels with the claim wherever the claim goes; a scan that did not look somewhere is not a scan
#: that found nothing there.
RESPONSE_CAVEAT = (
    "The dossier's own qualification travels with this: no announcement was found as of 2026-07-30, "
    "and its section c.11 records that leaderboard changelogs, Hugging Face discussion threads and "
    "LMArena were not searched. The correct claim is that no response was found, not that none "
    "exists."
)

#: The inclusion rule, written down so it can be checked against what the code does.
INCLUSION_RULE = (
    "One estimate per study, so no group is counted twice. Where a study reports several counts, "
    "take the adjacent-rank one, because a published ordering is a chain of adjacent comparisons "
    "and that is the estimand the claim 'the leaderboard ordering does not survive' is about. A "
    "study whose finding is stated without both a numerator and a denominator is excluded and "
    "reported as excluded; no count is inferred from a quantifier word."
)

ESTIMAND = (
    "The fraction of the model-comparison claims a re-analysis examined that its own preferred "
    "inferential procedure leaves unresolved."
)

#: The threshold the registered predictions are stated against. A quarter is chosen because it is
#: the point below which "a material fraction of published orderings does not survive" stops being
#: the natural reading of the six papers, and because it is not near any of the six observed
#: proportions, so the verdict does not turn on a rounding decision.
NULL_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Verification: no number enters the analysis unless the dossier still says it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteCheck:
    key: str
    line: int
    quote: str
    found: bool
    actual: str


def verify_quotes(dossier: Path | None = None) -> list[QuoteCheck]:
    """Re-read the dossier and confirm every quoted string is on the line it is attributed to.

    This exists because the failure mode it prevents has already happened once in this project's
    history: a summarising fetch produced plausible numbers that were not in the source. A quote
    that has drifted, a line number that has moved, or a transcription that never matched will fail
    here rather than survive into a write-up wearing a citation.

    It has already earned its place. The first run of this module failed here, on the Dlugosz row,
    because the sentence carrying that study's numerator wraps across two lines of the dossier and
    the quote had been recorded against only the second half of it. That is a small error and it is
    exactly the size of error that otherwise reaches print.

    Raises if the file is missing, because running the analysis without it would mean computing over
    numbers nothing has checked.
    """
    dossier = resolve_dossier(dossier)
    if not dossier.exists():
        raise FileNotFoundError(
            f"the evidence base is not at {dossier}. Set {DOSSIER_ENV_VAR} to its path. "
            f"The analysis does not run without it: every effect size in it is a quotation."
        )
    lines = dossier.read_text(encoding="utf-8").splitlines()
    checks = []
    sourced: list[tuple[str, int, str]] = [
        (e.key, number, quote) for e in SIX for number, quote in e.sources
    ]
    sourced += [("framing", number, quote) for number, quote in FRAMING_SOURCES]
    for key, number, quote in sourced:
        actual = lines[number - 1] if 0 < number <= len(lines) else ""
        checks.append(
            QuoteCheck(key=key, line=number, quote=quote, found=quote in actual, actual=actual)
        )
    return checks


def dossier_sha256(dossier: Path | None = None) -> str:
    return hashlib.sha256(resolve_dossier(dossier).read_bytes()).hexdigest()


def assert_quotes_verify(dossier: Path | None = None) -> list[QuoteCheck]:
    """`verify_quotes`, but a failure stops the run. Called before anything is computed."""
    checks = verify_quotes(dossier)
    bad = [c for c in checks if not c.found]
    if bad:
        detail = "; ".join(f"{c.key} at line {c.line}: found {c.actual[:80]!r}" for c in bad)
        raise ValueError(
            f"{len(bad)} of {len(checks)} quoted effect sizes are not on the lines they are "
            f"attributed to. Recheck the extraction against the dossier before anything is "
            f"computed: {detail}"
        )
    return checks


# ---------------------------------------------------------------------------
# The frozen spec
# ---------------------------------------------------------------------------

#: Stated in the spec's own notes so it is hashed along with the predictions. Preregistration is
#: worth exactly what it is honest about, and overclaiming here would be the same error the piece is
#: about.
FREEZE_HONESTY = (
    "What this freeze does and does not protect. The six effect sizes were read from the dossier's "
    "table before this spec was written, because the spec's inclusion rule and kill criterion refer "
    "to them, so inclusion is not preregistered and is not claimed to be. What is preregistered is "
    "every analysis choice downstream of the counts: the estimand, the analysis scale and its "
    "continuity correction, the tau2 estimator, the prediction-interval convention, the sensitivity "
    "set, and the three thresholds. Beyond that, and worth stating because it is checkable: while "
    "designing the estimator the analyst hand-computed the fixed-effect pooled value and the "
    "DerSimonian-Laird tau2 for the primary set, so H1 and H2 were registered with an approximation "
    "of their outcomes already in view and are protection against steering rather than risky "
    "predictions. H3 names a quantity that had not been computed in any form when this was frozen."
)


def study_spec() -> StudySpec:
    """The frozen contract: three predictions, two kill criteria, one analysis path.

    H1 is the substantive claim and it is weak, deliberately and admittedly. H2 is the
    methodological payload: if the prediction interval is not much wider than the confidence
    interval then reporting only the latter would have cost nothing and this module has no point.
    H3 is the one that was genuinely open at freeze time, and it is the one that decides whether the
    piece can say anything at all about heterogeneity.
    """
    return StudySpec(
        id="x8-leaderboard-meta",
        title=(
            "Do the 2026 leaderboard re-analyses agree, and what do they imply for a leaderboard "
            "nobody has re-analysed yet?"
        ),
        science="S00-eval-science",
        hypotheses=(
            Hypothesis(
                id="H1",
                statement=(
                    "Pooling the extractable 2026 re-analyses, the fraction of published "
                    "model-comparison claims that do not survive proper inference exceeds a "
                    "quarter."
                ),
                prediction=Prediction(
                    metric="pooled_proportion",
                    comparator=">",
                    threshold=NULL_FRACTION,
                    rationale=(
                        "Every one of the six papers reports a material fraction unresolved and "
                        "the smallest extractable proportion in the set is 3 of 8. This prediction "
                        "is registered so that the estimand and the inclusion rule cannot be "
                        "adjusted after seeing the pooled value, not because its outcome was in "
                        "doubt."
                    ),
                ),
            ),
            Hypothesis(
                id="H2",
                statement=(
                    "The prediction interval for a new re-analysis is substantially wider than the "
                    "confidence interval for the mean, so a report giving only the confidence "
                    "interval would overstate what the six studies establish."
                ),
                prediction=Prediction(
                    metric="width_ratio_logit",
                    comparator=">=",
                    threshold=1.8,
                    rationale=(
                        "With k below five the critical value alone is t(k-2) against z, which is "
                        "2.20 at k = 4, before any between-study variance is added. The threshold "
                        "is set at 1.8 rather than at 2.0 so the verdict does not sit on the "
                        "boundary of the estimator's own arithmetic."
                    ),
                ),
            ),
            Hypothesis(
                id="H3",
                statement=(
                    "At the realised number of studies the data cannot rule out between-study "
                    "heterogeneity large enough to matter, whatever the tau2 point estimate turns "
                    "out to be."
                ),
                prediction=Prediction(
                    metric="tau2_upper_limit_logit",
                    comparator=">",
                    threshold=0.25,
                    rationale=(
                        "A tau2 of 0.25 on the logit scale is a between-study standard deviation "
                        "of 0.5, which moves a proportion of 0.5 to roughly 0.38 or 0.62. If the "
                        "upper confidence limit is above that, no reading of these studies "
                        "supports treating them as estimating one common fraction. This quantity "
                        "had not been computed when this spec was frozen."
                    ),
                ),
            ),
        ),
        analysis="reward_lens.experiments.x8_leaderboard_meta.analyse",
        subjects=SubjectQuery(
            datasets=tuple(f"arxiv:{e.arxiv}" for e in SIX),
            extra={
                "estimand": ESTIMAND,
                "inclusion_rule": INCLUSION_RULE,
                "source": f"{DOSSIER_NAME} section {DOSSIER_SECTION}",
                "source_sha256": DOSSIER_SHA256,
                "analysis_scale": "logit, continuity correction 0.5 applied to every study",
                "tau2_estimator": "Paule-Mandel, with DerSimonian-Laird and REML reported beside it",
                "prediction_interval_rule": "t(k-2), Higgins-Thompson-Spiegelhalter",
                "confidence_interval_rule": "inverse-variance z, Hartung-Knapp as a sensitivity",
                "sensitivity_runs": [
                    "kotawala at 11 of 40 (all pairs, Open LLM Leaderboard v1)",
                    "kotawala at 6 of 9 (adjacent-rank under subject-level clustering)",
                    "dlugosz at 9 of 20 and at 11 of 20 (the 'only half' rounding)",
                    "dlugosz at 15 of 20 (after the integer-magnitude confound)",
                    "mandujano reyes entered at the bound 6 of 11",
                    "double-arcsine analysis scale",
                    "logit with no continuity correction",
                    "DerSimonian-Laird and REML tau2",
                    "Hartung-Knapp confidence interval",
                ],
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1",
                metric="k_extracted",
                comparator="<",
                threshold=3,
                description=(
                    "Fewer than three studies with a numerator and a denominator. The prediction "
                    "interval needs t(k-2) and does not exist below three, so the analysis is "
                    "refused and the piece becomes a report on why six papers could not be pooled."
                ),
            ),
            KillCriterion(
                id="K2",
                metric="sensitivity_sign_flips",
                comparator=">",
                threshold=0,
                description=(
                    "Any pre-registered sensitivity run that moves the pooled fraction across the "
                    "quarter threshold. If one does, the headline is reported as unresolved rather "
                    "than as a result, because a conclusion that depends on which of two defensible "
                    "extractions was used is a conclusion about the analyst."
                ),
            ),
        ),
        version=1,
        notes=FREEZE_HONESTY,
    )


# ---------------------------------------------------------------------------
# Freezing, and the clean-tree refusal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeOutcome:
    """A frozen study plus whether its git stamp is worth anything.

    Not a `Reading`, and deliberately not: the fifteen refusal reasons are conditions a measurement
    can hit, and "the working tree has uncommitted changes" is a condition of the process rather
    than of the evidence. Encoding it as a measurement refusal would put a process fact into a
    vocabulary built for measurement facts. It is carried as a flag instead, and `provisional` is
    the flag a reader has to see.
    """

    frozen: FrozenStudy
    clean: bool
    dirty_paths: tuple[str, ...]
    detail: str

    @property
    def provisional(self) -> bool:
        return not self.clean

    def render(self) -> str:
        head = f"study_id  {self.frozen.study_id}\nspec_hash {self.frozen.spec_hash}\nfrozen_at {self.frozen.frozen_at}\ngit_sha   {self.frozen.git_sha}"
        return head + "\n" + self.detail


def _dirty_paths(repo_dir: Path) -> list[str]:
    """Uncommitted or untracked paths, from a direct call rather than a cached sha."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_dir), stderr=subprocess.DEVNULL
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def repo_root() -> Path:
    """The checkout this module lives in, found from the file rather than from the cwd.

    This assumes the source layout, ``<root>/src/reward_lens/experiments/``. Run from an installed
    package there is no checkout above the module and the path this returns is not a repository,
    which is not an error: the git stamp then reads ``unknown`` and the spec hash, which is a
    content hash of the plan and is the part the preregistration rests on, is unaffected.
    """
    return Path(__file__).resolve().parents[3]


def freeze_x8(repo_dir: Path | None = None, frozen_at: str | None = None) -> FreezeOutcome:
    """Freeze the spec and record honestly whether the git stamp means anything.

    The campaign's freeze refuses outright on a dirty tree, and that is the right behaviour there:
    it is freezing a spend commitment against a commit somebody will later check out. Here the tree
    is expected to be dirty, because an analysis is normally frozen while the checkout it sits in is
    still being edited, and refusing would mean the spec never gets hashed at all.

    So the refusal is recorded rather than raised. The spec hash is the part that carries the
    preregistration and it is exact either way: it is a content hash of the predictions and the
    thresholds and it does not depend on the working tree. The git sha is the part that is
    provisional, and a `+dirty` suffix on it is the library's own marker for exactly that. A freeze
    with `provisional` set needs the git stamp reapplied against a clean commit before the study id
    is quoted anywhere as reproducible; the spec hash does not change when that happens, so the
    reapplication is a stamping step and not a re-freeze.
    """
    root = Path(repo_dir) if repo_dir is not None else repo_root()
    dirty = _dirty_paths(root)
    frozen = freeze(study_spec(), repo_dir=str(root), frozen_at=frozen_at)
    if dirty:
        shown = ", ".join(sorted(dirty)[:5])
        detail = (
            f"PROVISIONAL git stamp: {len(dirty)} uncommitted or untracked paths at freeze time "
            f"({shown}{', ...' if len(dirty) > 5 else ''}). The spec hash {frozen.spec_hash} is "
            f"exact and is what the predictions are registered under. The git sha is not, and "
            f"carries the +dirty marker. Reapply the stamp at commit time: re-run this freeze on a "
            f"clean tree and confirm the spec hash is unchanged."
        )
    else:
        detail = "Clean tree at freeze time; the git sha and the spec hash are both exact."
    return FreezeOutcome(
        frozen=frozen, clean=not dirty, dirty_paths=tuple(sorted(dirty)), detail=detail
    )


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityRun:
    """One pre-registered variation, with what it does to the headline.

    A run that cannot be computed on the data as extracted is recorded with ``available`` False and
    a reason, rather than raising or being quietly dropped. Dropping it would leave a pre-registered
    variation missing from the reported set with nothing saying so, and a reader counting the runs
    against the frozen list would find one short and no explanation.
    """

    name: str
    description: str
    pooled_p: float
    ci_p: tuple[float, float]
    prediction_p: tuple[float, float]
    tau2: float
    k: int
    available: bool = True
    unavailable_reason: str = ""

    @property
    def crosses(self) -> bool:
        """Whether this variation puts the pooled fraction on the other side of the threshold."""
        return self.available and self.pooled_p <= NULL_FRACTION


@dataclass
class X8Result:
    """Everything the write-up quotes, so nothing in it can come from anywhere else."""

    primary: meta.ProportionMeta
    extracted: tuple[Extraction, ...]
    refused: tuple[Extraction, ...]
    baselines: dict[str, Any]
    power: dict[str, float]
    sensitivity: tuple[SensitivityRun, ...]
    metrics: dict[str, float]
    outcomes: dict[str, str]
    kill_outcomes: dict[str, str]
    quote_checks: tuple[QuoteCheck, ...] = ()
    dossier_sha256: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _primary_counts() -> tuple[list[int], list[int], list[str]]:
    rows = [e for e in SIX if e.extracted]
    return (
        [int(e.unresolved) for e in rows],  # type: ignore[arg-type]
        [int(e.total) for e in rows],  # type: ignore[arg-type]
        [f"{e.authors} {e.arxiv}" for e in rows],
    )


def _fit(
    counts: Sequence[int],
    totals: Sequence[int],
    labels: Sequence[str],
    **kw: Any,
) -> meta.ProportionMeta:
    out = meta.proportion_meta(counts, totals, labels=labels, **kw)
    if not isinstance(out, meta.ProportionMeta):
        raise RuntimeError(f"the primary fit refused: {out.render()}")
    return out


def _sensitivity(name: str, description: str, fit: meta.ProportionMeta) -> SensitivityRun:
    return SensitivityRun(
        name=name,
        description=description,
        pooled_p=fit.pooled_p,
        ci_p=fit.ci_p,
        prediction_p=fit.prediction_p,
        tau2=fit.fit.het.tau2,
        k=fit.fit.k,
    )


def _unavailable(name: str, description: str, reason: str) -> SensitivityRun:
    return SensitivityRun(
        name=name,
        description=description,
        pooled_p=float("nan"),
        ci_p=(float("nan"), float("nan")),
        prediction_p=(float("nan"), float("nan")),
        tau2=float("nan"),
        k=0,
        available=False,
        unavailable_reason=reason,
    )


def analyse(dossier: Path | None = None) -> X8Result:
    """Run the frozen analysis plan. Nothing here chooses anything the spec did not fix.

    The order is: verify the quotes, then pool, then the baselines, then the power at the realised
    k, then the sensitivity set. The power calculation is deliberately last and deliberately not
    conditioned on the observed effect: it asks what this collection of studies could have detected,
    which is a question about the design, and it is reported whichever way the result falls.
    """
    checks = assert_quotes_verify(dossier)
    counts, totals, labels = _primary_counts()
    extracted = tuple(e for e in SIX if e.extracted)
    refused = tuple(e for e in SIX if not e.extracted)
    k = len(counts)

    if k < 3:  # K1, checked before the fit rather than after it
        raise RuntimeError(
            f"kill criterion K1 fired: only {k} of {len(SIX)} studies carry a numerator and a "
            f"denominator, and the prediction interval needs t(k-2). Report the refusal."
        )

    primary = _fit(counts, totals, labels)
    y, v = meta.proportion_effects(counts, totals, scale="logit", correction=0.5)

    # Baselines. Three, because they fail in different ways and a reader will reach for all three.
    naive_mean = float(np.mean([c / n for c, n in zip(counts, totals)]))
    naive_sd = float(np.std([c / n for c, n in zip(counts, totals)], ddof=1))
    votes = meta.vote_count(y, threshold=math.log(NULL_FRACTION / (1 - NULL_FRACTION)))
    baselines = {
        "vote_count": votes,
        "vote_count_all_six": (
            f"{len(SIX)} of {len(SIX)} papers report a material unresolved fraction; "
            f"{len(extracted)} of {len(SIX)} report one that can be pooled"
        ),
        "unweighted_mean": naive_mean,
        "unweighted_mean_sd": naive_sd,
        "unweighted_mean_ci": (
            naive_mean - 1.959963985 * naive_sd / math.sqrt(k),
            naive_mean + 1.959963985 * naive_sd / math.sqrt(k),
        ),
        "fixed_effect_p": meta.proportion_back(primary.fit.fixed.pooled, scale="logit"),
        "fixed_effect_ci_p": (
            meta.proportion_back(primary.fit.fixed.ci[0], scale="logit"),
            meta.proportion_back(primary.fit.fixed.ci[1], scale="logit"),
        ),
        "egger": meta.eggers_test(y, v),
    }

    # Power at the realised k, at both ends of what tau2 could be.
    delta = math.log(0.5 / 0.5) - math.log(NULL_FRACTION / (1 - NULL_FRACTION))
    tau2_hi = primary.fit.het.tau2_ci[1]
    power = {
        "delta_logit": delta,
        "at_tau2_zero": meta.power_for_pooled_effect(v, tau2=0.0, delta=delta),
        "at_tau2_hat": meta.power_for_pooled_effect(v, tau2=primary.fit.het.tau2, delta=delta),
        "at_tau2_upper": meta.power_for_pooled_effect(
            v, tau2=min(tau2_hi, 1e6) if math.isfinite(tau2_hi) else 10.0, delta=delta
        ),
        "q_test_at_typical_variance": meta.power_to_detect_heterogeneity(
            v, tau2=meta.typical_within_variance(v), n_sim=40000, seed=8
        ),
        "q_test_at_four_times_typical": meta.power_to_detect_heterogeneity(
            v, tau2=4 * meta.typical_within_variance(v), n_sim=40000, seed=8
        ),
    }

    # The pre-registered sensitivity set.
    runs: list[SensitivityRun] = []
    idx = {e.key: i for i, e in enumerate(extracted)}

    def swap(key: str, unresolved: int, total: int) -> tuple[list[int], list[int]]:
        c, t = list(counts), list(totals)
        c[idx[key]], t[idx[key]] = unresolved, total
        return c, t

    for name, desc, (c, t) in [
        (
            "kotawala-all-pairs",
            "Kotawala at 11 of 40, the all-pairs Open LLM Leaderboard v1 count instead of the "
            "adjacent-rank one",
            swap("kotawala", 11, 40),
        ),
        (
            "kotawala-clustered",
            "Kotawala at 6 of 9, the adjacent-rank count under subject-level clustering",
            swap("kotawala", 6, 9),
        ),
        (
            "dlugosz-low",
            "Dlugosz at 9 of 20, the low reading of 'only half'",
            swap("dlugosz", 9, 20),
        ),
        (
            "dlugosz-high",
            "Dlugosz at 11 of 20, the high reading of 'only half'",
            swap("dlugosz", 11, 20),
        ),
        (
            "dlugosz-after-confound",
            "Dlugosz at 15 of 20, counting the cases the integer-magnitude confound explains as "
            "also unresolved",
            swap("dlugosz", 15, 20),
        ),
    ]:
        runs.append(_sensitivity(name, desc, _fit(c, t, labels)))

    bounded_counts = counts + [6]
    bounded_totals = totals + [11]
    bounded_labels = list(labels) + ["Mandujano Reyes 2607.25257 (bound)"]
    runs.append(
        _sensitivity(
            "with-mandujano-bound",
            "Mandujano Reyes entered at 6 of 11, the weakest count consistent with 'most' of the "
            "adjacent-rank pairs among 12 models. Both integers are inferred and neither is quoted",
            _fit(bounded_counts, bounded_totals, bounded_labels),
        )
    )
    runs.append(
        _sensitivity(
            "double-arcsine",
            "Freeman-Tukey double-arcsine analysis scale instead of logit",
            _fit(counts, totals, labels, scale="double-arcsine"),
        )
    )
    # The uncorrected logit is only defined when no study sits on a boundary. The precondition is
    # checked rather than the failure caught: an infinite logit is a mathematical fact about the
    # input, not an anticipated measurement condition, so the run declares itself unavailable.
    boundary = [
        f"{lab} at {c} of {t}" for lab, c, t in zip(labels, counts, totals) if c == 0 or c == t
    ]
    if boundary:
        runs.append(
            _unavailable(
                "no-continuity-correction",
                "Logit with no continuity correction",
                f"undefined: {', '.join(boundary)} sits on a boundary and log(0) has no value. "
                f"The double-arcsine run covers the same question for boundary counts.",
            )
        )
    else:
        runs.append(
            _sensitivity(
                "no-continuity-correction",
                "Logit with no continuity correction",
                _fit(counts, totals, labels, correction=0.0),
            )
        )
    for method in ("DL", "REML"):
        runs.append(
            _sensitivity(
                f"tau2-{method}",
                f"{meta.TAU2_NAMES[method]} tau2 instead of Paule-Mandel",
                _fit(counts, totals, labels, tau2_method=method),
            )
        )
    runs.append(
        _sensitivity(
            "hartung-knapp",
            "Hartung-Knapp-Sidik-Jonkman confidence interval with a t(k-1) critical value",
            _fit(counts, totals, labels, knapp_hartung=True),
        )
    )
    runs.append(
        _sensitivity(
            "normal-prediction-rule",
            "The prediction interval metafor prints by default, with a z critical value instead of "
            "t(k-2)",
            _fit(counts, totals, labels, rule=meta.PredictionRule.NORMAL),
        )
    )

    flips = sum(1 for r in runs if r.crosses)
    metrics = {
        "k_extracted": float(k),
        "k_available": float(len(SIX)),
        "pooled_proportion": primary.pooled_p,
        "pooled_logit": primary.fit.pooled,
        "se_logit": primary.fit.se,
        "ci_low": primary.ci_p[0],
        "ci_high": primary.ci_p[1],
        "pi_low": primary.prediction_p[0],
        "pi_high": primary.prediction_p[1],
        "width_ratio_logit": primary.fit.width_ratio,
        "tau2_logit": primary.fit.het.tau2,
        "tau2_lower_limit_logit": primary.fit.het.tau2_ci[0],
        "tau2_upper_limit_logit": primary.fit.het.tau2_ci[1],
        "i2": primary.fit.het.i2,
        "q": primary.fit.het.q,
        "q_p": primary.fit.het.q_p,
        "sensitivity_sign_flips": float(flips),
        "sensitivity_runs": float(len(runs)),
    }

    spec = study_spec()
    outcomes = {}
    for h in spec.hypotheses:
        value = metrics.get(h.prediction.metric)
        outcomes[h.id] = (
            "void" if value is None else ("confirmed" if h.prediction.check(value) else "refuted")
        )
    kill_outcomes = {}
    for kc in spec.kill_criteria:
        value = metrics.get(kc.metric)
        kill_outcomes[kc.id] = (
            "void" if value is None else ("fired" if kc.fired(value) else "passed")
        )

    return X8Result(
        primary=primary,
        extracted=extracted,
        refused=refused,
        baselines=baselines,
        power=power,
        sensitivity=tuple(runs),
        metrics=metrics,
        outcomes=outcomes,
        kill_outcomes=kill_outcomes,
        quote_checks=tuple(checks),
        dossier_sha256=dossier_sha256(dossier),
        extras={
            "normal_rule_prediction_p": next(
                r.prediction_p for r in runs if r.name == "normal-prediction-rule"
            )
        },
    )


# ---------------------------------------------------------------------------
# The write-up
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def student_t_over_z(k: int, alpha: float = meta.ALPHA) -> float:
    """How much wider t(k-2) makes an interval than z does. 2.20 at k = 4, 1.12 at k = 13."""
    from scipy.stats import norm
    from scipy.stats import t as student_t

    return float(student_t.ppf(1 - alpha / 2, k - 2) / norm.ppf(1 - alpha / 2))


def findings_section(result: X8Result, outcome: FreezeOutcome | None = None) -> str:
    """The write-up section, generated from the numbers rather than written around them.

    Every sentence that states a direction is conditioned on the value, so the section reads
    correctly if the pooled fraction lands on either side of the threshold and if the prediction
    interval turns out narrow. Nothing in it needs editing when the numbers move.
    """
    p = result.primary
    m = result.metrics
    het = p.fit.het
    lo_ci, hi_ci = p.ci_p
    lo_pi, hi_pi = p.prediction_p
    ci_excl, pi_excl = p.fit.excludes(math.log(NULL_FRACTION / (1 - NULL_FRACTION)))
    tau2_lo, tau2_hi = het.tau2_ci
    tau2_hi_s = "unbounded" if not math.isfinite(tau2_hi) else f"{tau2_hi:.3f}"
    nr_lo, nr_hi = result.extras["normal_rule_prediction_p"]

    lines: list[str] = []
    add = lines.append

    add("## X8: what the 2026 leaderboard re-analyses support about the next leaderboard")
    add("")
    add(
        f"Six groups re-analysed public model leaderboards during 2026, in {N_TRADITIONS} different "
        f"statistical traditions across {N_BENCHMARKS} benchmarks, independently, none of them from "
        f"a frontier lab. Every one concluded that a material fraction of published orderings does "
        f"not survive its own inferential procedure. No leaderboard operator has been found to cite "
        f"any of them and no leaderboard has been found to have changed. This section pools them "
        f"and reports what they imply for a leaderboard nobody has checked yet."
    )
    add("")
    add(RESPONSE_CAVEAT)
    add("")

    add("### The number, and the interval that matters")
    add("")
    add(
        f"Across the {result.primary.fit.k} re-analyses whose findings carry both a numerator and a "
        f"denominator, the pooled fraction of published model comparisons that do not survive "
        f"proper inference is **{p.pooled_p:.2f}** ({_pct(p.pooled_p)}), with a 95% confidence "
        f"interval of [{lo_ci:.2f}, {hi_ci:.2f}]."
    )
    add("")
    add(
        f"That interval is about the *mean* across the studies pooled. The interval a reader "
        f"actually wants is about the *next* study, and it is the 95% prediction interval: "
        f"**[{lo_pi:.2f}, {hi_pi:.2f}]**. It is {p.fit.width_ratio:.2f} times wider on the analysis "
        f"scale. Go and re-analyse a leaderboard nobody in this set looked at, and the fraction of "
        f"its orderings that fails is, on this evidence, somewhere between {_pct(lo_pi)} and "
        f"{_pct(hi_pi)}."
    )
    add("")
    if ci_excl and not pi_excl:
        add(
            f"The confidence interval excludes {_pct(NULL_FRACTION)} and the prediction interval "
            f"does not. The average is established and the next leaderboard is not. A version of "
            f"this analysis that reported only the confidence interval would have licensed a claim "
            f"about the next leaderboard that the six studies do not support, which is the same "
            f"move the six studies were written to object to."
        )
    elif ci_excl and pi_excl:
        add(
            f"Both intervals exclude {_pct(NULL_FRACTION)}, so the claim survives being asked "
            f"about a leaderboard that has not been checked yet. That is a stronger result than "
            f"the six papers individually support and it is the one worth publishing."
        )
    else:
        add(
            f"The confidence interval contains {_pct(NULL_FRACTION)}, so the pooled fraction is "
            f"not distinguishable from it, and the prediction interval is necessarily wider still."
        )
    add("")
    if abs(nr_lo - lo_ci) < 5e-3 and abs(nr_hi - hi_ci) < 5e-3:
        add(
            f"The comparison worth making is against what a reader would get from the default "
            f"settings of the most widely used meta-analysis package, which substitutes a normal "
            f"critical value for t(k-2). Here the estimated between-study variance is zero, so "
            f"under a normal critical value the prediction interval becomes "
            f"[{nr_lo:.2f}, {nr_hi:.2f}], which is the confidence interval to two decimal places. "
            f"A reader following the defaults would print the same pair of numbers twice and end up "
            f"with no interval about a new leaderboard at all, and would have no signal that "
            f"anything had gone missing. The t(k-2) form is what stops that: it keeps the "
            f"uncertainty in having estimated tau2 as zero from four points, which is the "
            f"uncertainty that matters most when tau2 comes back as zero."
        )
    else:
        add(
            f"For comparison, the prediction interval a reader would get from the default settings "
            f"of the most widely used meta-analysis package is [{nr_lo:.2f}, {nr_hi:.2f}], because "
            f"it substitutes a normal critical value for t(k-2). At k = {p.fit.k} that is a factor "
            f"of {student_t_over_z(p.fit.k):.2f} in the width, and it errs in the direction of "
            f"confidence."
        )
    add("")

    add("### Say the sample size out loud")
    add("")
    add(
        f"Six studies is few, and only {result.primary.fit.k} of the six could be pooled. At k = "
        f"{result.primary.fit.k} the between-study variance has {het.q_df} degrees of freedom. The "
        f"point estimate is tau2 = {het.tau2:.3f} by {meta.TAU2_NAMES[het.tau2_method]} and its 95% "
        f"Q-profile interval is [{tau2_lo:.3f}, {tau2_hi_s}]. Cochran's Q is {het.q:.2f} on "
        f"{het.q_df} degrees of freedom (p = {het.q_p:.2f}) and I2 is {het.i2:.0f}%."
    )
    add("")
    add(
        f"None of that establishes that the six agree. Simulated at the within-study variances "
        f"these studies actually have, Cochran's Q detects a between-study variance equal to the "
        f"typical within-study variance only {_pct(result.power['q_test_at_typical_variance'])} of "
        f"the time, and a variance four times that only "
        f"{_pct(result.power['q_test_at_four_times_typical'])} of the time. A non-significant Q "
        f"here is a statement about the test."
    )
    add("")
    add(
        "This is the part of the analysis that is about itself. A meta-analysis of a handful of "
        "studies that came back with a narrow interval and a confident headline would be making, "
        "one level up, the error the six papers are collectively about: quoting a point estimate "
        "as though the sample behind it were larger than it is. The wide interval is the finding."
    )
    add("")

    add(f"### What could not be pooled: {len(result.refused)} of the six")
    add("")
    for e in result.refused:
        add(
            f"- **{e.authors}, arXiv {e.arxiv}** ({e.benchmark}). The recorded finding is "
            f'"{e.quote}" (line {e.line}). The quantifier fixes a direction and not a count, and no '
            f"denominator appears anywhere in the source, so no proportion can be formed without "
            f"inventing one. Excluded, and reported here rather than dropped."
        )
    add("")
    add(
        f"Be careful about what that does and does not say. It is a statement about the summary "
        f"these numbers were read from, not about the two papers: neither was fetched for this "
        f"analysis, and the counts may well be in their tables. What it does establish is that "
        f"{len(result.refused)} of the {len(SIX)} results in the only assembled account of this "
        f"literature reach a reader without a denominator, so a reader who wants to pool them has "
        f"to go and get four integers first. That is the next piece of work on this analysis, it is "
        f"an afternoon, and it takes the realised k from {result.primary.fit.k} to {len(SIX)}, "
        f"which on its own would cut the prediction interval materially through the degrees of "
        f"freedom alone: t(k-2) falls from {student_t_over_z(result.primary.fit.k) * 1.959963985:.2f} "
        f"to {student_t_over_z(len(SIX)) * 1.959963985:.2f}."
    )
    add("")

    add("### The studies, and where each number came from")
    add("")
    add("| # | Study | Benchmark | Tradition | Unresolved | Fraction | Dossier line |")
    add("|---|---|---|---|---|---|---|")
    for i, e in enumerate(SIX, start=1):
        frac = "not stated" if e.proportion is None else f"{e.proportion:.3f}"
        count = "no denominator" if e.unresolved is None else f"{e.unresolved} of {e.total}"
        add(
            f"| {i} | {e.authors} `{e.arxiv}` | {e.benchmark} | {e.tradition} | {count} | "
            f"{frac} | {e.line} |"
        )
    add("")
    add(
        f"Estimand: {ESTIMAND} Inclusion rule: {INCLUSION_RULE} Source: "
        f"`{DOSSIER_NAME}` section {DOSSIER_SECTION}, sha256 `{result.dossier_sha256[:16]}...`, "
        f"with every quoted string re-checked against the file at the line above before the "
        f"analysis ran."
    )
    add("")

    add("### Baselines")
    add("")
    vc = result.baselines["vote_count"]
    add(
        f"- **Vote count**, which is what the field currently has: {result.baselines['vote_count_all_six']}. "
        f"Of the poolable ones, {vc.positive} of {vc.k} sit above {_pct(NULL_FRACTION)}. A vote "
        f"count has no interval, is insensitive to how large the fractions were, and gives the same "
        f"answer whether every study was decisive or every study was a coin flip."
    )
    bm = result.baselines
    add(
        f"- **Unweighted mean of the observed fractions**: {bm['unweighted_mean']:.3f} "
        f"(sd {bm['unweighted_mean_sd']:.3f}, naive 95% interval "
        f"[{bm['unweighted_mean_ci'][0]:.2f}, {bm['unweighted_mean_ci'][1]:.2f}]). It ignores that "
        f"the denominators range from {min(e.total for e in result.extracted)} to "
        f"{max(e.total for e in result.extracted)}."
    )
    add(
        f"- **Fixed-effect model**, which assumes every study estimates the same fraction: "
        f"{bm['fixed_effect_p']:.3f}, 95% CI [{bm['fixed_effect_ci_p'][0]:.2f}, "
        f"{bm['fixed_effect_ci_p'][1]:.2f}]. Reported so the reader can see what the random-effects "
        f"model changed."
    )
    egger = result.baselines["egger"]
    add(
        f"- **Publication bias**: not assessed. Egger's test refuses below ten studies and there "
        f"are {result.primary.fit.k}. A funnel-asymmetry test at this k has almost no power, so a "
        f"null result from it would be a statement about the test rather than about the "
        f"literature. The refusal is the honest output: `{egger.reason.name}`."
    )
    add("")

    add("### Power at the realised sample size")
    add("")
    add(
        f"Against a null of {_pct(NULL_FRACTION)}, this collection of studies has power "
        f"{result.power['at_tau2_zero']:.2f} to detect a true fraction of 50% if there is no "
        f"between-study variance, {result.power['at_tau2_hat']:.2f} at the estimated tau2, and "
        f"{result.power['at_tau2_upper']:.2f} at the upper end of what tau2 could plausibly be. "
        f"The spread across that row is the honest summary of what k = {result.primary.fit.k} buys."
    )
    add("")

    add("### Sensitivity")
    add("")
    live = [r for r in result.sensitivity if r.available]
    dead = [r for r in result.sensitivity if not r.available]
    add(
        f"{len(result.sensitivity)} pre-registered variations, of which {len(live)} could be "
        f"computed on the data as extracted. The pooled fraction ranges from "
        f"{min(r.pooled_p for r in live):.2f} to {max(r.pooled_p for r in live):.2f} across them, "
        f"and {int(m['sensitivity_sign_flips'])} cross the {_pct(NULL_FRACTION)} threshold."
    )
    add("")
    add("| Variation | Pooled | 95% CI | 95% prediction interval | tau2 |")
    add("|---|---|---|---|---|")
    add(
        f"| *primary* | {p.pooled_p:.3f} | [{lo_ci:.2f}, {hi_ci:.2f}] | "
        f"[{lo_pi:.2f}, {hi_pi:.2f}] | {het.tau2:.3f} |"
    )
    for r in result.sensitivity:
        if r.available:
            add(
                f"| {r.name} | {r.pooled_p:.3f} | [{r.ci_p[0]:.2f}, {r.ci_p[1]:.2f}] | "
                f"[{r.prediction_p[0]:.2f}, {r.prediction_p[1]:.2f}] | {r.tau2:.3f} |"
            )
        else:
            add(f"| {r.name} | not available | {r.unavailable_reason} | | |")
    add("")
    if dead:
        add(
            f"{len(dead)} variation(s) are listed as not available rather than omitted, so the "
            f"count in the table matches the count in the frozen spec."
        )
        add("")

    add("### Registered predictions and their verdicts")
    add("")
    spec = study_spec()
    add("| id | prediction | value | verdict |")
    add("|---|---|---|---|")
    for h in spec.hypotheses:
        pr = h.prediction
        value = m.get(pr.metric)
        add(
            f"| {h.id} | `{pr.metric}` {pr.comparator} {pr.threshold} | "
            f"{'n/a' if value is None else f'{value:.4g}'} | **{result.outcomes[h.id]}** |"
        )
    for kc in spec.kill_criteria:
        value = m.get(kc.metric)
        add(
            f"| {kc.id} (kill) | `{kc.metric}` {kc.comparator} {kc.threshold} | "
            f"{'n/a' if value is None else f'{value:.4g}'} | **{result.kill_outcomes[kc.id]}** |"
        )
    add("")
    add(FREEZE_HONESTY)
    add("")
    if outcome is not None:
        add(
            f"Frozen as `{outcome.frozen.study_id}`, spec hash `{outcome.frozen.spec_hash}`, at "
            f"{outcome.frozen.frozen_at}, against git `{outcome.frozen.git_sha}`."
            + (
                " The git stamp is PROVISIONAL: the working tree was not clean at freeze time. The "
                "spec hash is exact and does not depend on the tree, so the stamp is reapplied at "
                "commit time by re-running the freeze and confirming the hash is unchanged."
                if outcome.provisional
                else ""
            )
        )
        add("")

    add("### What is not claimed")
    add("")
    add(
        "- The six studies do not share an estimand exactly. Four count rank-adjacent pairs or "
        "model comparisons on a leaderboard; one counts retrieval-method comparisons in a RAG "
        "stress test; one counts models rather than pairs. Pooling them assumes they are estimating "
        "the same underlying fraction, and that assumption is the analysis's largest soft spot. It "
        "is why the prediction interval, which allows for the studies genuinely differing, is the "
        "reported headline rather than the confidence interval."
    )
    add(
        "- The six were not sampled from anything. They are the ones a single literature scan "
        "found, and a re-analysis that found leaderboard orderings intact would be harder to "
        "publish than one that did not. With four poolable studies there is no way to test for that "
        "and no correction is applied; the direction of the bias, if it exists, is toward the "
        "fraction reported here being too high."
    )
    add(
        f"- I2 is reported at {het.i2:.0f}% and should not be the headline. It depends on the "
        f"precision of the included studies as well as on their disagreement, and in machine "
        f"learning precision is a budget decision rather than a constraint: the same studies run on "
        f"ten times the items would show a higher I2 with tau2 unchanged. tau2 and its interval are "
        f"the interpretable pair."
    )
    add(
        "- Nothing here says the six papers are wrong. They are almost certainly right about the "
        "leaderboards they examined. The claim is narrower and it is about what follows for a "
        "leaderboard they did not examine."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    dossier: Path | None = None, repo_dir: Path | None = None, frozen_at: str | None = None
) -> tuple[FreezeOutcome, X8Result]:
    """Verify, freeze, then analyse. The order is the point and it is not a parameter."""
    dossier = resolve_dossier(dossier)
    assert_quotes_verify(dossier)
    outcome = freeze_x8(repo_dir=repo_dir, frozen_at=frozen_at)
    result = analyse(dossier)
    return outcome, result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dossier", type=Path, default=DOSSIER)
    ap.add_argument("--json", type=Path, default=None, help="write the full result as JSON")
    ap.add_argument("--markdown", type=Path, default=None, help="write the write-up section")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outcome, result = run(args.dossier)
    section = findings_section(result, outcome)

    if not args.quiet:
        print(outcome.render())
        print()
        print(result.primary.render())
        print()
        print(section)

    if args.json:
        payload = {
            "frozen": {
                "study_id": str(outcome.frozen.study_id),
                "spec_hash": outcome.frozen.spec_hash,
                "frozen_at": outcome.frozen.frozen_at,
                "git_sha": outcome.frozen.git_sha,
                "provisional": outcome.provisional,
                "dirty_paths": list(outcome.dirty_paths),
            },
            "dossier_sha256": result.dossier_sha256,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extraction": [
                {
                    "key": e.key,
                    "arxiv": e.arxiv,
                    "authors": e.authors,
                    "line": e.line,
                    "quote": e.quote,
                    "unresolved": e.unresolved,
                    "total": e.total,
                    "status": e.status,
                }
                for e in SIX
            ],
            "primary": result.primary.as_dict(),
            "metrics": result.metrics,
            "outcomes": result.outcomes,
            "kill_outcomes": result.kill_outcomes,
            "power": result.power,
            "sensitivity": [
                {
                    "name": r.name,
                    "description": r.description,
                    "pooled_p": r.pooled_p,
                    "ci_p": list(r.ci_p),
                    "prediction_p": list(r.prediction_p),
                    "tau2": r.tau2,
                    "k": r.k,
                }
                for r in result.sensitivity
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.markdown:
        args.markdown.write_text(section + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "DOSSIER",
    "DOSSIER_ENV_VAR",
    "DOSSIER_NAME",
    "DOSSIER_SECTION",
    "DOSSIER_SHA256",
    "ESTIMAND",
    "INCLUSION_RULE",
    "NULL_FRACTION",
    "SIX",
    "DossierNotConfigured",
    "Extraction",
    "FreezeOutcome",
    "QuoteCheck",
    "SensitivityRun",
    "X8Result",
    "analyse",
    "assert_quotes_verify",
    "dossier_sha256",
    "findings_section",
    "freeze_x8",
    "main",
    "repo_root",
    "resolve_dossier",
    "run",
    "study_spec",
    "verify_quotes",
]
