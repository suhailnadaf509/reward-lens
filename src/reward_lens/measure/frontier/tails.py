"""N3, the reward tail index: the precondition the whole tilt layer rests on, measured.

``K(lambda) = log E_0[e^{lambda r}]`` is finite only if the reward's moment generating function
exists, and for a reward whose upper tail is regularly varying with index ``gamma > 0`` it does not,
at any positive lambda. So `LIGHT_TAILED` is not a technicality attached to the frontier: it is the
condition under which the frontier's x-axis is a number rather than a divergent integral.

The estimator here is Hill, with a stability protocol, and the sample size it needs is specific: a
defensible estimate needs roughly 30,000 prompts at ``q = 0.95`` to give about 1,570 exceedances,
and below that this instrument refuses rather than reporting a number from 200 prompts.
The refusal is not an envelope violation. The tail index stays perfectly well defined on a small
bank; what is unavailable is the estimate, so the refusal names the n it needs and carries the
wide-interval plot as a bound.

**Two estimators, and the reason there are two.** The Hill estimator

    gamma_hat(k) = (1/k) sum_{i=1}^{k} [ log X_(i) - log X_(k+1) ]

over the k largest order statistics is the standard choice, and it is what the one real
measurement in the literature is: a Hill estimate around 0.20 on an open reward model, described as
consistent with light-tailed error. Reporting it is what makes this instrument comparable to that
measurement. It has two properties worth stating rather than working around.

It is invariant under ``r -> a r`` and **not** under ``r -> r + b``, because a location shift does
not survive a log. The quantity is location invariant, the estimator is not, and that gap is a
property of the estimator rather than something to be fixed by shifting the data quietly. So the
Pickands estimator

    gamma_hat_P(k) = (1 / log 2) log [ (X_(k) - X_(2k)) / (X_(2k) - X_(4k)) ]

is computed at the same thresholds. It is exactly location and scale invariant by construction,
which is what lets this instrument carry the ``reward.affine`` group the registry declares for it,
and it pays for that with a good deal more variance. Both are reported.

The second property is the decisive one for the verdict. **Hill's support is ``gamma > 0``.** A
light-tailed reward has ``gamma <= 0``: zero in the Gumbel domain, negative for a bounded reward.
Hill can therefore never return evidence of light-tailedness. It returns a small positive number
and the reader calls it "consistent with light-tailed", which is what the cited 0.20 is. Pickands
covers ``gamma`` on the whole real line, so it is the estimator the `LIGHT_TAILED` verdict is taken
from, and a bounded reward gets a negative estimate rather than a small positive one.

Kill condition, from the catalogue record: if every grader tested is unambiguously light-tailed, the
instrument becomes a precondition check rather than a reported quantity, and it renders as an
envelope line on the card instead of a row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import ConditionReading, EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.provenance import Provenance
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    SubjectRef,
)
from reward_lens.measure.frontier._base import FrontierInstrument
from reward_lens.measure.frontier.horizon import ALL_SUBSTRATES

#: The exceedance count a defensible tail estimate needs. The requirement is stated as "roughly
#: 30,000 prompts at q = 0.95 to get 1,570 exceedances", and 30,000 at 0.95 is 1,500 rather than
#: 1,570, so the two halves of that sentence differ by 5%. The exceedance count is
#: the binding requirement and the prompt count is derived from it, because the count is what the
#: estimator's variance depends on and the prompt count depends on the quantile chosen.
MIN_EXCEEDANCES = 1570

#: The quantile the exceedance requirement is stated at.
DEFAULT_TAIL_QUANTILE = 0.95

#: Above this, the light-tailed assumption is reported as failing. It is a **default rather than a
#: measurement**, and it is the one number in this package that should be argued about before it is
#: relied on. Two facts bracket it. Strictly, the moment generating function exists only for
#: gamma <= 0, so any positive index at all breaks the tilt layer. Practically, the one cited
#: measurement is a Hill estimate of about 0.20 called consistent with light-tailed error, so a
#: bound that rejects 0.20 would contradict the measurement this layer relies on. 0.25 admits
#: that measurement and rejects the finite-variance boundary at gamma = 0.5.
DEFAULT_GAMMA_MAX = 0.25

#: How many consecutive k the stability protocol looks over when hunting a plateau.
DEFAULT_PLATEAU_WINDOW = 15

#: The catalogue's baseline for N3: a light-tailed parametric fit at the same threshold, for
#: contrast. An exponential upper tail has gamma = 0 exactly, so the comparator is the mean excess
#: over the threshold, which is the exponential tail's own scale parameter and is what
#: `loops.tilt.critical_lambda_from_tail` inverts to get lambda_c.
TAIL_BASELINES: tuple[BaselineID, ...] = ("baseline.exponential_tail_scale",)

TAIL_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.POLICY: Access.QUERY,
}

TAIL_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the Hill and Pickands estimators on the upper order statistics depend on no regime of any "
        "run: they are functionals of one sample's own tail and nothing about how the sample was "
        "produced can make them wrong. What they depend on is exceedance count, and an inadequate "
        "one is a refusal naming the n it needs rather than a violated envelope, because the "
        "quantity stays well defined and only the estimate is unavailable. Encoding a sample-size "
        "problem as a regime condition would have made it look like a property of the run."
    ),
)


def hill(sorted_desc: np.ndarray, k: int) -> float:
    """`gamma_hat(k)` from the k largest order statistics. Scale invariant, not location invariant.

    Returns nan when the threshold order statistic is not positive, which is a real and common case
    on reward models whose scores are logits: the log in the estimator has no value there. Shifting
    the sample to make it positive would produce a number, and the number would depend on the shift,
    which is the whole reason this returns nan instead.
    """
    if k < 2 or k + 1 > sorted_desc.size:
        return float("nan")
    threshold = sorted_desc[k]
    if not threshold > 0.0:
        return float("nan")
    top = sorted_desc[:k]
    return float(np.mean(np.log(top) - np.log(threshold)))


def pickands(sorted_desc: np.ndarray, k: int) -> float:
    """`gamma_hat_P(k) = log[(X_(k) - X_(2k)) / (X_(2k) - X_(4k))] / log 2`.

    Exactly invariant under ``r -> a r + b`` for ``a > 0``, because both the numerator and the
    denominator are differences of order statistics, so the scale cancels in the ratio and the
    location cancels in each difference. Defined on the whole real line, so a bounded reward gets a
    negative index rather than the small positive one Hill is forced to return.

    Returns nan when ``4k`` exceeds the sample or the denominator is not positive, the latter being
    a tie in the middle order statistics rather than a failure of the estimator.
    """
    if k < 1 or 4 * k > sorted_desc.size:
        return float("nan")
    a = sorted_desc[k - 1] - sorted_desc[2 * k - 1]
    b = sorted_desc[2 * k - 1] - sorted_desc[4 * k - 1]
    if not (b > 0.0 and a > 0.0):
        return float("nan")
    return float(np.log(a / b) / np.log(2.0))


@dataclass(frozen=True)
class Plateau:
    """The stability protocol's answer: a k range, the estimate on it, and whether it is stable."""

    found: bool
    k_lo: int
    k_hi: int
    gamma: float
    spread: float
    detail: str


def find_plateau(
    ks: np.ndarray,
    gammas: np.ndarray,
    *,
    window: int = DEFAULT_PLATEAU_WINDOW,
    relative_spread: float = 0.5,
) -> Plateau:
    """The stability protocol, as a plateau hunt over the estimate's own plot.

    The Hill plot is threshold sensitive by construction: at small k it is upward biased and noisy,
    at large k it is contaminated by the body of the distribution. Reading one k is a choice
    disguised as a measurement, so the protocol is to look for a stretch of k over which the
    estimate does not move, report the estimate there, and say how wide the stretch was.

    Implemented as the window of ``window`` consecutive k with the smallest standard deviation of
    ``gamma_hat`` inside it. The estimate is the median over that window, which is what makes it
    robust to a single wild k rather than to the shape of the plot. A window whose spread exceeds
    ``relative_spread`` times the magnitude of the estimate is reported as no plateau found, and
    then the plot is the reading and a point estimate is not available.
    """
    ok = np.isfinite(gammas)
    if ok.sum() < window:
        return Plateau(
            found=False,
            k_lo=0,
            k_hi=0,
            gamma=float("nan"),
            spread=float("nan"),
            detail=(
                f"only {int(ok.sum())} of {gammas.size} thresholds gave a finite estimate, which is "
                f"fewer than the {window} a plateau needs"
            ),
        )
    ks_ok = ks[ok]
    g_ok = gammas[ok]
    best_i, best_sd = 0, float("inf")
    for i in range(g_ok.size - window + 1):
        sd = float(np.std(g_ok[i : i + window]))
        if sd < best_sd:
            best_i, best_sd = i, sd
    seg = g_ok[best_i : best_i + window]
    gamma = float(np.median(seg))
    scale = max(abs(gamma), 1e-12)
    if best_sd > relative_spread * scale:
        return Plateau(
            found=False,
            k_lo=int(ks_ok[best_i]),
            k_hi=int(ks_ok[best_i + window - 1]),
            gamma=gamma,
            spread=best_sd,
            detail=(
                f"the flattest stretch of {window} thresholds still moves by {best_sd:.3g} around "
                f"{gamma:.3g}, which is more than {relative_spread:.0%} of the estimate. There is "
                f"no plateau, so the plot is the reading and a single k is not"
            ),
        )
    return Plateau(
        found=True,
        k_lo=int(ks_ok[best_i]),
        k_hi=int(ks_ok[best_i + window - 1]),
        gamma=gamma,
        spread=best_sd,
        detail=(
            f"flat to {best_sd:.3g} over k in [{int(ks_ok[best_i])}, "
            f"{int(ks_ok[best_i + window - 1])}]"
        ),
    )


@register_payload
@dataclass
class TailReading:
    """The two plots, the plateau on each, and the light-tailed verdict.

    ``invariant_gamma`` is the Pickands estimate on its plateau and is the number the verdict is
    taken from, because it is the one defined on the whole real line and the one that survives the
    `reward.affine` group. ``hill_gamma`` is the Hill estimate on its own plateau and is what makes
    the reading comparable to the published measurement.
    """

    n: int
    quantile: float
    exceedances: int
    ks: np.ndarray
    hill_plot: np.ndarray
    pickands_plot: np.ndarray
    hill_se: np.ndarray
    hill_gamma: float
    hill_plateau: str
    invariant_gamma: float
    invariant_ci: tuple[float, float]
    invariant_plateau: str
    plateau_found: bool
    gamma_max: float
    light_tailed: bool | None
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says

    def condition_reading(self) -> ConditionReading:
        """The `LIGHT_TAILED` verdict, in the form the validity envelope consults.

        Three states and the third is real: `None` means the interval straddles the bound, so
        somebody looked and could not tell. That is different from a pass and the envelope treats
        it as different.
        """
        return ConditionReading(
            condition=RegimeCondition.LIGHT_TAILED,
            holds=self.light_tailed,
            statistic=self.invariant_gamma,
            threshold=self.gamma_max,
            detail=self.says,
        )


def _verdict(lo: float, hi: float, gamma_max: float) -> bool | None:
    """Pass when the whole interval is below the bound, fail when it is entirely above, else None."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi <= gamma_max:
        return True
    if lo > gamma_max:
        return False
    return None


def _bootstrap_pickands(
    x: np.ndarray, k_frac: float, resamples: int, seed: int
) -> tuple[float, float]:
    """A percentile interval for the Pickands estimate at a fixed fraction of the sample.

    The k is held as a fraction rather than as a count so that each resample estimates at the same
    place in its own tail. Resampling the whole sample rather than the exceedances is deliberate:
    which observations land in the tail is part of what is uncertain, and conditioning on the
    realised tail would report an interval narrower than the estimator deserves.
    """
    if resamples <= 0:
        return float("nan"), float("nan")
    n = x.size
    rng = np.random.default_rng(seed)
    out = np.empty(resamples, dtype=np.float64)
    k = max(1, int(round(k_frac * n)))
    for b in range(resamples):
        s = np.sort(x[rng.integers(0, n, n)])[::-1]
        out[b] = pickands(s, k)
    out = out[np.isfinite(out)]
    if out.size < max(10, resamples // 10):
        return float("nan"), float("nan")
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def measure_tail_index(
    reward: Sequence[float] | np.ndarray,
    *,
    quantile: float = DEFAULT_TAIL_QUANTILE,
    min_exceedances: int = MIN_EXCEEDANCES,
    gamma_max: float = DEFAULT_GAMMA_MAX,
    window: int = DEFAULT_PLATEAU_WINDOW,
    resamples: int = 300,
    seed: int = 0,
    instrument: str = "RewardTailIndex",
) -> TailReading | Refusal:
    """The Hill and Pickands plots, the plateau on each, and the light-tailed verdict.

    Refuses with `ABOVE_LOD_BELOW_LOQ` when the bank gives fewer than ``min_exceedances`` points
    above the stated quantile: the tail is there and can be seen, and the sample cannot quantify its
    index. The refusal carries the plot it could compute as a bound, because a wide-interval plot is
    a more useful thing to hold than nothing at all, and it names the n that would close the gap.
    """
    x = np.asarray(reward, dtype=np.float64).ravel()
    if x.size < 8:
        raise ValueError(f"a tail estimate needs a bank; got {x.size} scores")
    if not np.all(np.isfinite(x)):
        raise ValueError("the reward contains non-finite values, which is a scoring bug upstream")

    n = int(x.size)
    threshold = float(np.quantile(x, quantile))
    exceedances = int(np.count_nonzero(x > threshold))
    sorted_desc = np.sort(x)[::-1]

    k_hi = max(4, min(n // 4, max(exceedances, 4)))
    ks = np.arange(2, k_hi + 1, dtype=int)
    hill_plot = np.array([hill(sorted_desc, int(k)) for k in ks])
    pick_plot = np.array([pickands(sorted_desc, int(k)) for k in ks])
    with np.errstate(invalid="ignore"):
        hill_se = hill_plot / np.sqrt(ks.astype(np.float64))

    hill_pl = find_plateau(ks, hill_plot, window=min(window, max(2, ks.size // 3)))
    pick_pl = find_plateau(ks, pick_plot, window=min(window, max(2, ks.size // 3)))

    k_star = 0.5 * (pick_pl.k_lo + pick_pl.k_hi) if pick_pl.k_hi else max(2, exceedances)
    lo, hi = _bootstrap_pickands(x, max(k_star, 1.0) / n, resamples, seed)
    verdict = _verdict(lo, hi, gamma_max) if pick_pl.found else None

    excess = x[x > threshold] - threshold
    baselines = {
        "baseline.exponential_tail_scale": float(excess.mean()) if excess.size else float("nan"),
    }

    reading = TailReading(
        n=n,
        quantile=float(quantile),
        exceedances=exceedances,
        ks=ks,
        hill_plot=hill_plot,
        pickands_plot=pick_plot,
        hill_se=hill_se,
        hill_gamma=hill_pl.gamma,
        hill_plateau=hill_pl.detail,
        invariant_gamma=pick_pl.gamma,
        invariant_ci=(lo, hi),
        invariant_plateau=pick_pl.detail,
        plateau_found=bool(pick_pl.found),
        gamma_max=float(gamma_max),
        light_tailed=verdict,
        baselines=baselines,
    )
    state = {True: "holds", False: "fails", None: "cannot be decided"}[verdict]
    reading.says = (
        f"Pickands gamma-hat = {pick_pl.gamma:.3g} [{lo:.3g}, {hi:.3g}] over k in "
        f"[{pick_pl.k_lo}, {pick_pl.k_hi}]; Hill gamma-hat = {hill_pl.gamma:.3g}. The light-tailed "
        f"assumption {state} at a bound of {gamma_max:.3g}."
    )

    if exceedances < min_exceedances:
        needed = int(np.ceil(min_exceedances / max(1e-12, 1.0 - quantile)))
        return bounded_refusal(
            instrument,
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"{exceedances} exceedances above the q = {quantile:.3g} quantile on n = {n} "
                f"rollouts, against the {min_exceedances} a defensible index needs. The tail is "
                f"visible and its index is not quantifiable from this bank: at this k the Hill "
                f"estimate's own asymptotic standard error is gamma/sqrt(k), which is "
                f"{100.0 / np.sqrt(max(exceedances, 1)):.0f}% of the estimate before any "
                f"threshold-choice uncertainty is counted"
            ),
            remedy=(
                f"score {needed:,} rollouts to reach {min_exceedances} exceedances at "
                f"q = {quantile:.3g}, or lower the quantile and accept that the estimate is then "
                f"about the shoulder rather than the tail. The bound attached to this refusal is "
                f"the plot as far as this bank gets, which is worth reading and is not worth "
                f"quoting as a number."
            ),
            bound=make_evidence(
                observable="frontier.tail_index",
                observable_version="1.0",
                subject=SubjectRef(readout="tail"),
                value=reading,
                uncertainty=Uncertainty(n=n, method="percentile-bootstrap"),
                provenance=Provenance(),
            ),
            exceedances=exceedances,
            min_exceedances=min_exceedances,
            quantile=float(quantile),
            n=n,
            n_required=needed,
        )

    return reading


class RewardTailIndex(FrontierInstrument):
    """N3. The upper-tail index of the reward, as a plot with a stability protocol.

    Not to be confused with `measure.indices.tail.TailIndex`, which is A4: a point estimate of the
    right-tail exponent, aggregate and per-feature, reported as `grader.tail_index`. This one is the
    threshold-swept Hill and Pickands pair with plateau selection, reported as `frontier.tail_index`,
    and it exists to decide the `LIGHT_TAILED` precondition the tilt layer rests on.

    Kill condition: if every grader tested is unambiguously light-tailed, the instrument becomes a
    precondition check rather than a reported quantity, and it renders as an envelope line on the
    card instead of a row.
    """

    name = "RewardTailIndex"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N3"
    deviations = (
        "the `reward.affine` group is carried by the Pickands estimate rather than by the Hill "
        "estimate. Hill is invariant under a rescaling and not under a shift, because a location "
        "shift does not survive a log, and the quantity is location invariant even though that "
        "estimator is not",
        "the light-tailed verdict is taken from Pickands rather than from Hill, because Hill's "
        "support is gamma > 0 and it cannot represent the light-tailed case it is being asked "
        "about",
        "the exceedance requirement is stated as a count rather than as a prompt count. Section "
        "3.0.1 gives both and they differ by 5% at q = 0.95; the count is what the variance "
        "depends on",
    )

    quantity = "frontier.tail_index"
    requires: dict[Component, Access] = TAIL_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN})
    envelope = TAIL_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = TAIL_BASELINES
    rung = 0

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray | None = None,
        *,
        quantile: float = DEFAULT_TAIL_QUANTILE,
        min_exceedances: int = MIN_EXCEEDANCES,
        gamma_max: float = DEFAULT_GAMMA_MAX,
        window: int = DEFAULT_PLATEAU_WINDOW,
        resamples: int = 300,
        seed: int = 0,
    ) -> None:
        self.reward = reward
        self.quantile = float(quantile)
        self.min_exceedances = int(min_exceedances)
        self.gamma_max = float(gamma_max)
        self.window = int(window)
        self.resamples = int(resamples)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.reward is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no proxy scores were supplied, so there is no tail to estimate",
                remedy="pass `reward=` the grader's score on each of n base-policy rollouts.",
            )
        return measure_tail_index(
            self.reward,
            quantile=self.quantile,
            min_exceedances=self.min_exceedances,
            gamma_max=self.gamma_max,
            window=self.window,
            resamples=self.resamples,
            seed=self.seed,
            instrument=self.name,
        )


__all__ = [
    "DEFAULT_GAMMA_MAX",
    "DEFAULT_PLATEAU_WINDOW",
    "DEFAULT_TAIL_QUANTILE",
    "MIN_EXCEEDANCES",
    "TAIL_ACCESS",
    "TAIL_BASELINES",
    "TAIL_ENVELOPE",
    "Plateau",
    "RewardTailIndex",
    "TailReading",
    "find_plateau",
    "hill",
    "measure_tail_index",
    "pickands",
]
