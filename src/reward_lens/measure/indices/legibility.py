"""A15 Legibility frontier and tacit residual.

Formal definition, A15. Read interpretation as rate-distortion. Build a program
``r̂_K = Σ w_i π_i`` from predicates ``π_i`` of description length at most ``K``, fit to the model's
scores, and measure ``fidelity(K) =`` the ranking agreement of ``r̂_K`` with the true reward ``r``. The
frontier ``fidelity(K)`` rises with the description-length budget ``K``; its knee ``K*`` is the point of
diminishing returns, the legible complexity of the reward. The tacit residual ``ρ = r − r̂_{K*}`` is what
no short program captures, the reward's irreducibly illegible part.

Deviation from A15: fidelity is Spearman rank agreement (the ranking-agreement reading); predicates are
selected cheapest-first up to the budget, and the knee is the smallest budget reaching within a
tolerance of the maximum fidelity (a simple, deterministic knee rule stated here rather than left
implicit). The predicate library and its description lengths come from the concept layer's difference
dictionary in production; the synthetic test supplies a small predicate matrix with known costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    measured_without_input,
    missing_injection,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks), the ranking-agreement fidelity of A15."""
    from reward_lens.stats.effects import spearman_with_ci

    return float(spearman_with_ci(a, b, n_resamples=1, seed=0).point)


def legibility_frontier(
    predicates: np.ndarray,
    reward: np.ndarray,
    costs: Sequence[float],
    budgets: Sequence[float] | None = None,
    *,
    knee_tol: float = 0.02,
) -> dict[str, object]:
    """The legibility frontier, its knee ``K*``, and the tacit residual (A15).

    ``predicates`` is ``(n, P)`` predicate activations, ``costs`` their description lengths, ``reward``
    the ``(n,)`` scores. For each budget ``K`` the cheapest predicates with cumulative cost ``≤ K`` are
    selected, a linear ``r̂_K`` is fit by least squares, and ``fidelity(K) = Spearman(r̂_K, r)``. The knee
    ``K*`` is the smallest budget whose fidelity is within ``knee_tol`` of the maximum, and the tacit
    residual is ``r − r̂_{K*}``. Returns the budgets, the fidelity curve, ``K*``, the fidelity there, and
    the tacit residual vector and its variance fraction.
    """
    pred = np.asarray(predicates, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64).ravel()
    costs = np.asarray(costs, dtype=np.float64).ravel()
    order = np.argsort(costs, kind="mergesort")
    cum = np.cumsum(costs[order])
    if budgets is None:
        budgets = sorted(set(float(c) for c in cum))
    budgets = list(budgets)

    fidelity: list[float] = []
    fits: dict[float, np.ndarray] = {}
    for k in budgets:
        chosen = order[cum <= k + 1e-12]
        if chosen.size == 0:
            fidelity.append(float("nan"))
            continue
        design = np.column_stack([np.ones(pred.shape[0]), pred[:, chosen]])
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        r_hat = design @ coef
        fits[k] = r_hat
        fidelity.append(_spearman(r_hat, r))

    fidelity_arr = np.asarray(fidelity, dtype=np.float64)
    finite = fidelity_arr[np.isfinite(fidelity_arr)]
    max_fid = float(finite.max()) if finite.size else float("nan")
    k_star = float(budgets[-1]) if budgets else float("nan")
    for k, f in zip(budgets, fidelity_arr):
        if np.isfinite(f) and f >= max_fid - knee_tol:
            k_star = float(k)
            break
    r_hat_star = fits.get(k_star, r)
    tacit = r - r_hat_star
    var_r = float(np.var(r, ddof=0))
    tacit_frac = float(np.var(tacit, ddof=0) / var_r) if var_r > 0 else float("nan")
    return {
        "budgets": [float(b) for b in budgets],
        "fidelity": fidelity_arr.tolist(),
        "k_star": k_star,
        "fidelity_at_knee": float(dict(zip(budgets, fidelity_arr)).get(k_star, float("nan"))),
        "max_fidelity": max_fid,
        "tacit_residual": tacit,
        "tacit_variance_fraction": tacit_frac,
    }


class Legibility(BaseObservable):
    """A15 legibility frontier ``fidelity(K)``, its knee ``K*``, and the tacit residual.

    Requires scores. The predicate library and description-length costs are injected (the concept layer's
    difference dictionary supplies them in production). Reports the fidelity curve over the budget, the
    knee, and the illegible tacit residual. Gauge is INVARIANT: fidelity is a rank agreement, and the
    tacit residual's variance fraction is scale-free.

    What it cannot do. The frontier is only as wide as the predicate library it is handed, so a
    low fidelity at every budget means the library missed the reward's structure and not that the
    reward is illegible; those two readings are indistinguishable in this payload. Predicates are
    taken cheapest-first rather than searched over, so the curve is one greedy path through the
    space and the knee is the knee of that path. The tacit residual is what this predicate set
    failed to capture, which is a weaker claim than what no short program can capture.
    """

    name = "Legibility"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A15"
    deviations = (
        "fidelity is Spearman rank agreement; predicates are selected cheapest-first and the knee is "
        "the smallest budget within tolerance of the maximum fidelity",
        "the predicate library and its description lengths are the production path (concept diff "
        "dictionary)",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.legibility_frontier"
    #: The grader is scored on the view; the predicate library and its description lengths come
    #: from the concept layer's difference dictionary.
    requires: AccessMatrix = {
        Component.GRADER: Access.QUERY,
        Component.RECORD: Access.RECORD,
    }
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset(
            {RegimeCondition.STATIONARY_GRADER, RegimeCondition.GROUP_NONDEGENERATE}
        ),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Fidelity is Spearman rank agreement, which any increasing transformation of the reward leaves
    #: alone, and an affine rescale with positive scale is one.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.length", "baseline.single_cheapest_predicate")
    rung = 0

    def __init__(
        self,
        predicates: np.ndarray | None = None,
        costs: Sequence[float] | None = None,
        *,
        budgets: Sequence[float] | None = None,
    ) -> None:
        self.predicates = predicates
        self.costs = costs
        self.budgets = budgets

    def preflight(self, ctx: Context) -> PreflightResult:
        """The predicate library and its costs, or a refusal. A budget sweep with no costs is an arbitrary subset.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.predicates is None or self.costs is None:
            return missing_injection(
                self,
                needs={
                    "predicates": "an (n, P) matrix of predicate activations",
                    "costs": "a length-P vector of description lengths, one per predicate",
                },
                have="neither was injected",
                remedy=(
                    "Construct `Legibility(predicates=..., costs=...)`. `predicates` holds one column per "
                    "predicate in the library evaluated on each sample; `costs` holds that predicate's "
                    "description length in the same column order. The frontier sweeps a cost budget, and the "
                    "costs are what make the sweep an ordering rather than an arbitrary subset."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.predicates is None or self.costs is None:
            raise measured_without_input(self)
        reward = reward_scores(ctx.signal, ctx.view, ctx.readout)
        report = legibility_frontier(self.predicates, reward, self.costs, self.budgets)
        payload = {
            "budgets": report["budgets"],
            "fidelity": report["fidelity"],
            "k_star": report["k_star"],
            "fidelity_at_knee": report["fidelity_at_knee"],
            "max_fidelity": report["max_fidelity"],
            "tacit_variance_fraction": report["tacit_variance_fraction"],
            "n_items": int(reward.size),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=int(reward.size), method="none"))


__all__ = ["legibility_frontier", "Legibility"]
