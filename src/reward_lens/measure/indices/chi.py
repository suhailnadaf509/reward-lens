"""A12 χ and C1's susceptibility triple: ``S``, ``β = C⁻¹S`` and ``Gβ``, always together.

The first half of this module is the shipped index ``χ_i = Cov_0(f_i, r)``, unchanged. The second
half is the correction it needs: a marginal covariance read as an influence is a selection
*differential*, and the direct push is the coefficient vector of the multiple regression of fitness
on all traits jointly. Both ship, side by side, because the policy is to report `S` as primary and
`β` as the direct-effect estimate with its suppressor caveat attached, and because the two
literatures that meet here reached that same conclusion forty years and two fields apart.

Formal definition, A12. For a feature bank ``{f_i}`` evaluated on base-policy samples,
``χ_i = Cov_0(f_i, r)`` is the covariance of feature ``i`` with the reward under the base policy
``π_0``. This is the fluctuation-dissipation identity for the exponential tilt family
``π_λ ∝ π_0 exp(λ r)``: to first order, ``d E_λ[f_i]/dλ |_{λ=0} = Cov_0(f_i, r)``, so ``χ_i`` is the
predicted initial drift of feature ``i`` once the policy starts optimizing against the reward. A
feature with ``χ_i > 0`` will be pushed up early in optimization.

The predicted hack modes are the features the reward rewards but the gold objective does not:
``χ_i > 0`` with ``Cov_0(f_i, gold) ≤ 0``. Those are the directions optimization will inflate while
the true objective is flat or falling, which is the operational definition of a reward hack this
index forecasts before any optimization is run.

Deviations from A12: the second-order term ``d²E_λ[f]/dλ² = κ_3(f, r, r)`` that A12 also names is not
computed here; χ is the first-order (zeroth-order-in-λ covariance) susceptibility, which is the drift
predictor the cards use. The features come from a ``FeatureBank`` (the concept layer's contract); with
the synthetic ``LinearFeatureBank`` the planted ``Cov(feature, reward)`` is recovered exactly.

**What this index is an expansion about, which is not where anyone trains.** ``χ = Cov_0(f, r)`` is
evaluated on a fixed bank of base-policy samples at ``λ = 0``, so all three of its arguments are the
wrong ones for a live run. It multiplies the feature by ``r``, the raw reward, where the quantity that
actually multiplies the gradient is the advantage ``A``. It averages over a fixed base-policy bank,
where the sampling distribution that matters is the policy's own group at step ``t``. And it is
expanded about zero optimisation pressure, where a run in progress sits at ``θ_t``. It is a first-order
expansion about a regime nobody trains in, and that is a property of the index rather than of the
implementation: nothing about the numbers below is wrong, they are answers to a question one step
removed from the one a practitioner is asking. The ledger form ``Δz = η·Cov_group(A, f) + ρ``, which
re-expands about the current step, is a separate instrument.

**And ``χ`` is a selection differential, not a selection gradient.** In the language this mathematics
comes from, ``S = Cov(f, A)`` is the *marginal* association between a trait and fitness, and it
conflates direct with indirect selection. The direct push is the coefficient vector of the multiple
regression of fitness on all traits jointly, ``β = C⁻¹S`` with ``C = Cov(f, f)``, and **``S`` and ``β``
can differ in sign**. Reporting a marginal covariance and reading it as an influence is the single
most-warned-against error in that field, and this index does exactly that. A feature at the top of the
spectrum below may be there because it rides on another feature, and ranking on ``χ`` alone cannot
tell you. The registry records this as a bias statement rather than a footnote, so anything consuming
the reading carries it too.

**Which covariance operator ``C`` is, and why the answer changes the reading.** `C` is "the observed
feature covariance under the policy's own sampling distribution", which on any group-relative record
admits two readings that do not agree. *Pooled* is the covariance over all rollouts; *within-group*
centres each feature inside its own prompt group first, and between them sits prompt-to-prompt
heterogeneity, which is a property of the task distribution rather than of the policy. The ledger
settles it: its sampling distribution is "the policy's own group at step `t`" and it writes the
differential as ``Cov_group(A, f_i)``, the same object. So `C` is within-group, and a `β` from a
within-group `S` against a pooled `C` is not solving `S = Cβ` at all. Every reading here carries the
operator it used, because a `β` whose operator is unstated is not reproducible.

**And ``β`` is reported twice, in raw units and variance-standardised, because only one of them can
be ranked.** `β` in the features' own units is what ``Δz = ηGβ`` consumes. It is also not
comparable across features: on the campaign bank `len_chars` is counted in characters and
`hedging_rate` is a proportion, so the ordering of raw `|β|` is an ordering of units. The
variance-standardised form ``β̃_i = sd(f_i)·β_i`` is the response in
standard deviations of the feature per unit of reward, it is what Lande and Arnold's own worked
examples use, and it is the one a ranking is made on. Both ship on every reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import COVARIANT_LINEAR, INVARIANT
from reward_lens.core.quantity import BiasStatement
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    FeatureBank,
    final_activations,
    load_default_bank,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def susceptibility(features: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """The susceptibility spectrum ``χ_i = Cov_0(f_i, r)`` (A12).

    ``features`` is ``(n, k)`` and ``reward`` is ``(n,)``; returns the ``(k,)`` vector of covariances
    between each feature and the reward, using the population (biased) covariance so a planted
    ``Cov(f_i, r)`` is recovered on the nose as ``n`` grows. Centering both sides makes χ invariant to
    the arbitrary additive origins of the features and the reward.
    """
    f = np.asarray(features, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64).ravel()
    if f.ndim == 1:
        f = f[:, None]
    fc = f - f.mean(axis=0, keepdims=True)
    rc = r - r.mean()
    return (fc * rc[:, None]).mean(axis=0)


def predicted_hack_modes(
    chi: np.ndarray,
    chi_gold: np.ndarray,
    *,
    tol: float = 0.0,
) -> np.ndarray:
    """Boolean mask of predicted hack features: ``χ_i > tol`` and ``Cov_0(f_i, gold) ≤ tol`` (A12).

    A hack mode is a feature the reward pulls up (``χ_i`` positive) while the gold objective does not
    reward it (``χ_gold_i`` non-positive). ``tol`` sets a dead band around zero so noise near the
    boundary does not flip the flag; the default ``0.0`` is the literal definition.
    """
    chi = np.asarray(chi, dtype=np.float64).ravel()
    chi_gold = np.asarray(chi_gold, dtype=np.float64).ravel()
    return (chi > tol) & (chi_gold <= tol)


def _chi_shuffle_null(features: np.ndarray, reward: np.ndarray, seed: int, n: int) -> dict:
    """Per-feature label-permutation null for χ (the noise floor for "feature carries reward").

    Uses ``stats.nulls.shuffle_null`` per feature, permuting the reward against the fixed feature
    column so the null spectrum is the χ a decoupled feature/reward pairing would show. Returns the
    per-feature p-values and the shared null mean magnitude.
    """
    from reward_lens.stats.nulls import shuffle_null

    f = np.asarray(features, dtype=np.float64)
    r = np.asarray(reward, dtype=np.float64).ravel()
    if f.ndim == 1:
        f = f[:, None]

    def _cov(vals: np.ndarray, labels: np.ndarray) -> float:
        v = vals - vals.mean()
        lab = labels - labels.mean()
        return float((v * lab).mean())

    p_values: list[float] = []
    null_means: list[float] = []
    for i in range(f.shape[1]):
        res = shuffle_null(f[:, i], r, _cov, n=n, seed=seed + i)
        p_values.append(res["p_value"])
        null_means.append(abs(res["null_mean"]))
    return {
        "p_values": p_values,
        "null_mean_abs": float(np.mean(null_means)) if null_means else 0.0,
    }


class Chi(BaseObservable):
    """A12 susceptibility spectrum ``χ_i = Cov_0(f_i, r)`` over a feature bank on base-policy samples.

    Requires activation capture and scores. Captures the base-policy activations, turns them into
    feature values through the feature bank (injected, else the concept layer's default, else a
    graceful no-bank report), scores the same samples, and reports the χ spectrum with a per-feature
    shuffle null. When a ``gold`` signal is supplied it also scores the gold objective and flags the
    predicted hack modes (``χ_i > 0`` with ``χ_gold_i ≤ 0``).

    Gauge is INVARIANT: χ is a within-signal spectrum relative to a fixed feature bank. Comparing
    individual ``χ_i`` across signals requires the same bank read in a shared frame; that is a
    cross-signal comparison the caller must set up, and is noted as a deviation.

    What it cannot do, and the two limits are structural rather than fixable here. It is a first-order
    expansion about ``λ = 0`` on a fixed base-policy bank, which is not the operating point of any run
    in progress; and it is a marginal covariance read as an influence, which is the selection
    differential ``S`` and not the selection gradient ``β``, so a feature can rank high because it
    rides on another feature. The module docstring states both in full and ``BIAS`` below carries the
    second into the registry, where anything consuming this reading inherits it. Neither is a defect
    in this code: fixing them means a different estimator, not a repair of this one.

    Kill condition, inherited from the catalogue record for the susceptibility triple: if the direct
    push ``β`` never reorders features against ``S`` on stored data, the whole transplant is elegant
    and inert. That test is free and should be run before anything is built on top of this index.

    This instrument does not declare ``LINEAR_READOUT``. It reaches the readout only through
    ``final_activations``, which uses the readout's *site* to decide where to capture and never
    touches its vector, and a site exists on every readout whether or not it is linear. The
    declaration was dropped after an audit of all fourteen sites in ``measure/``.
    """

    name = "Chi"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A12"
    deviations = (
        "first-order susceptibility only; the second-order term kappa_3(f, r, r) is not computed",
        "spectrum is relative to the supplied feature bank; cross-signal comparison of individual "
        "chi_i needs the same bank read in a shared frame",
    )

    #: What this rung of `selection.differential_S` is wrong about, in the words a reader can act on.
    #: Direction is ``unknown`` rather than a sign because a suppressor can push an individual
    #: ``χ_i`` either way: a feature that is causally irrelevant but correlated with a rewarded one
    #: reads high, and a feature with a real direct effect that is masked by a correlated opposite
    #: reads low or with the wrong sign entirely. There is no magnitude here because the size of the
    #: gap is ``S - Cβ``, which needs the feature covariance ``C`` this rung does not compute.
    BIAS = BiasStatement(
        direction="unknown",
        why=(
            "a marginal covariance read as an influence, expanded about a regime nobody trains in. "
            "Two separate things are wrong with it and neither is a coding error. It is the "
            "selection differential S, not the gradient beta = C-inverse S, so a feature that "
            "merely rides on a rewarded one ranks high and a feature whose direct effect is masked "
            "by a correlated opposite can rank low or with the wrong sign. And it is evaluated on a "
            "fixed base-policy bank at zero optimisation pressure, against the raw reward r rather "
            "than the advantage A, so it answers what would drift first from the base policy and "
            "not what is drifting now. Rank features on it only against the random-feature and "
            "coherent-irrelevant baselines, and read a single chi_i as an association."
        ),
    )

    # -- the observable declarations ---------------------------------------
    quantity = "selection.differential_S"
    #: The catalogue's access_min for this quantity is ``GRADER: RECORD`` for S, which is the right
    #: floor for an estimator reading a logged bank. This implementation is not that estimator: it
    #: builds the bank itself, so it runs the grader on inputs of its choosing and reads activations
    #: while doing it.
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    #: Exactly the two conditions the catalogue record carries for C1. A third, STATIONARY_GRADER,
    #: is arguably required too, since the bank is scored by one grader and the spectrum means
    #: nothing if that grader moved mid-measurement; it is left out here because adding a condition
    #: the merged catalogue does not carry is an amendment to the record rather than a reading of it.
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.LINEAR_RESPONSE, RegimeCondition.GROUP_NONDEGENERATE}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: ``Cov(f, a·r + b) = a·Cov(f, r)``: the spectrum scales by the reward's own scale factor, so
    #: this is covariant with weight 1 under `reward.affine` and not invariant. The catalogue cell
    #: names `repr.basis` beside it, under which χ *is* invariant, because an orthogonal map acting
    #: on both the activations and the bank leaves every inner product alone. Both are declared:
    #: `resolve_relation` has always read a mapping from group id to `Relation`, and the annotation
    #: on `BaseObservable.invariance_relation` used to forbid the form the kernel implements, so
    #: only the group that constrains the value could be stated and the second was a comment. It is
    #: a declaration now.
    #:
    #: The `repr.basis` half holds **only for a bank whose directions transform with the
    #: representation**, which is what a concept-layer bank is: `h → Qh` and `w → Qw` leave every
    #: `f_i = h·w_i` alone, so χ does not move. Checking it therefore needs a probe that featurises
    #: through the rotating readouts. A probe that reads a fixed coordinate of the activation vector
    #: is measuring a different functional, and χ genuinely moves under it (1.88 against a reading
    #: of order 1 on the retrofit payload, against 1.8e-15 for the rotating bank). The constraint is
    #: recorded here because it is a property of the claim rather than of any one test.
    invariance = "reward.affine"
    invariance_relation = {"reward.affine": COVARIANT_LINEAR, "repr.basis": INVARIANT}
    #: The catalogue's baseline list for C1 opens with "the raw χ the shipped library reports",
    #: which this instrument *is*, so what remains are the two that discriminate: a random feature
    #: and a feature that is semantically coherent and causally irrelevant. The per-feature label
    #: permutation is the third and it is not aspirational: ``_chi_shuffle_null`` computes it on
    #: every run and the p-values ship in the payload.
    baselines = (
        "baseline.random_feature",
        "baseline.coherent_irrelevant_semantic",
        "baseline.label_permutation_null",
    )
    rung = 0

    def __init__(
        self,
        feature_bank: FeatureBank | None = None,
        *,
        gold: Any = None,
        null_draws: int = 2000,
        seed: int = 0,
    ) -> None:
        self.feature_bank = feature_bank
        self.gold = gold
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        bank = self.feature_bank or load_default_bank(signal)
        acts = final_activations(signal, ctx.view, readout=ctx.readout)
        reward = reward_scores(signal, ctx.view, ctx.readout)

        if bank is None:
            return ctx.emit(
                {
                    "chi": [],
                    "feature_names": [],
                    "note": "no feature bank available (concepts absent and none injected)",
                    "n_items": int(acts.shape[0]),
                },
                uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"),
            )

        features = bank.featurize(acts)
        chi = susceptibility(features, reward)
        null = _chi_shuffle_null(features, reward, self.seed, self.null_draws)

        payload: dict[str, Any] = {
            "chi": chi.tolist(),
            "feature_names": list(getattr(bank, "names", tuple())),
            "null_p_values": null["p_values"],
            "null_mean_abs": null["null_mean_abs"],
            "n_items": int(acts.shape[0]),
            "n_features": int(chi.size),
        }

        if self.gold is not None:
            gold_reward = reward_scores(self.gold, ctx.view, ctx.readout)
            chi_gold = susceptibility(features, gold_reward)
            hacks = predicted_hack_modes(chi, chi_gold)
            payload["chi_gold"] = chi_gold.tolist()
            payload["predicted_hack_modes"] = hacks.tolist()
            payload["n_predicted_hacks"] = int(hacks.sum())

        return ctx.emit(payload, uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"))


# ---------------------------------------------------------------------------
# C1: the susceptibility triple
# ---------------------------------------------------------------------------

#: The two sampling distributions the definition of `C` admits. `within_group` is the one the
#: ledger settles on and the default everywhere here; `pooled` is what a flat bank with no group
#: structure has, and it is a legitimate answer for that subject rather than a fallback for a
#: missing argument.
OPERATORS = ("within_group", "pooled")


@dataclass(frozen=True)
class FeatureCovariance:
    """``C = Cov(f, f)`` under a named sampling distribution, with what decides whether to invert it.

    ``conditioning`` is `n_D = Σλ/λ_max` on the **correlation** matrix rather than on `C` itself.
    Both are dimensionless and the formula does not say which, and the choice is not cosmetic: on a
    bank whose features are counted in characters and in proportions, `λ_max` of the raw covariance
    is the character variance and `n_D` reads 1.0019 out of 7 no matter how the features are
    correlated. The number exists to flag the multicollinearity that makes `C⁻¹`
    untrustworthy, and only the correlation form measures that. ``conditioning_raw`` carries the
    literal reading of the formula beside it so the choice is auditable rather than silent.

    ``dof`` is `n - G` for the within-group operator and `n - 1` for the pooled one, which is the
    number of independent contrasts the estimate rests on. A `C` whose `dof` is below `k` is rank
    deficient and cannot be inverted at all; that is a refusal with a remedy, not a pseudo-inverse.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    operator: str
    dof: int
    n_rows: int
    n_groups: int
    n_degenerate: int = 0

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def scales(self) -> np.ndarray:
        """`sd(f_i)` under this operator: the divisor that makes a feature comparable to another."""
        return np.sqrt(np.clip(np.diag(self.matrix), 0.0, None))

    @property
    def correlation(self) -> np.ndarray:
        """`C` with unit diagonal. Constant features keep a zero row rather than dividing by zero."""
        sd = self.scales
        safe = np.where(sd > 0, sd, 1.0)
        corr = np.asarray(self.matrix / np.outer(safe, safe), dtype=np.float64)
        dead = sd <= 0
        corr[dead, :] = 0.0
        corr[:, dead] = 0.0
        return corr

    @property
    def conditioning(self) -> float:
        """`n_D = Σλ/λ_max` on the correlation matrix. `k` means orthogonal, 1 means one direction."""
        return _n_d(self.correlation)

    @property
    def conditioning_raw(self) -> float:
        """The same formula applied to `C` itself, which is the literal reading of it."""
        return _n_d(self.matrix)

    @property
    def constant_features(self) -> tuple[str, ...]:
        """Features with no spread under this operator, so no coefficient is identified for them."""
        return tuple(n for n, s in zip(self.names, self.scales) if s <= 0.0)


def _n_d(matrix: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    top = float(eigs.max()) if eigs.size else 0.0
    return float(eigs.sum() / top) if top > 0 else float("nan")


def feature_covariance(
    features: np.ndarray,
    names: Sequence[str],
    *,
    group_ids: np.ndarray | None = None,
    operator: str = "within_group",
) -> FeatureCovariance:
    """`C = Cov(f, f)` under the within-group or the pooled sampling distribution.

    The within-group form is `C = Σ_g Σ_j (f_gj − f̄_g)(f_gj − f̄_g)ᵀ / (n − G)`, the pooled unbiased
    estimator of the covariance inside a prompt group: each group's own unbiased covariance weighted
    by its `k_g − 1`. It matches `measure.ledger.price.selection_differential` term for term, which
    matters, because `S` and `C` have to be estimated under the same distribution or `β = C⁻¹S` is
    not solving `S = Cβ`. Groups of one contribute nothing and are counted in ``n_degenerate``.

    Passing no ``group_ids`` selects the pooled operator, and that is recorded rather than inferred:
    a bank of independent samples with no prompt structure genuinely has the pooled covariance as
    its sampling distribution, and a bank that has groups and did not pass them is a different
    reading wearing the same name.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    k = f.shape[1]
    if len(names) != k:
        raise ValueError(
            f"{len(names)} names for {k} feature columns. A column without a name cannot be "
            f"reported and a name without a column is a feature nobody measured."
        )
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}; got {operator!r}")

    keep = np.all(np.isfinite(f), axis=1)
    if group_ids is None or operator == "pooled":
        rows = f[keep]
        n = rows.shape[0]
        centred = rows - rows.mean(axis=0) if n else rows
        dof = max(n - 1, 0)
        matrix = (centred.T @ centred) / dof if dof > 0 else np.full((k, k), np.nan)
        return FeatureCovariance(
            names=tuple(names),
            matrix=matrix,
            operator="pooled",
            dof=dof,
            n_rows=n,
            n_groups=0,
        )

    g = np.asarray(group_ids)[keep]
    rows = f[keep]
    acc = np.zeros((k, k), dtype=np.float64)
    n_used = 0
    n_groups = 0
    n_degenerate = 0
    for label in np.unique(g):
        mask = g == label
        size = int(np.count_nonzero(mask))
        if size < 2:
            n_degenerate += 1
            continue
        centred = rows[mask] - rows[mask].mean(axis=0)
        acc += centred.T @ centred
        n_used += size
        n_groups += 1
    dof = n_used - n_groups
    matrix = acc / dof if dof > 0 else np.full((k, k), np.nan)
    return FeatureCovariance(
        names=tuple(names),
        matrix=matrix,
        operator="within_group",
        dof=dof,
        n_rows=n_used,
        n_groups=n_groups,
        n_degenerate=n_degenerate,
    )


def differential(
    features: np.ndarray,
    values: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    operator: str = "within_group",
) -> np.ndarray:
    """`S = Cov(f, A)` under the same operator `feature_covariance` uses.

    ``values`` is the advantage inside a live run and the raw reward on a bank at zero optimisation
    pressure; the arithmetic does not care and the interpretation does, which is what the shipped
    index's docstring is about. Under the within-group operator this is exactly
    `measure.ledger.price.selection_differential(...).value`, recomputed here rather than imported
    so that `measure/indices/` keeps no dependency on `measure/ledger/`; the acceptance test asserts
    the two agree to floating point on the same arrays.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    a = np.asarray(values, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.all(np.isfinite(f), axis=1)
    f, a = f[keep], a[keep]
    if group_ids is None or operator == "pooled":
        if f.shape[0] < 2:
            return np.full(f.shape[1], np.nan)
        centred = f - f.mean(axis=0)
        return np.asarray(centred.T @ (a - a.mean()) / (f.shape[0] - 1), dtype=np.float64)
    g = np.asarray(group_ids)[keep]
    total = np.zeros(f.shape[1], dtype=np.float64)
    n_used = n_groups = 0
    for label in np.unique(g):
        mask = g == label
        size = int(np.count_nonzero(mask))
        if size < 2:
            continue
        total += (f[mask] - f[mask].mean(axis=0)).T @ (a[mask] - a[mask].mean())
        n_used += size
        n_groups += 1
    dof = n_used - n_groups
    return total / dof if dof > 0 else np.full(f.shape[1], np.nan)


@dataclass(frozen=True)
class SusceptibilityTripleReading:
    """`S`, `β` and `Gβ` on one feature basis, with the conditioning that licenses reading `β`.

    ``response`` is `Gβ`, the predicted per-step response, and it is None until a `G` is supplied.
    That is the honest state of the triple at `GRADER: RECORD` access: `S` and `β` are both free from
    a scored group and `G` needs a Fisher solve against the policy, so a reading with no `G` is a
    pair rather than a triple and says so instead of substituting `C` for `G`.

    ``basis`` is not decoration. `β` is a direct effect **conditional on the measured feature set**,
    which is the Table 2 fallacy: reading off every coefficient of a multiple regression as a causal
    effect is valid only if the measured features contain every trait selection acts on, and they
    never do. The basis travels with the number because the basis is part of the claim.
    """

    names: tuple[str, ...]
    differential_S: np.ndarray
    gradient_beta: np.ndarray
    gradient_beta_standardised: np.ndarray
    differential_S_standardised: np.ndarray
    covariance: FeatureCovariance
    ridge: float
    response: np.ndarray | None = None
    response_source: str = ""

    @property
    def basis(self) -> tuple[str, ...]:
        return self.names

    @property
    def conditioning(self) -> float:
        return self.covariance.conditioning

    @property
    def sign_disagreements(self) -> tuple[str, ...]:
        """Features where `S` and `β` disagree in sign: the suppressors, named."""
        s, b = self.differential_S, self.gradient_beta
        both = np.isfinite(s) & np.isfinite(b)
        return tuple(
            n
            for n, ok, si, bi in zip(self.names, both, np.sign(s), np.sign(b))
            if ok and si != 0 and bi != 0 and si != bi
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "names": list(self.names),
            "S": self.differential_S.tolist(),
            "beta": self.gradient_beta.tolist(),
            "S_standardised": self.differential_S_standardised.tolist(),
            "beta_standardised": self.gradient_beta_standardised.tolist(),
            "operator": self.covariance.operator,
            "conditioning_n_D": self.conditioning,
            "conditioning_n_D_raw": self.covariance.conditioning_raw,
            "ridge": self.ridge,
            "dof": self.covariance.dof,
            "n_rows": self.covariance.n_rows,
            "n_groups": self.covariance.n_groups,
            "sign_disagreements": list(self.sign_disagreements),
        }
        if self.response is not None:
            out["response_Gbeta"] = self.response.tolist()
            out["response_source"] = self.response_source
        return out

    def render(self) -> str:
        head = (
            f"susceptibility triple  operator {self.covariance.operator}, "
            f"n_D {self.conditioning:.3f} of {len(self.names)}, dof {self.covariance.dof}"
        )
        lines = [head]
        for i, name in enumerate(self.names):
            row = (
                f"    {name:<20} S {self.differential_S[i]:+.5g}  "
                f"β {self.gradient_beta[i]:+.5g}  β̃ {self.gradient_beta_standardised[i]:+.5g}"
            )
            if self.response is not None:
                row += f"  Gβ {self.response[i]:+.5g}"
            lines.append(row)
        if self.sign_disagreements:
            lines.append(
                f"    S and β disagree in sign on: {', '.join(self.sign_disagreements)}. Those "
                f"features covary with reward through something else in the basis."
            )
        return "\n".join(lines)


def selection_gradient(
    differential_S: np.ndarray,
    covariance: FeatureCovariance,
    *,
    ridge: float = 0.0,
    instrument: str = "SusceptibilityTriple",
) -> tuple[np.ndarray, np.ndarray] | Refusal:
    """`β = C⁻¹S`, with `S` and `C` estimated under the same operator. Raw and standardised.

    ``ridge`` is `δ` in `C + δ·(tr C / k)·I`, which is the shrinkage target the multivariate-selection
    literature uses and it defaults to zero. Do not reach for it to make a badly conditioned `C`
    behave: shrinking toward a scaled identity drives `β → S/(δ·tr C/k)`, so a heavily shrunk `β`
    reproduces `S` exactly and has stopped being a gradient. On the campaign bank the sign flips
    vanish at `δ ≈ 0.09` and `ρ(S, β) = 1.0` by `δ = 1`, and Ledoit-Wolf picks `δ = 0.738` there,
    well past the point where the estimator answers a different question. An estimator that minimises
    the Frobenius risk of `Ĉ` is not chosen to preserve the ordering `β` exists to change.

    Refuses rather than pseudo-inverting when `C` is singular, because a pseudo-inverse silently
    projects the answer onto the directions that happened to vary and reports the projection as a
    coefficient. Constant features are named in the refusal, since dropping them by name is the fix.
    """
    S = np.asarray(differential_S, dtype=np.float64).ravel()
    C = np.asarray(covariance.matrix, dtype=np.float64)
    k = covariance.k
    if S.size != k:
        raise ValueError(f"S has {S.size} entries and C is {k}x{k}; they describe different bases")

    constant = covariance.constant_features
    if constant:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"{len(constant)} of {k} features have no spread under the "
                f"{covariance.operator} operator ({', '.join(constant)}), so C is singular and no "
                f"coefficient is identified for them or for anything correlated with them"
            ),
            remedy=(
                f"drop {', '.join(constant)} from the basis and re-run, or widen the sample until "
                f"they vary. A feature that never moved carries no information about what selection "
                f"did to it, and dropping it by name keeps the surviving coefficients honest about "
                f"which basis they are conditional on."
            ),
            statistics={
                "constant_features": list(constant),
                "operator": covariance.operator,
                "dof": covariance.dof,
            },
        )

    if covariance.dof < k:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the {covariance.operator} covariance has {covariance.dof} degrees of freedom for "
                f"{k} features, so C is rank deficient and C⁻¹S is not defined"
            ),
            remedy=(
                f"pool the covariance over a window of steps until the degrees of freedom exceed "
                f"{k}, or reduce the basis to at most {covariance.dof} features. Both are stated "
                f"choices; a pseudo-inverse here would be a silent one."
            ),
            statistics={"dof": covariance.dof, "k": k, "operator": covariance.operator},
        )

    target = C
    if ridge:
        target = C + float(ridge) * (float(np.trace(C)) / k) * np.eye(k)
    try:
        beta = np.linalg.solve(target, S)
    except np.linalg.LinAlgError as exc:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=f"C is numerically singular under the {covariance.operator} operator: {exc}",
            remedy=(
                "report S alone at rung 0, or supply a ridge δ and state it. n_D on the correlation "
                "matrix says how much of the basis is one direction wearing several names."
            ),
            statistics={
                "conditioning_n_D": covariance.conditioning,
                "operator": covariance.operator,
            },
        )
    return beta, covariance.scales * beta


def susceptibility_triple(
    features: np.ndarray,
    values: np.ndarray,
    names: Sequence[str],
    *,
    group_ids: np.ndarray | None = None,
    operator: str = "within_group",
    ridge: float = 0.0,
    metric_G: np.ndarray | None = None,
    response_source: str = "",
    instrument: str = "SusceptibilityTriple",
) -> SusceptibilityTripleReading | Refusal:
    """C1 in one call: `S`, `β = C⁻¹S` and, when a `G` is supplied, the response `Gβ`.

    `metric_G` is `G = J F⁻¹ Jᵀ` in the same basis and the same order as ``names``. It is an
    argument rather than something computed here, for two reasons that both matter. It needs a
    backward pass against the policy and this package is import-free of torch by contract; and the
    basis has to be identical element for element, which a caller can guarantee and a helper cannot.
    Without it the reading carries `S` and `β` and no response, which is the true state of the
    measurement at `GRADER: RECORD` rather than a gap to be filled with `C`.
    """
    S = differential(features, values, group_ids=group_ids, operator=operator)
    C = feature_covariance(features, names, group_ids=group_ids, operator=operator)
    got = selection_gradient(S, C, ridge=ridge, instrument=instrument)
    if isinstance(got, Refusal):
        return got
    beta, beta_std = got

    response: np.ndarray | None = None
    if metric_G is not None:
        G = np.asarray(metric_G, dtype=np.float64)
        if G.shape != (C.k, C.k):
            raise ValueError(
                f"G is {G.shape} and the basis has {C.k} features. `Δz = ηGβ` is only defined when "
                f"G, β and the feature names are one basis in one order."
            )
        response = G @ beta

    scales = C.scales
    safe = np.where(scales > 0, scales, np.nan)
    return SusceptibilityTripleReading(
        names=tuple(names),
        differential_S=S,
        gradient_beta=beta,
        gradient_beta_standardised=beta_std,
        differential_S_standardised=S / safe,
        covariance=C,
        ridge=float(ridge),
        response=response,
        response_source=response_source,
    )


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

#: C1's envelope, verbatim from the catalogue record. `LINEAR_RESPONSE` because `S = Cβ` is a
#: first-order expansion and nothing here is true outside it; `GROUP_NONDEGENERATE`
#: because a group with no score spread contributes no contrast to either `S` or `C`.
TRIPLE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.LINEAR_RESPONSE, RegimeCondition.GROUP_NONDEGENERATE}),
    measured_by=MEASURED_BY,
    on_violation="refuse",
)

#: The suppressor caveat, attached where the number is reported rather than kept on a caveats page.
TABLE_2_FALLACY = (
    "beta is a direct effect conditional on the measured basis. Reading every coefficient of a "
    "multiple regression as a causal effect is valid only when the measured features contain every "
    "trait selection acts on, and they never do; the evolutionary-biology name for the mistake is "
    "the Table 2 fallacy and it applies here without translation. The basis is part of the claim, "
    "so it ships on the reading."
)

#: The other direction of the argument, stated where it can be checked rather than discovered in
#: review. Haufe et al. (NeuroImage 87:96-110, 2014) hold that a pattern `a` is proportional to
#: `Σ_x w` and a filter `w` to `Σ_x⁻¹ a`, and that the filter is precisely what assigns importance
#: to variables with zero association to the target. `chi = Cov(f, r)` is the pattern; `beta = C⁻¹S`
#: is the filter. Both are correct for their own question and this one is genuinely a direct-effect
#: question, so beta is defensible; the convergence is worth stating because it earns trust.
#: Morrissey, Kruuk and Wilson (2010) and Haufe (2014), forty years and two fields apart, both
#: concluded the marginal covariance is the safer thing to publish, which is why `S` stays primary.
PATTERN_VERSUS_FILTER = (
    "S is the pattern and beta is the filter, in Haufe's sense. A filter assigns weight to a "
    "variable with zero association to the target whenever that variable cancels noise in another, "
    "so a large |beta| on a feature with S near zero is a suppressor relationship and not a "
    "discovery. Report S first, beta beside it, and the conditioning of C beside both."
)


class _TripleInstrument(BaseObservable):
    """Shared plumbing for the three readings of the susceptibility triple.

    The data path is the shipped index's: capture base-policy activations, turn them into feature
    values through the feature bank, score the same samples. What is added is the covariance
    operator and the group labels it needs, so the same bank can be read under either sampling
    distribution and the reading says which.

    What these instruments cannot do, and it is structural. Without ``group_ids`` there is no
    within-group operator to compute, so a flat bank is read pooled and the reading says so; that
    operator needs a record that knows which rollouts answered which prompt. And none of them
    produces `G`, so `Gβ` is present only when a caller supplies one.
    """

    capabilities = Capability.ACTIVATIONS | Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "the selection differential and gradient"
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = TRIPLE_ENVELOPE
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    baselines = (
        "baseline.chi_marginal_covariance",
        "baseline.random_feature",
        "baseline.coherent_irrelevant_semantic",
    )
    deviations = (
        "S is Cov(f, r) on a scored bank rather than Cov(f, A) inside a live run; on a bank at "
        "zero optimisation pressure there is no advantage to take",
        "no G is computed here, so the response Gbeta is present only when a caller supplies G",
    )

    def __init__(
        self,
        feature_bank: FeatureBank | None = None,
        *,
        group_ids: np.ndarray | None = None,
        operator: str = "within_group",
        ridge: float = 0.0,
        metric_G: np.ndarray | None = None,
        response_source: str = "",
    ) -> None:
        self.feature_bank = feature_bank
        self.group_ids = None if group_ids is None else np.asarray(group_ids)
        self.operator = operator
        self.ridge = float(ridge)
        self.metric_G = metric_G
        self.response_source = response_source

    def compute(self, ctx: Context) -> SusceptibilityTripleReading | Refusal:
        signal = ctx.signal
        if signal is None:
            # `Context.signal` is optional because `preflight` never touches it, so an instrument
            # that does has to say what it needs rather than dereference a None one line later.
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no reward signal on the context, so there is nothing to score or capture",
                remedy=(
                    "put the grader on the context as `signal`. This instrument reads activations "
                    "and scores from the same samples, and both come from the signal."
                ),
            )
        bank = self.feature_bank or load_default_bank(signal)
        acts = final_activations(signal, ctx.view, readout=ctx.readout)
        if bank is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"no feature bank is available for this signal, so there is no basis to "
                    f"regress on; {int(acts.shape[0])} activations were captured"
                ),
                remedy=(
                    "pass a FeatureBank to the constructor, or fit the concept layer's default bank "
                    "for this signal first. beta is a coefficient in a named basis and there is no "
                    "basis-free version of it."
                ),
            )
        reward = reward_scores(signal, ctx.view, ctx.readout)
        features = bank.featurize(acts)
        names = tuple(getattr(bank, "names", tuple())) or tuple(
            f"f{i}" for i in range(features.shape[1])
        )
        gids = self.group_ids
        if gids is not None and gids.shape[0] != features.shape[0]:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"{gids.shape[0]} group labels for {features.shape[0]} scored rows, so the "
                    f"within-group partition does not describe this bank"
                ),
                remedy=(
                    "supply one group label per row of the same view this instrument scored, or "
                    "pass operator='pooled' to read the bank as unstructured samples."
                ),
            )
        return susceptibility_triple(
            features,
            reward,
            names,
            group_ids=gids,
            operator=self.operator,
            ridge=self.ridge,
            metric_G=self.metric_G,
            response_source=self.response_source,
            instrument=self.name,
        )

    def payload(self, reading: SusceptibilityTripleReading) -> dict[str, Any]:
        raise NotImplementedError

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return self.measure(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        got = self.compute(ctx)
        if isinstance(got, Refusal):
            # `measure` is typed to Evidence by the 2.0.1 protocol and `estimate` is the typed
            # entry point, so a refusal reaching here is returned through the same object rather
            # than raised. Callers on the `run(...)` path get the Refusal as the value.
            return got  # type: ignore[return-value]
        payload = dict(got.as_dict())
        payload.update(self.payload(got))
        payload["caveats"] = [TABLE_2_FALLACY, PATTERN_VERSUS_FILTER]
        # `estimate` calls `measure` directly rather than going through `measure.base.run`, so
        # nothing else sets `ctx._observable` and `emit` would otherwise write the reading as
        # `anonymous` with `quantity=""`. The previous value is restored rather than cleared, so a
        # nested call keeps its own identity.
        previous = ctx._observable
        ctx._observable = self  # type: ignore[assignment]
        try:
            return ctx.emit(
                payload,
                uncertainty=Uncertainty(n=int(got.covariance.n_rows), method="none"),
            )
        finally:
            ctx._observable = previous


class SelectionGradientIndex(_TripleInstrument):
    """C1 rung 1: `β = C⁻¹S`, the direct push, with the conditioning of `C` beside it.

    `Cov(f_i, A)` is a selection *differential*: the marginal association between a trait and
    fitness, which conflates direct selection with selection on everything the trait correlates
    with. The gradient is the coefficient vector of the multiple regression of fitness on all traits
    jointly, `S = Cβ` so `β = C⁻¹S`, and **`S` and `β` can differ in sign**. On the campaign's own
    seven-feature bank they do: `len_tokens` has the second-largest positive `S` and a negative `β`,
    because it correlates 0.974 with `len_chars` and the joint fit gives the whole positive direct
    effect to `len_chars`.

    What it cannot do. `β` is conditional on the measured basis, which is the Table 2 fallacy and is
    not fixable by a better estimator; and `β` is regression against the **observed** covariance `C`,
    while the response to selection uses the **reachable** covariance `G`. `C = G + N` with `N ⪰ 0`,
    so a feature can have a large `β` and still not move. That is C2's question and this instrument
    does not answer it.
    """

    name = "SelectionGradient"
    version = "1.0"
    quantity = "selection.gradient_beta"
    #: `Cov(f, a·r + b) = a·Cov(f, r)` and `C` does not move, so `β` scales by `a`: covariant with
    #: weight 1 under `reward.affine`, exactly as `χ` is. Under `repr.basis` an orthogonal map on
    #: the activations and the bank leaves every feature value alone, so `β` is invariant there.
    #: Both are declared through the mapping form of `invariance_relation`, which
    #: `resolve_relation` reads.
    invariance = "reward.affine"
    invariance_relation = {"reward.affine": COVARIANT_LINEAR, "repr.basis": INVARIANT}
    rung = 1

    #: The bias of this rung, in the words a reader can act on. Direction is unknown rather than a
    #: sign because omitting a trait selection acts on moves a coefficient either way: an omitted
    #: cause correlated with a measured trait inflates the measured trait's `β`, and an omitted
    #: suppressor deflates it.
    BIAS = BiasStatement(
        direction="unknown",
        why=(
            "a direct effect conditional on the measured basis. Every trait selection acts on that "
            "is not in the basis loads onto whichever measured features correlate with it, so a "
            "coefficient here is the direct effect of the feature plus the indirect effect of "
            "everything unmeasured it stands in for. Adding a feature can change the sign of "
            "another. Report it with the basis, against S, and with the conditioning n_D, and read "
            "a single beta as a direct effect within this basis rather than as an influence."
        ),
    )

    def payload(self, reading: SusceptibilityTripleReading) -> dict[str, Any]:
        return {"primary": "beta", "rung": self.rung}


class SelectionResponseIndex(_TripleInstrument):
    """C1 rung 3: the predicted response `Δz = η G β`, which needs a `G` and refuses without one.

    This is the third member of the triple and the one that answers "what will actually move".
    `G = J F⁻¹ Jᵀ` is the covariance a parameter move can reach, so `Gβ` is the part of the direct
    push the parameterisation can deliver. Supplying `C` in place of `G` would make `Gβ` equal `S`
    identically and turn the triple into two numbers wearing three names, so a missing `G` is a
    refusal naming what would produce one.
    """

    name = "SelectionResponse"
    version = "1.0"
    quantity = "selection.response_Gbeta"
    #: `Gβ` scales with the reward's own scale through `β`, and `G` is a Fisher-metric object that
    #: does not move under a reparameterisation of the policy.
    invariance = "reward.affine"
    invariance_relation = {"reward.affine": COVARIANT_LINEAR, "policy.reparam": INVARIANT}
    rung = 3
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD, Component.POLICY: Access.BACKWARD}

    def compute(self, ctx: Context) -> SusceptibilityTripleReading | Refusal:
        if self.metric_G is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "the response Gβ needs G = J F⁻¹ Jᵀ in this feature basis and none was supplied"
                ),
                remedy=(
                    "compute G against the policy checkpoint (k Jacobian-vector products plus k "
                    "Fisher solves, minutes on a small model) and pass it as metric_G in the same "
                    "feature order, or read SelectionGradient for beta alone at rung 1. Do not pass "
                    "C here: Gβ would then equal S identically and the triple would be two numbers."
                ),
            )
        return super().compute(ctx)

    def payload(self, reading: SusceptibilityTripleReading) -> dict[str, Any]:
        return {"primary": "response_Gbeta", "rung": self.rung}


class SelectionConditioning(_TripleInstrument):
    """C1 and C2's shared scalar: `n_D = Σλ/λ_max`, how many directions the basis really has.

    It ships beside every `β` because `C⁻¹` is only as trustworthy as `C` is conditioned, and a
    basis whose `n_D` is far below `k` is one direction wearing several names. On the campaign's
    seven surface features `n_D` is 3.33 pooled and 3.47 within-group, so seven named features carry
    between three and four independent directions and four of the seven coefficients are being
    identified off shared variance.
    """

    name = "SelectionConditioning"
    version = "1.0"
    quantity = "selection.G_conditioning"
    #: `n_D` is a function of `C` alone, and an affine rescaling of the reward does not touch `C`.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    rung = 1

    def payload(self, reading: SusceptibilityTripleReading) -> dict[str, Any]:
        return {"primary": "conditioning_n_D", "rung": self.rung}


__all__ = [
    "OPERATORS",
    "PATTERN_VERSUS_FILTER",
    "TABLE_2_FALLACY",
    "TRIPLE_ENVELOPE",
    "Chi",
    "FeatureCovariance",
    "SelectionConditioning",
    "SelectionGradientIndex",
    "SelectionResponseIndex",
    "SusceptibilityTripleReading",
    "differential",
    "feature_covariance",
    "predicted_hack_modes",
    "selection_gradient",
    "susceptibility",
    "susceptibility_triple",
]
