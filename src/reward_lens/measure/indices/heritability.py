"""C2 h²: which behavioural features a gradient step can move at all, and what moves with them.

The derivation is worth following, because the quantity falls out of it rather than being
transplanted. Let `J = ∂z/∂θ`, so by the score-function identity
`J_i = ∇_θ E[f_i] = E[f_i ∇_θ log π]`. Let `F = E[∇log π ∇log πᵀ]`. A natural-gradient step is
`Δθ = η F⁻¹ g` with `g = E[A ∇_θ log π]`, so to first order `Δz = J Δθ = η J F⁻¹ E[A ∇log π]`.
Decompose the advantage in the feature basis under the sampling distribution,
`A = Σ_i β_i (f_i − E f_i) + ε` with `ε` orthogonal to the feature span and `β = C⁻¹S` exactly the
selection-gradient regression coefficient. Then `E[A ∇log π] = Jᵀβ + e` with `e = E[ε ∇log π]`,
giving

    Δz = η G β + η J F⁻¹ e,   where   G := J F⁻¹ Jᵀ

which is Lande's equation with the residual named: `η J F⁻¹ e` is the part of the response driven by
whatever in the advantage the feature set does not explain, and its size is a measure of how good
the basis is.

Now the part that produces the diagnostic. For any unit `v`, write `u = v·f`. Then
`Jᵀv = E[u ∇log π]` and `vᵀGv = E[u ∇log π]ᵀ F⁻¹ E[u ∇log π] = sup_{dᵀFd = 1} (E[u (d·∇log π)])²`,
and by Cauchy-Schwarz `E[u (d·∇log π)]² ≤ Var(u)·dᵀFd = Var(u)`. So `vᵀGv ≤ vᵀCv` for every `v`,
hence `C ⪰ G` and `N := C − G ⪰ 0`. **`N` is exactly the part of feature variance no parameter
perturbation can produce.** Equality holds iff the feature lies in the closure of the span of
score-function directions, which is the case for a fully parameterised exponential family (where
`J = F = C` and `G = C`) and is not the case for a neural policy and an arbitrary feature.

Per feature,

    h²_i = G_ii / C_ii ∈ [0, 1]

the heritability of a behavioural feature under a policy parameterisation: the fraction of a
feature's observed variance across rollouts that a parameter move can reach. **A feature with
`h² ≈ 0` varies across rollouts purely by sampling and cannot be moved by any gradient step, no
matter how hard selection acts on it.** `χ` ranks by `S`, and a feature sitting mostly in `N` has
`S ≠ 0` and never responds. It looks susceptible and is inert. That is not estimation error, it is
the wrong operator, and this is the one-number test.

**Prior art, cited rather than rediscovered.** Heritability as a diagnostic on an evolved policy's
representation is not new: De Carlo, Ferrante, Zeeuwe, Ellers and Eiben (arXiv 2110.11187, journal
version *Evolutionary Intelligence*, 2023, doi 10.1007/s12065-023-00860-0) compute `h²` of
morphological and behavioural traits of evolved robots, compare direct against indirect encodings,
and track it over the course of evolution, explicitly to ask which traits the representation can
transmit. Same question, genetic rather than gradient machinery: no Fisher metric and no language
models. The companion indices are Hansen and Houle (2008), with eighteen years of estimator theory
attached.

**The terminological trap, carried rather than glossed.** Hansen and Pélabon have a paper titled
"Heritability is not Evolvability" and the evolutionary-computation field draws the distinction
sharply. Here `h² = G_ii/C_ii` is a **ratio**: dimensionless, bounded in [0, 1], about the
*proportion* of variance that is reachable. `e(β) = βᵀGβ` is a **scale**: it has the units of a
feature squared and it is about the *amount* of achievable response. A feature can have `h² = 0.9`
and a tiny `e`, or `h² = 0.05` on a feature so variable that the reachable 5% is still the largest
absolute response in the basis. Both ship on every reading and neither substitutes for the other.

**What this module does not do.** It does not compute `G`. `G = J F⁻¹ Jᵀ` needs `k` Jacobian-vector
products and `k` Fisher solves against the policy, `measure/indices/` is import-free of torch by
contract, and `measure/efficiency/` is where that computation lives for the cost book. `G` arrives
here as an argument in a named basis, and the rung-0 estimator below needs no `G` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BiasStatement
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import ANY_SUBSTRATE, MEASURED_BY

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: Every phase in which a training record exists to read. C2's rung 0 reads a finished record and
#: rung 2 reads a checkpoint, so the pre-run phase is the one it does not answer in: before a run
#: starts there is no `Δz` to regress and no checkpoint to take a Fisher against.
RECORD_PHASES: frozenset[Phase] = frozenset({Phase.IN_RUN, Phase.POST_RUN})

#: C2's envelope, verbatim from the catalogue record. One condition, and it is the load-bearing one:
#: every line of the derivation above is a first-order expansion, and `LINEAR_RESPONSE` is measured
#: by `selection.explained_fraction`, the fraction of the observed motion the selection term
#: explains. When `Λ` is near zero the response is not being driven by the selection differential at
#: all, so the ratio `Δz / (η·S)` is a ratio of two things that are not related by the breeder's
#: equation, and every rung of this instrument is reading noise. That check is the envelope's job
#: rather than a threshold invented here, which is why this instrument sets no threshold of its own.
HERITABILITY_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.LINEAR_RESPONSE}),
    measured_by={RegimeCondition.LINEAR_RESPONSE: MEASURED_BY[RegimeCondition.LINEAR_RESPONSE]},
    on_violation="refuse",
)

#: What C2 exists to say, in the sentence a reader acts on. Kept as a constant because it belongs on
#: every reading rather than in a caveats page.
INERT_BUT_SUSCEPTIBLE = (
    "a feature with h2 near zero varies across rollouts and no gradient step can move it. If a "
    "susceptibility index ranked it high, the index was reading the observed covariance C where the "
    "reachable covariance G belongs, and C = G + N with N the part of the variance no parameter "
    "perturbation can produce."
)


# ---------------------------------------------------------------------------
# Rung 2: h2 from a supplied G
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeritabilityReading:
    """Per-feature `h² = G_ii/C_ii`, with the bound check that makes it self-verifying.

    ``undefined`` names the features with `C_ii = 0`. Their heritability is `0/0`, which is not zero
    and not one: a feature that did not vary across rollouts carries no variance to be reachable or
    unreachable, so the question does not apply to it. They carry NaN and their names travel on the
    reading, because a heritability vector with silent NaNs in it is exactly the object somebody
    sorts.

    ``bound_violations`` names the features whose `h²` exceeds 1 beyond floating-point tolerance.
    `C ⪰ G` is a theorem, so that is an instrument bug in whatever produced `G` or `C` and not a
    finding, in the same way an efficiency above 1 is. It is reported by name rather than clipped.
    """

    names: tuple[str, ...]
    h2: np.ndarray
    G_diagonal: np.ndarray
    C_diagonal: np.ndarray
    rung: int
    method: str
    damping: float | None = None
    damping_stable: bool | None = None
    n_samples: int = 0
    undefined: tuple[str, ...] = ()
    bound_violations: tuple[str, ...] = ()
    psd_residual_min_eigenvalue: float = float("nan")
    notes: tuple[str, ...] = ()

    @property
    def inert(self) -> tuple[str, ...]:
        """Features whose reachable share of their own variance is under 1%."""
        return tuple(n for n, h in zip(self.names, self.h2) if np.isfinite(h) and h < 0.01)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "h2": [None if not np.isfinite(v) else float(v) for v in self.h2],
            "G_diagonal": self.G_diagonal.tolist(),
            "C_diagonal": self.C_diagonal.tolist(),
            "rung": self.rung,
            "method": self.method,
            "damping": self.damping,
            "damping_stable": self.damping_stable,
            "n_samples": self.n_samples,
            "undefined": list(self.undefined),
            "bound_violations": list(self.bound_violations),
            "psd_residual_min_eigenvalue": self.psd_residual_min_eigenvalue,
            "inert": list(self.inert),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [f"heritability  rung {self.rung} ({self.method}), n = {self.n_samples}"]
        for n, h, g, c in zip(self.names, self.h2, self.G_diagonal, self.C_diagonal):
            state = "undefined (C_ii = 0)" if not np.isfinite(h) else f"h2 {h:.4f}"
            lines.append(f"    {n:<20} {state:<22} G_ii {g:.5g}  C_ii {c:.5g}")
        if self.inert:
            lines.append(f"    inert (h2 < 0.01): {', '.join(self.inert)}")
        return "\n".join(lines)


def heritability(
    metric_G: np.ndarray,
    feature_covariance: np.ndarray,
    names: Sequence[str],
    *,
    rung: int = 2,
    method: str = "fisher_solve",
    damping: float | None = None,
    damping_stable: bool | None = None,
    n_samples: int = 0,
    tol: float = 1e-8,
    instrument: str = "FeatureHeritability",
) -> HeritabilityReading | Refusal:
    """`h²_i = G_ii / C_ii` with the `C ⪰ G` bound checked rather than assumed.

    `G` and `C` must be in the same basis in the same order, which is why ``names`` is required
    rather than optional: two matrices in two orders produce a heritability vector that is a valid
    ratio of the wrong pairs, and nothing downstream can detect it.

    The bound check is on `N = C − G` and it is done two ways, because the two catch different bugs.
    Per feature, `h²_i > 1 + tol` means `N_ii < 0`, which is a diagonal violation and usually means
    `G` and `C` were estimated on different samples. On the whole matrix, the smallest eigenvalue of
    `N` below `−tol·λ_max(C)` means `N` is not positive semi-definite even where the diagonal looks
    fine, which is the violation a per-feature check cannot see and which usually means the Fisher
    solve did not converge. Both are reported; a diagonal violation refuses, because `h² > 1` is a
    number nobody should be handed.
    """
    G = np.asarray(metric_G, dtype=np.float64)
    C = np.asarray(feature_covariance, dtype=np.float64)
    k = len(names)
    if G.shape != (k, k) or C.shape != (k, k):
        raise ValueError(
            f"G is {G.shape} and C is {C.shape} for {k} names. h2 is a ratio of the diagonals of "
            f"two matrices in one basis in one order, and there is no way to check an order that "
            f"was not supplied."
        )

    gd = np.diag(G).astype(np.float64).copy()
    cd = np.diag(C).astype(np.float64).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        h2 = np.where(cd > 0, gd / np.where(cd > 0, cd, 1.0), np.nan)
    undefined = tuple(n for n, c in zip(names, cd) if not c > 0)

    if len(undefined) == k:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"every feature in the basis has zero observed variance, so h2 = G_ii/C_ii is 0/0 "
                f"for all {k} of them"
            ),
            remedy=(
                "widen the sample until the features vary, or choose a basis this policy's "
                "rollouts actually move on. A feature that never differed between rollouts has no "
                "variance for a parameter move to reach or fail to reach."
            ),
            statistics={"names": list(names), "C_diagonal": cd.tolist()},
        )

    violations = tuple(n for n, v in zip(names, h2) if np.isfinite(v) and v > 1.0 + tol)
    residual = C - G
    scale = float(np.max(np.abs(np.diag(C)))) or 1.0
    try:
        min_eig = float(np.linalg.eigvalsh((residual + residual.T) / 2.0).min())
    except np.linalg.LinAlgError:
        min_eig = float("nan")

    if violations:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"h2 exceeds 1 on {', '.join(violations)}, so N = C - G has a negative diagonal "
                f"entry. C >= G is a theorem (Cauchy-Schwarz on the score-function directions), "
                f"so this is a bug in G or in C and not a finding: max h2 = "
                f"{float(np.nanmax(h2)):.6g}"
            ),
            remedy=(
                "estimate G and C on the same rollouts in the same basis and the same order. The "
                "usual cause is a G computed at one checkpoint against a C pooled over a window of "
                "steps, and the fix is to pool both or neither. If they already match, the Fisher "
                "solve has not converged: raise the damping, check the solution is stable across a "
                "decade of it, and report the value you used."
            ),
            statistics={
                "bound_violations": list(violations),
                "max_h2": float(np.nanmax(h2)),
                "min_eigenvalue_C_minus_G": min_eig,
                "damping": damping,
            },
        )

    notes: list[str] = []
    if np.isfinite(min_eig) and min_eig < -tol * scale:
        notes.append(
            f"N = C - G has a negative eigenvalue ({min_eig:.4g} against a C scale of {scale:.4g}) "
            f"even though every diagonal entry is non-negative. The per-feature ratios are within "
            f"bounds and the matrix is not, so an off-diagonal of G or C is wrong; treat any "
            f"quadratic form in G, including the evolvability indices, as unverified here."
        )
    if undefined:
        notes.append(
            f"{len(undefined)} of {k} features have zero observed variance and carry NaN rather "
            f"than a heritability: {', '.join(undefined)}."
        )

    return HeritabilityReading(
        names=tuple(names),
        h2=h2,
        G_diagonal=gd,
        C_diagonal=cd,
        rung=rung,
        method=method,
        damping=damping,
        damping_stable=damping_stable,
        n_samples=int(n_samples),
        undefined=undefined,
        bound_violations=(),
        psd_residual_min_eigenvalue=min_eig,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The Hansen and Houle indices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolvabilityReading:
    """`e(β)`, `c(β)` and `a(β) = c/e`, on a unit-normalised direction.

    Hansen and Houle define all three on a unit selection gradient, and the normalisation is not
    bookkeeping: `e` and `c` are both homogeneous of degree two in `‖β‖`, so their *ratio* is
    scale-free and each of them alone is not. Reporting `e` without saying `β` was normalised makes
    it a number in the units of `β` squared times a feature squared, which is not a quantity.

    ``beta_outside_G`` is the fraction of `β`'s squared length lying outside the range of `G`. When
    it is non-zero, `βᵀG⁻¹β` is infinite and `c` is exactly zero: the direction has a component no
    parameter move can produce at all, so no part of the response along it is autonomous. That is a
    real answer rather than a numerical failure, and it is reported as one.
    """

    evolvability: float
    conditional_evolvability: float
    autonomy: float
    rank_G: int
    k: int
    beta_outside_G: float
    conditioning: float
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "evolvability": self.evolvability,
            "conditional_evolvability": self.conditional_evolvability,
            "autonomy": self.autonomy,
            "rank_G": self.rank_G,
            "k": self.k,
            "beta_outside_G": self.beta_outside_G,
            "conditioning_n_D": self.conditioning,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        return (
            f"evolvability e(β) = {self.evolvability:.5g}, conditional c(β) = "
            f"{self.conditional_evolvability:.5g}, autonomy a(β) = {self.autonomy:.4f} "
            f"({100 * (1 - self.autonomy):.0f}% of the response to this pressure is collateral)"
        )


def evolvability_indices(
    metric_G: np.ndarray,
    beta: np.ndarray,
    *,
    rank_tol: float = 1e-10,
    instrument: str = "Autonomy",
) -> EvolvabilityReading | Refusal:
    """`e(β) = βᵀGβ`, `c(β) = 1/(βᵀG⁻¹β)`, `a(β) = c/e ∈ [0, 1]` on a unit-normalised `β`.

    Autonomy is what a practitioner can act on before a run starts: `a(β) = 0.31` says 69% of the
    response to this selection pressure is collateral movement in traits correlated with the one
    being pushed on. Low autonomy means you cannot push on what the reward wants without moving
    everything entangled with it, and it is computable from `G` and `β` with no RL run at all.

    `G⁻¹` is taken through the eigendecomposition rather than `solve`, because `G` is routinely
    singular: it is `J F⁻¹ Jᵀ` with `J` of rank at most the number of rollouts, and a `solve` on it
    either raises or returns a large arbitrary number depending on rounding. Directions inside the
    range of `G` get the true inverse; a component of `β` outside it makes `βᵀG⁻¹β` infinite, `c`
    exactly zero and `a` exactly zero, which is the correct statement and not a degenerate case.
    """
    G = np.asarray(metric_G, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64).ravel()
    k = b.size
    if G.shape != (k, k):
        raise ValueError(f"G is {G.shape} and beta has {k} entries; they describe different bases")
    norm = float(np.linalg.norm(b))
    if not norm > 0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail="beta is the zero vector, so there is no direction of selection to evaluate",
            remedy=(
                "supply the selection gradient from a group where the advantage actually varied. "
                "A zero beta means no feature in the basis was selected on at all, which is a "
                "finding about the step rather than an input to the evolvability indices."
            ),
        )
    u = b / norm

    sym = (G + G.T) / 2.0
    eigs, vecs = np.linalg.eigh(sym)
    top = float(eigs.max()) if eigs.size else 0.0
    if not top > 0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail="G has no positive eigenvalue, so no direction in this basis is reachable at all",
            remedy=(
                "check the Fisher solve: a G that is numerically zero usually means the score "
                "vectors were computed with no gradient graph, or the damping swamped the signal. "
                "Report the damping and confirm the solution is stable across a decade of it."
            ),
            statistics={"max_eigenvalue": top},
        )
    keep = eigs > rank_tol * top
    rank = int(keep.sum())
    coords = vecs.T @ u
    inside = float(np.sum(coords[keep] ** 2))
    outside = float(max(0.0, 1.0 - inside))

    e = float(u @ sym @ u)
    notes: tuple[str, ...]
    if outside > rank_tol:
        c = 0.0
        notes = (
            f"{100 * outside:.1f}% of beta's squared length lies outside the range of G "
            f"(rank {rank} of {k}), so that component of the selection pressure cannot be produced "
            f"by any parameter move and the conditional evolvability is exactly zero.",
        )
    else:
        quad = float(np.sum(coords[keep] ** 2 / eigs[keep]))
        c = 1.0 / quad if quad > 0 else float("inf")
        notes = ()
    a = c / e if e > 0 else float("nan")
    return EvolvabilityReading(
        evolvability=e,
        conditional_evolvability=c,
        autonomy=a,
        rank_G=rank,
        k=k,
        beta_outside_G=outside,
        conditioning=float(eigs.sum() / top),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rung 0: realised heritability from a record alone
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealisedHeritabilityReading:
    """`ĥ²` by regression of cumulative `Δz` on cumulative `η·S` across a window of steps.

    This is the multi-generation realised-heritability estimator from animal breeding, and it needs
    no backward passes at all: both series are already in the Price ledger. The univariate reduction
    is exact when `G` and `C` are diagonal, since `Δz_i = η G_ii β_i` and `S_i = C_ii β_i` give
    `Δz_i / (η S_i) = G_ii/C_ii` term for term. Off the diagonal the estimate absorbs the response
    dragged in through correlated features, which is a bias with no sign in general.

    ``in_bounds`` is per feature and it is the self-check `C ⪰ G` licenses: an estimate outside
    [0, 1] by more than twice its own standard error is not a heritability. It is reported rather
    than clipped, and the instrument refuses on it.
    """

    names: tuple[str, ...]
    h2: np.ndarray
    standard_error: np.ndarray
    cumulative_delta_z: np.ndarray
    cumulative_selection: np.ndarray
    selection_share: np.ndarray
    n_steps: int
    in_bounds: np.ndarray
    notes: tuple[str, ...] = ()

    @property
    def undefined(self) -> tuple[str, ...]:
        """Features with no estimate at all: no cumulative selection to regress against."""
        return tuple(n for n, v in zip(self.names, self.h2) if not np.isfinite(v))

    @property
    def out_of_bounds(self) -> tuple[str, ...]:
        """Features whose estimate exists and lies outside [0, 1] beyond twice its own error.

        Undefined is not out of bounds, and the distinction is the whole point. A feature the
        estimator could not touch has produced no claim to violate a bound, and naming it here
        would make a refusal about the three features that did produce a number read as a refusal
        about five. Pointing this at the 200-step record is what found the difference: two of its
        five surface features are constant, so their cumulative selection is identically zero.
        """
        return tuple(
            n for n, v, ok in zip(self.names, self.h2, self.in_bounds) if np.isfinite(v) and not ok
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "h2_realised": [None if not np.isfinite(v) else float(v) for v in self.h2],
            "standard_error": [
                None if not np.isfinite(v) else float(v) for v in self.standard_error
            ],
            "cumulative_delta_z": self.cumulative_delta_z.tolist(),
            "cumulative_selection": self.cumulative_selection.tolist(),
            "selection_share": self.selection_share.tolist(),
            "n_steps": self.n_steps,
            "in_bounds": [bool(v) for v in self.in_bounds],
            "out_of_bounds": list(self.out_of_bounds),
            "undefined": list(self.undefined),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [f"realised heritability over {self.n_steps} step pairs"]
        for i, n in enumerate(self.names):
            state = (
                "   undefined (no cumulative selection)"
                if not np.isfinite(self.h2[i])
                else ("" if self.in_bounds[i] else "   OUT OF [0, 1]")
            )
            lines.append(
                f"    {n:<20} h2 {self.h2[i]:+.5g} +/- {self.standard_error[i]:.3g}   "
                f"Σ|ηS|/Σ|Δz| {self.selection_share[i]:.3g}{state}"
            )
        return "\n".join(lines)


def ledger_arrays(ledgers: Sequence[Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """`(names, Δz, η·Cov)` from a sequence of `measure.ledger.price.StepLedger`, by duck typing.

    Read structurally rather than through an import so `measure/indices/` keeps no dependency on
    `measure/ledger/`. Any object with `.rows` whose entries carry `.feature`, `.delta_z` and
    `.selection` works, which is also what the labelled-table adapter produces.
    """
    if not ledgers:
        raise ValueError("no ledgers supplied; realised heritability needs a window of steps")
    names = tuple(row.feature for row in ledgers[0].rows)
    for led in ledgers:
        got = tuple(row.feature for row in led.rows)
        if got != names:
            raise ValueError(
                f"ledger at step {getattr(led, 'step', '?')} carries the basis {list(got)} and the "
                f"first carries {list(names)}. A cumulative regression across two bases is a "
                f"regression of one quantity on another."
            )
    dz = np.asarray([[row.delta_z for row in led.rows] for led in ledgers], dtype=np.float64)
    sel = np.asarray([[row.selection for row in led.rows] for led in ledgers], dtype=np.float64)
    return names, dz, sel


def realised_heritability(
    delta_z: np.ndarray,
    selection: np.ndarray,
    names: Sequence[str],
    *,
    instrument: str = "RealisedHeritability",
) -> RealisedHeritabilityReading | Refusal:
    """The rung-0 estimator: slope of cumulative `Δz` on cumulative `η·S`, through the origin.

    ``delta_z`` and ``selection`` are `(n_steps, k)`, one row per step pair, exactly the two columns
    the Price ledger already reports. The regression is through the origin because the breeder's
    equation has no intercept: zero cumulative selection predicts zero cumulative response, and
    fitting an intercept lets a drift with no selection behind it be absorbed into one.

    The standard error is the ordinary regression standard error of the slope. It is optimistic:
    consecutive cumulative sums are strongly autocorrelated by construction, so the effective number
    of independent observations is far below `n_steps`. It is reported as the scale of the
    uncertainty rather than as an interval, and the bound check uses two of them.
    """
    dz = np.asarray(delta_z, dtype=np.float64)
    sel = np.asarray(selection, dtype=np.float64)
    if dz.shape != sel.shape:
        raise ValueError(f"delta_z is {dz.shape} and selection is {sel.shape}; they must match")
    if dz.ndim != 2:
        raise ValueError("delta_z and selection must be (n_steps, k)")
    n_steps, k = dz.shape
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} feature columns")
    if n_steps < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"realised heritability regresses across steps and {n_steps} were supplied",
            remedy=(
                "widen the window to at least two step pairs. A single step gives a ratio of two "
                "numbers, not a regression, and the ratio has no uncertainty attached to it."
            ),
            statistics={"n_steps": n_steps},
        )

    cum_dz = np.cumsum(dz, axis=0)
    cum_sel = np.cumsum(sel, axis=0)
    h2 = np.full(k, np.nan)
    se = np.full(k, np.nan)
    share = np.full(k, np.nan)
    for i in range(k):
        x, y = cum_sel[:, i], cum_dz[:, i]
        denom = float(x @ x)
        moved = float(np.sum(np.abs(dz[:, i])))
        share[i] = float(np.sum(np.abs(sel[:, i]))) / moved if moved > 0 else np.nan
        if not denom > 0:
            continue
        slope = float(x @ y) / denom
        resid = y - slope * x
        dof = max(n_steps - 1, 1)
        se[i] = float(np.sqrt(max((resid @ resid) / dof, 0.0) / denom))
        h2[i] = slope

    in_bounds = np.asarray(
        [
            bool(np.isfinite(v) and (v + 2 * s) >= 0.0 and (v - 2 * s) <= 1.0)
            if np.isfinite(v) and np.isfinite(s)
            else False
            for v, s in zip(h2, se)
        ]
    )

    notes: list[str] = []
    flat = [n for n, v in zip(names, h2) if not np.isfinite(v)]
    if flat:
        notes.append(
            f"{len(flat)} features had no cumulative selection to regress against and carry NaN: "
            f"{', '.join(flat)}."
        )
    return RealisedHeritabilityReading(
        names=tuple(names),
        h2=h2,
        standard_error=se,
        cumulative_delta_z=cum_dz[-1],
        cumulative_selection=cum_sel[-1],
        selection_share=share,
        n_steps=n_steps,
        in_bounds=in_bounds,
        notes=tuple(notes),
    )


def refuse_out_of_bounds(
    reading: RealisedHeritabilityReading, *, instrument: str = "RealisedHeritability"
) -> Refusal | None:
    """The bound check as a refusal: `ĥ²` outside [0, 1] is not a heritability.

    `C ⪰ G` bounds `h²` in [0, 1] as a theorem, so an estimate of 12,130 is not a large heritability
    and it is not an outlier. It means the numerator and the denominator are not related by the
    breeder's equation on this record, which happens whenever the response is driven by something
    other than the selection differential. The diagnostic that says so is `Σ|η·S| / Σ|Δz|`, and it
    goes in the refusal because it is the number that tells the reader what to do next.
    """
    bad = reading.out_of_bounds
    if not bad:
        return None
    idx = [reading.names.index(n) for n in bad]
    worst = max(idx, key=lambda i: abs(reading.h2[i]) if np.isfinite(reading.h2[i]) else 0.0)
    share = float(np.nanmin(reading.selection_share[idx]))
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ENVELOPE_VIOLATED,
        detail=(
            f"realised h2 is outside [0, 1] on {len(bad)} of {len(reading.names)} features "
            f"({', '.join(bad)}); the largest is {reading.h2[worst]:.4g} on "
            f"{reading.names[worst]} against an upper bound of 1. C >= G bounds h2 in [0, 1], so "
            f"the numerator and the denominator are not related by the breeder's equation here: "
            f"the cumulative selection term is {share:.3g} of the cumulative motion"
        ),
        remedy=(
            "read the selection-explained fraction Λ for this window first. When Λ is near zero the "
            "response is not being driven by the selection differential and no ratio of the two "
            "estimates a heritability. The subject this estimator needs is a run whose consecutive "
            "steps share their prompts, so Δz is a difference between two policies rather than "
            "between two task samples, and whose step size is large enough that η·S is a material "
            "part of the motion. Failing that, compute G directly and read h2 = G_ii/C_ii at rung 2."
        ),
        statistics={
            "out_of_bounds": list(bad),
            "max_h2": float(reading.h2[worst]),
            "min_selection_share": share,
            "n_steps": reading.n_steps,
        },
    )


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


class _HeritabilityInstrument(BaseObservable):
    """Shared declarations for C2. Every reading here is a Fisher-metric quantity.

    Access is `POLICY: RECORD` at rung 0 and `POLICY: BACKWARD` at rung 2, which is exactly the
    difference between reading a training record somebody else produced and running a Fisher solve
    against their checkpoint.
    """

    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "feature heritability on the Fisher metric"
    substrates = ANY_SUBSTRATE
    phases = RECORD_PHASES
    envelope = HERITABILITY_ENVELOPE
    #: `policy.reparam`: `h²`, `e`, `c` and `a` are all built from `G = J F⁻¹ Jᵀ`, which is the
    #: pushforward of the Fisher metric onto the feature basis and does not move under a smooth
    #: reparameterisation of `θ`. `‖Δθ‖` would, and none of these is that.
    invariance = "policy.reparam"
    invariance_relation = INVARIANT

    def payload(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def compute(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return self.measure(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        got = self.compute()
        if isinstance(got, Refusal):
            return got  # type: ignore[return-value]
        payload = dict(got.as_dict())
        payload["note"] = INERT_BUT_SUSCEPTIBLE
        # `Context.emit` reads the instrument's name, version, gauge status and quantity off
        # `ctx._observable`, and `measure.base.run` is the only thing that sets it. Every
        # instrument here reads its inputs from its own constructor rather than from a signal, so
        # `estimate` calls `measure` directly and never goes through `run`. Without this the reading
        # would be emitted as `anonymous` with `quantity=""` and the unit machinery would have
        # nothing to key on. The previous value is restored rather than cleared so a nested call
        # does not lose its own identity.
        previous = ctx._observable
        ctx._observable = self  # type: ignore[assignment]
        try:
            return ctx.emit(payload, uncertainty=self.uncertainty(got))
        finally:
            ctx._observable = previous

    def uncertainty(self, reading: Any) -> Uncertainty:
        return Uncertainty(n=int(getattr(reading, "n_samples", 0) or 0), method="none")


class FeatureHeritability(_HeritabilityInstrument):
    """C2 rung 2: `h²_i = G_ii/C_ii`, the fraction of a feature's variance a parameter move reaches.

    Takes `G` and `C` in one basis and reports the per-feature ratio with the `C ⪰ G` bound checked
    both on the diagonal and on the whole of `N = C − G`. It computes neither matrix: `G` needs `k`
    Jacobian-vector products and `k` Fisher solves against the policy and `measure/efficiency/` is
    where that lives, `C` comes from the same rollouts through
    `measure.indices.chi.feature_covariance`, and the one thing this instrument insists on is that
    they are the same basis in the same order.

    What it cannot do. It says nothing about how large the reachable response is, only what fraction
    of the observed variance is reachable: `h²` is a ratio and `e(β)` is a scale, and Hansen and
    Pélabon wrote a paper called "Heritability is not Evolvability" about exactly this substitution.
    Read `Autonomy` beside it.
    """

    name = "FeatureHeritability"
    version = "1.0"
    quantity = "selection.heritability_h2"
    requires: AccessMatrix = {Component.POLICY: Access.BACKWARD}
    baselines = ("baseline.assume_h2_is_one", "baseline.marginal_correlation")
    rung = 2
    deviations = (
        "G is supplied rather than computed here; this instrument checks the bound and forms the "
        "ratio, and the Fisher solve that produces G lives in measure/efficiency/",
    )

    #: The bias of the ratio, and it is the damping that carries it. `(F + λI)⁻¹ ⪯ F⁻¹`, so a damped
    #: Fisher solve shrinks every quadratic form in `F⁻¹` and therefore shrinks `G = J F⁻¹ Jᵀ`. `C`
    #: is unaffected. So `h²` from a damped solve is biased **downward**, and the size of the bias
    #: is the thing the stability check across a decade of `λ` exists to bound.
    BIAS = BiasStatement(
        direction="downward",
        why=(
            "G is computed with a damped Fisher inverse, (F + lambda I)^-1, which is dominated by "
            "F^-1 in the positive semi-definite order. So every quadratic form in it shrinks, G "
            "shrinks with it, C does not move, and h2 comes out too small. The size of the "
            "shrinkage is what the stability sweep across a decade of lambda measures, and a "
            "reading whose h2 moves materially across that decade is reporting the damping rather "
            "than the parameterisation."
        ),
    )

    def __init__(
        self,
        metric_G: np.ndarray,
        feature_covariance: np.ndarray,
        names: Sequence[str],
        *,
        damping: float | None = None,
        damping_stable: bool | None = None,
        method: str = "fisher_solve",
        n_samples: int = 0,
    ) -> None:
        self.metric_G = np.asarray(metric_G, dtype=np.float64)
        self.feature_covariance = np.asarray(feature_covariance, dtype=np.float64)
        self.feature_names = tuple(names)
        self.damping = damping
        self.damping_stable = damping_stable
        self.method = method
        self.n_samples = int(n_samples)

    def compute(self) -> HeritabilityReading | Refusal:
        return heritability(
            self.metric_G,
            self.feature_covariance,
            self.feature_names,
            rung=self.rung,
            method=self.method,
            damping=self.damping,
            damping_stable=self.damping_stable,
            n_samples=self.n_samples,
            instrument=self.name,
        )


class Evolvability(_HeritabilityInstrument):
    """C2's scale: `e(β) = βᵀGβ`, how much response this selection direction can actually buy.

    The companion to `h²` and not a substitute for it. `h²` is dimensionless and bounded; this has
    the units of a feature squared and is unbounded above, and the two answer different questions.
    A basis where every `h²` is 0.9 and every `e` is 1e-12 is a parameterisation that can reach
    almost all of a variance that is itself negligible.
    """

    name = "Evolvability"
    version = "1.0"
    quantity = "selection.evolvability"
    requires: AccessMatrix = {Component.POLICY: Access.BACKWARD}
    baselines = ("baseline.marginal_correlation",)
    rung = 2

    def __init__(self, metric_G: np.ndarray, beta: np.ndarray, *, n_samples: int = 0) -> None:
        self.metric_G = np.asarray(metric_G, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        self.n_samples = int(n_samples)

    def compute(self) -> EvolvabilityReading | Refusal:
        return evolvability_indices(self.metric_G, self.beta, instrument=self.name)

    def uncertainty(self, reading: Any) -> Uncertainty:
        return Uncertainty(n=self.n_samples, method="none")


class Autonomy(Evolvability):
    """C2's collateral-damage forecast: `a(β) = c(β)/e(β) ∈ [0, 1]`, before any RL runs.

    `a(β) = 0.31` says 69% of the response to this selection pressure is movement in traits
    correlated with the one being pushed on rather than in the target itself. It needs `G` and a
    direction and nothing else, so it is answerable at the point where somebody is still choosing
    what to reward, which is the only point at which the answer is cheap to act on.
    """

    name = "Autonomy"
    version = "1.0"
    quantity = "selection.autonomy"
    rung = 2
    baselines = ("baseline.assume_full_autonomy",)


class GConditioning(_HeritabilityInstrument):
    """`n_D = Σλ/λ_max` on `G`: how many directions the parameterisation can actually reach.

    The same scalar C1 reports on `C`, computed on `G` instead, and the pair is the informative
    thing. `n_D(C) = 6` with `n_D(G) = 2` says the rollouts vary in six directions and parameter
    moves can produce two of them, which is `N` seen through its rank rather than through a diagonal.
    """

    name = "GConditioning"
    version = "1.0"
    quantity = "selection.G_conditioning"
    requires: AccessMatrix = {Component.POLICY: Access.BACKWARD}
    baselines = ("baseline.assume_full_rank",)
    rung = 1

    def __init__(self, metric_G: np.ndarray, *, n_samples: int = 0) -> None:
        self.metric_G = np.asarray(metric_G, dtype=np.float64)
        self.n_samples = int(n_samples)

    def compute(self) -> Any:
        eigs = np.linalg.eigvalsh((self.metric_G + self.metric_G.T) / 2.0)
        top = float(eigs.max()) if eigs.size else 0.0
        if not top > 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail="G has no positive eigenvalue, so its effective dimensionality is undefined",
                remedy=(
                    "check the Fisher solve before reading anything else off this G. A numerically "
                    "zero G usually means the score vectors carried no gradient graph."
                ),
                statistics={"max_eigenvalue": top},
            )
        return _Scalar(
            {
                "conditioning_n_D": float(eigs.sum() / top),
                "k": int(eigs.size),
                "eigenvalues": eigs.tolist(),
                "rank": int(np.sum(eigs > 1e-10 * top)),
            },
            self.n_samples,
        )


class RealisedHeritability(_HeritabilityInstrument):
    """C2 rung 0: `ĥ²` from a training record alone, with no backward passes at all.

    `ĥ² = (cumulative Δz) / (cumulative η·S)`, regressed across a window of steps. This is the
    standard multi-generation estimator from animal breeding and it works on the two columns the
    Price ledger already reports, which makes it the only rung of C2 reachable at `POLICY: RECORD`.

    What it cannot do, and it is the reason the envelope has one condition. The estimator assumes
    the response is driven by the selection differential, which is exactly what `LINEAR_RESPONSE`
    asserts and `selection.explained_fraction` measures. On a run where the two are unrelated the
    ratio is well defined, large, and meaningless: the bound check reports it as outside [0, 1] and
    the instrument refuses rather than handing back a heritability of ten thousand.
    """

    name = "RealisedHeritability"
    version = "1.0"
    quantity = "selection.heritability_h2"
    requires: AccessMatrix = {Component.POLICY: Access.RECORD, Component.RECORD: Access.RECORD}
    baselines = ("baseline.assume_h2_is_one", "baseline.length")
    rung = 0
    deviations = (
        "the univariate reduction Δz_i/(η S_i) = G_ii/C_ii is exact only when G and C are diagonal; "
        "off the diagonal the estimate absorbs response dragged in through correlated features",
        "the standard error is the ordinary regression one and is optimistic, because consecutive "
        "cumulative sums are autocorrelated by construction",
    )

    #: Direction unknown, and the reason is worth stating rather than guessing at. The off-diagonal
    #: term `Σ_{j≠i} G_ij β_j` enters the numerator with whatever sign the correlated features carry:
    #: a feature dragged up by a correlated one reads high, and one held back by a correlated
    #: opposite reads low or negative.
    BIAS = BiasStatement(
        direction="unknown",
        why=(
            "the univariate breeder's equation is exact only on a diagonal G and C. Off the "
            "diagonal the numerator carries the response dragged in through every correlated "
            "feature, with whatever sign those features were selected in, so a correlated pair can "
            "read heritable when neither is and a suppressed feature can read negative. Report it "
            "with the conditioning n_D of C, and prefer rung 2 wherever a Fisher solve is reachable."
        ),
    )

    def __init__(
        self,
        delta_z: np.ndarray,
        selection: np.ndarray,
        names: Sequence[str],
        *,
        enforce_bounds: bool = True,
    ) -> None:
        self.delta_z = np.asarray(delta_z, dtype=np.float64)
        self.selection = np.asarray(selection, dtype=np.float64)
        self.feature_names = tuple(names)
        self.enforce_bounds = bool(enforce_bounds)

    @classmethod
    def from_ledgers(cls, ledgers: Sequence[Any], **kwargs: Any) -> "RealisedHeritability":
        """Build straight from a `measure.ledger.price.ledger_series` result."""
        names, dz, sel = ledger_arrays(ledgers)
        return cls(dz, sel, names, **kwargs)

    def compute(self) -> RealisedHeritabilityReading | Refusal:
        got = realised_heritability(
            self.delta_z, self.selection, self.feature_names, instrument=self.name
        )
        if isinstance(got, Refusal) or not self.enforce_bounds:
            return got
        return refuse_out_of_bounds(got, instrument=self.name) or got

    def uncertainty(self, reading: Any) -> Uncertainty:
        return Uncertainty(n=int(getattr(reading, "n_steps", 0) or 0), method="regression")


@dataclass(frozen=True)
class _Scalar:
    """A one-field reading, so `GConditioning` shares the emit path without a bespoke type."""

    fields: Mapping[str, Any]
    n_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


__all__ = [
    "HERITABILITY_ENVELOPE",
    "INERT_BUT_SUSCEPTIBLE",
    "RECORD_PHASES",
    "Autonomy",
    "Evolvability",
    "EvolvabilityReading",
    "FeatureHeritability",
    "GConditioning",
    "HeritabilityReading",
    "RealisedHeritability",
    "RealisedHeritabilityReading",
    "evolvability_indices",
    "heritability",
    "ledger_arrays",
    "realised_heritability",
    "refuse_out_of_bounds",
]
