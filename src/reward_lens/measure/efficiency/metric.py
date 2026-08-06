"""`G = J F⁻¹ Jᵀ`: the behavioural covariance a parameter move can actually reach.

`G` is the object both the capacity book and the cost book are built on. With `J = ∂z/∂θ` the
Jacobian of the measured feature means and `F` the Fisher information, `G` answers "which patterns
of behavioural change can a parameter step produce, and at what price in nats". Two things follow
from it that this module is here to supply: heritability `h²_i = G_ii/C_ii`, and the minimum
information cost `KL_min(Δz) = ½ Δzᵀ G⁻¹ Δz` that `measure.efficiency.cost` divides into.

**Three estimators, and they are not interchangeable.**

`covariance_bound` (rung 0, record only). `C ⪰ G` holds for every feature basis: no
parameter perturbation can produce more feature variance than the rollouts already show. So
`G = C` is an upper bound on the metric and therefore makes `KL_min` a **lower bound** on the true
minimum cost, and a lower bound on `KL_min` is a lower bound on efficiency. It needs no gradients,
no checkpoint and no torch. It is the honest default and its readings say "at least this many nats"
rather than "this many nats".

`realised` (rung 0, record only). Lande's equation `Δz = η G β` regressed across a window, which is
the multi-generation realised-heritability estimator from animal breeding written in matrix form:
each row of `G` is the least-squares fit of one feature's movement on `η β` across steps. It needs
a run where selection actually explains movement. On a run where it does not, the fit is noise and
comes back non-PSD, and a non-PSD `G` is not a metric, so this refuses rather than projecting a
fiction onto the cone.

`fisher_kernel` (rung 2, POLICY:BACKWARD). The real construction, computed without ever forming
`F`. With `S` the matrix of per-rollout score vectors `∇_θ log π(y_a)` and `Φ` the feature matrix,
both centred within their prompt group, the push-through identity gives

    Ĝ = (1/m) Φᵀ [ K (K + mλI)⁻¹ ] Φ,   K = S Sᵀ,   m = n − n_groups

so the whole of `G` comes out of the `n × n` Gram matrix of scores. That costs `n` backward passes
and one `n × n` solve, against the `k` Jacobian-vector products plus `k` conjugate-gradient Fisher
solves the direct construction costs. It is also exactly, not approximately, consistent with
`C ⪰ G`: the eigenvalues of `K(K + mλI)⁻¹` lie in `[0, 1)`, so `Ĝ ⪯ (1/m)ΦᵀΦ = Ĉ` holds in finite
samples and `h² ∈ [0, 1]` is arithmetic rather than an assumption.

**What the rung-2 estimator cannot do, three lines in rather than on a caveats page.** With fewer
rollouts than parameters, which is every real policy, the *undamped* plug-in is not merely noisy,
it is degenerate: `Φ` lies entirely inside the row space of `S`, the projection is the identity on
it, and `Ĝ = Ĉ` **exactly**, so `h² = 1` for every feature no matter what is true. On the 200-step
fixture at λ→0 the three moving features come back at `h² = 0.99999`. The damping is therefore not
a numerical convenience, it is the entire content of the estimate, which is why the `λ` is reported
and why `damping_stable` is on every reading. A `damping_stable` of False means the number you are
holding is a function of a regularisation constant, and it should be read as the
`covariance_bound` with a shrinkage applied rather than as a measurement of `G`.

The second limit is memory. The kernel form holds `n · |θ|` floats: 78 MB for eight rollouts of a
2.45M-parameter policy, and 224 GB for eight rollouts of a 7B one. Above roughly a hundred million
parameters the score matrix has to be sketched or the `k` Fisher solves done directly, and neither
is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.measure.ledger.price import StepLedger, StepSample, selection_differential

#: Eigenvalues below this multiple of the largest are treated as structural zeros when `G` is
#: inverted. Chosen at 1e-10 rather than at numpy's default `rcond`, which scales with the matrix
#: dimension and would silently discard a genuinely small but real eigenvalue on a `k` of three.
#: A feature that is constant over the window sits exactly at zero and is meant to be discarded;
#: one whose eigenvalue is 1e-9 of the largest is expensive to move, not immovable, and the
#: difference matters because the first is a bookkeeping fact and the second is a finding.
RANK_TOLERANCE = 1e-10

#: The default relative damping for `fisher_kernel`, expressed as a fraction of `trace(K)/n` so it
#: is dimensionless and survives a rescaling of the parameterisation. **Chosen: 1e-2**, which on the
#: 200-step fixture leaves `h²` at 0.992 and puts the estimator visibly in the regime the module
#: docstring describes. It is a placeholder that behaves like a decision, and it is a default
#: rather than a fitted value: nothing here is tuned to make a result come out.
DEFAULT_DAMPING = 1e-2

#: How much `G` may move across a decade of `λ` and still be called stable. **Chosen: 0.05** on the
#: Frobenius norm, which is tight enough that the fixture fails it (as it should) and loose enough
#: that float noise does not.
DEFAULT_STABILITY_TOL = 0.05


def _within_group_centre(matrix: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Subtract each prompt group's own mean from its rows.

    Within-group and not pooled, for the reason that settles it for the ledger and which
    applies with more force here: `E_y[∇_θ log π(y|x)] = 0` holds **per prompt**, not across
    prompts. Centring pooled would leave the between-prompt mean score in `S`, and that is a
    property of the task distribution rather than of the policy.
    """
    out = np.asarray(matrix, dtype=np.float64).copy()
    for label in np.unique(group_ids):
        mask = group_ids == label
        out[mask] -= out[mask].mean(axis=0)
    return out


@dataclass(frozen=True)
class MetricG:
    """`G = J F⁻¹ Jᵀ`: the covariance a parameter move can actually reach.

    ``names`` is the feature basis and it is the join key between this, the ledger's `Δz` and the
    cost book's shares. It is `StepSample.names` in `StepSample.names` order, whole: a feature that
    is constant over the window keeps its column, which comes out as a structural zero in `matrix`
    and is handled by the pseudo-inverse rather than by dropping the name and breaking the join.

    ``damping`` is the absolute `λ` in `(F + λI)⁻¹` and ``damping_stable`` says whether the answer
    survives a decade of it. Both are fields rather than parameters because a `G` that hides its
    regularisation cannot be checked for the stability it also has to claim.

    ``covariance`` is `C`, the within-group feature covariance the same rollouts produce. It is not
    part of the interface `measure.reconcile` reads and it is here because two things need it:
    `h²  = G_ii/C_ii`, which is C2's quantity and not this package's, and the `C ⪰ G` self-check,
    which is the one arithmetic property that catches a sign or a denominator error in any of the
    three estimators.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    damping: float
    damping_stable: bool
    conditioning: float
    rung: int
    method: str
    n_samples: int
    #: Additive to the fixed interface, defaulted, invisible to a reader that uses only the fields
    #: above.
    covariance: np.ndarray | None = None
    n_groups: int = 0
    rank_tolerance: float = RANK_TOLERANCE
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        k = len(self.names)
        if matrix.shape != (k, k):
            raise ValueError(
                f"G is {matrix.shape} against {k} feature names. A metric whose shape does not "
                f"match its basis cannot be joined to a ledger row."
            )
        asymmetry = float(np.max(np.abs(matrix - matrix.T))) if k else 0.0
        scale = float(np.max(np.abs(matrix))) if k else 1.0
        if asymmetry > 1e-8 * max(scale, 1.0):
            raise ValueError(
                f"G is not symmetric: max |G - Gᵀ| is {asymmetry:.6g} against a scale of "
                f"{scale:.6g}. `G = J F⁻¹ Jᵀ` is symmetric by construction, so this is an "
                f"estimator bug and not a property of the data."
            )
        if k:
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            floor = -1e-8 * max(float(eigenvalues.max()), 1.0)
            if float(eigenvalues.min()) < floor:
                raise ValueError(
                    f"G has a negative eigenvalue at {float(eigenvalues.min()):.6g}, against a "
                    f"largest of {float(eigenvalues.max()):.6g}. `G` is positive semi-definite by "
                    f"construction, so a negative eigenvalue means the estimate is not a "
                    f"metric. `KL_min` computed against it would be negative for some `Δz`, which "
                    f"is a cost of less than nothing. Refuse instead, or raise the damping."
                )

    # -- the spectrum ------------------------------------------------------

    def eigen(self) -> tuple[np.ndarray, np.ndarray]:
        """`(g, U)` with `G = U diag(g) Uᵀ`, ascending. The basis the decomposition is exact in."""
        values, vectors = np.linalg.eigh(0.5 * (self.matrix + self.matrix.T))
        return np.clip(values, 0.0, None), vectors

    @property
    def rank(self) -> int:
        """Directions a parameter move can reach, at `rank_tolerance`."""
        values, _ = self.eigen()
        if values.size == 0 or values.max() <= 0.0:
            return 0
        return int(np.count_nonzero(values > self.rank_tolerance * values.max()))

    def heritability(self) -> dict[str, float]:
        """`h²_i = G_ii / C_ii` per feature, or an empty mapping when `C` was not carried.

        C2's quantity, computed here because `G` and `C` come out of one pass and separating them
        would mean estimating `C` twice from the same rollouts under two conventions. A feature
        with no spread over the window has no `h²` and is reported as NaN rather than as zero: a
        ratio of zero to zero is undefined, and zero would read as "cannot be moved".
        """
        if self.covariance is None:
            return {}
        diag_g = np.diag(np.asarray(self.matrix, dtype=np.float64))
        diag_c = np.diag(np.asarray(self.covariance, dtype=np.float64))
        return {
            name: float(diag_g[i] / diag_c[i]) if diag_c[i] > 0.0 else float("nan")
            for i, name in enumerate(self.names)
        }

    def single_feature_bound(self, feature: str, delta: float) -> float:
        """`δ² / (2 G_ii)`: the least it can cost to move one feature by `δ`.

        Infinite when `G_ii` is zero, which is the correct reading and not a failure: a feature the
        parameterisation cannot move is a feature no number of nats will move, and that is `h² = 0`
        seen from the cost side.
        """
        index = self.names.index(feature)
        g_ii = float(self.matrix[index, index])
        if g_ii <= 0.0:
            return float("inf")
        return float(delta * delta / (2.0 * g_ii))

    # -- the quadratic form ------------------------------------------------

    def _spectral_solve(self, dz: np.ndarray) -> tuple[float, float, np.ndarray]:
        """`(½ Δzᵀ G⁺ Δz, out-of-range fraction, Δz̃)` where `Δz̃ = Uᵀ Δz`.

        The pseudo-inverse is not a convenience here, it is the honest answer to a real case: a
        feature basis with a constant feature, or a `G` estimated from fewer rollouts than
        features, has a null space, and a movement inside that null space is a movement no
        parameter step could have produced. Rather than inverting through it and returning a large
        number, the null component is measured and returned as ``out_of_range``, and the caller
        decides whether a `KL_min` computed on the reachable part alone is worth having.
        """
        values, vectors = self.eigen()
        rotated = vectors.T @ np.asarray(dz, dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0, rotated
        cutoff = self.rank_tolerance * max(float(values.max()), 0.0)
        live = values > cutoff
        total = float(np.dot(rotated, rotated))
        out_of_range = float(np.dot(rotated[~live], rotated[~live]) / total) if total > 0.0 else 0.0
        if not np.any(live):
            return 0.0, out_of_range, rotated
        value = 0.5 * float(np.sum(rotated[live] ** 2 / values[live]))
        return value, out_of_range, rotated

    def kl_min(self, dz: np.ndarray) -> tuple[float, float]:
        """`(½ Δzᵀ G⁻¹ Δz, out-of-range fraction)`. The exact minimum cost, in nats."""
        value, out_of_range, _ = self._spectral_solve(dz)
        return value, out_of_range

    def eigen_shares(self, dz: np.ndarray) -> np.ndarray:
        """`½ (Δz̃_j)² / g_j` per eigen-direction. Exactly additive, and the directions are unnamed.

        This is the exact decomposition and it is the only one that needs no allocation
        rule. `measure.efficiency.cost` turns it into shares of *named* features, which is a
        different and harder problem because named features are correlated.
        """
        values, vectors = self.eigen()
        rotated = vectors.T @ np.asarray(dz, dtype=np.float64)
        cutoff = self.rank_tolerance * max(float(values.max()), 0.0) if values.size else 0.0
        out = np.zeros_like(values)
        live = values > cutoff
        out[live] = 0.5 * rotated[live] ** 2 / values[live]
        return out

    def submatrix(self, indices: Sequence[int]) -> "MetricG":
        """`G_TT` on a subset of the basis, the value of the coalition `T` in the cost game."""
        idx = np.asarray(list(indices), dtype=int)
        sub = np.asarray(self.matrix, dtype=np.float64)[np.ix_(idx, idx)]
        return MetricG(
            names=tuple(self.names[i] for i in idx),
            matrix=sub,
            damping=self.damping,
            damping_stable=self.damping_stable,
            conditioning=self.conditioning,
            rung=self.rung,
            method=self.method,
            n_samples=self.n_samples,
            rank_tolerance=self.rank_tolerance,
        )

    def render(self) -> str:
        values, _ = self.eigen()
        stability = "stable" if self.damping_stable else "NOT stable across a decade of lambda"
        head = (
            f"G[{self.method}] rung {self.rung} over {self.n_samples} rollouts in "
            f"{self.n_groups} groups; lambda {self.damping:.4g} ({stability}); "
            f"n_D {self.conditioning:.3f}, rank {self.rank} of {len(self.names)}"
        )
        spectrum = "    eigenvalues " + ", ".join(f"{v:.5g}" for v in values[::-1])
        lines = [head, spectrum]
        h2 = self.heritability()
        if h2:
            lines.append("    h^2 " + ", ".join(f"{n} {h2[n]:.4f}" for n in self.names))
        lines.extend("    note: " + n for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pooling a window into one estimation sample
# ---------------------------------------------------------------------------


def pooled_rollouts(samples: Sequence[StepSample]) -> tuple[np.ndarray, np.ndarray, int]:
    """Every rollout in the window as one `(n, k)` matrix, with globally distinct group labels.

    The group labels are re-issued per step. `StepSample.group_ids` counts groups from zero inside
    its own step, so stacking two steps without relabelling would centre step 5's first prompt
    group against step 6's first prompt group, which are different prompts. That is the kind of
    error that produces a plausible number.

    Returns `(features, group_ids, n_groups)`.
    """
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0
    for sample in samples:
        if sample.n == 0:
            continue
        blocks.append(np.asarray(sample.features, dtype=np.float64))
        local = np.asarray(sample.group_ids, dtype=np.int64)
        remap = {g: i for i, g in enumerate(np.unique(local))}
        labels.append(np.asarray([remap[g] + offset for g in local], dtype=np.int64))
        offset += len(remap)
    if not blocks:
        k = len(samples[0].names) if samples else 0
        return np.zeros((0, k)), np.zeros(0, dtype=np.int64), 0
    return np.vstack(blocks), np.concatenate(labels), offset


def _conditioning(matrix: np.ndarray) -> float:
    """`n_D = Σ eig / max eig`: the effective dimension, which is a stable rank and not a rank."""
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    values = np.clip(values, 0.0, None)
    top = float(values.max()) if values.size else 0.0
    return float(values.sum() / top) if top > 0.0 else 0.0


# ---------------------------------------------------------------------------
# The three estimators
# ---------------------------------------------------------------------------


def _kernel_g(
    centred_features: np.ndarray,
    centred_scores: np.ndarray,
    dof: int,
    absolute_damping: float,
) -> np.ndarray:
    """`(1/m) Φᵀ K (K + mλI)⁻¹ Φ` with `K = S Sᵀ`. Never forms `F`.

    The push-through identity `S(SᵀS + cI)⁻¹Sᵀ = SSᵀ(SSᵀ + cI)⁻¹` is what moves the solve from
    `|θ| × |θ|` to `n × n`. It is exact, not an approximation, so the only error in `G` beyond the
    finite sample is the damping the caller asked for and the reading reports.
    """
    gram = centred_scores @ centred_scores.T
    n = gram.shape[0]
    shrunk = gram @ np.linalg.inv(gram + dof * absolute_damping * np.eye(n))
    matrix = centred_features.T @ shrunk @ centred_features / dof
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _realised_g(
    samples: Sequence[StepSample],
    ledgers: Sequence[StepLedger],
    names: Sequence[str],
) -> tuple[np.ndarray, int, str]:
    """Lande's equation regressed across a window: `Δz_t = η_t G β_t`, one row of `G` per feature.

    The rung-0 estimator, in matrix form. `β = C⁻¹S` is the selection *gradient*, so this
    regresses what moved on what selection pushed for after the correlation between features has
    been divided out, which is the whole difference between the differential and the gradient.

    Returns `(G, n_points, detail)`. The caller checks positive semi-definiteness, because on a run
    where selection explains nothing the least-squares fit is a fit to noise and comes back
    indefinite, and an indefinite `G` is not a metric.
    """
    by_index = {sample.index: sample for sample in samples}
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    skipped = 0
    for ledger in ledgers:
        sample = by_index.get(ledger.step)
        if sample is None or sample.n == 0:
            skipped += 1
            continue
        centred = _within_group_centre(sample.features, sample.group_ids)
        dof = sample.n - len(np.unique(sample.group_ids))
        if dof < 1:
            skipped += 1
            continue
        covariance = centred.T @ centred / dof
        differential = selection_differential(
            sample.features, sample.advantages, sample.group_ids, sample.names
        )
        if not np.all(np.isfinite(differential.value)):
            skipped += 1
            continue
        beta = np.linalg.pinv(covariance) @ differential.value
        delta_z = np.asarray([row.delta_z for row in ledger.rows], dtype=np.float64)
        if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(delta_z))):
            skipped += 1
            continue
        rows_x.append(ledger.eta * beta)
        rows_y.append(delta_z)
    if len(rows_x) < len(names) + 1:
        return np.zeros((len(names), len(names))), len(rows_x), "too few usable step pairs"
    design = np.vstack(rows_x)
    response = np.vstack(rows_y)
    solution, *_ = np.linalg.lstsq(design, response, rcond=None)
    matrix = 0.5 * (solution.T + solution)
    detail = f"{len(rows_x)} step pairs regressed, {skipped} skipped"
    return matrix, len(rows_x), detail


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def metric_g(
    samples: Sequence[StepSample],
    *,
    method: str = "covariance_bound",
    scores: np.ndarray | None = None,
    ledgers: Sequence[StepLedger] | None = None,
    damping: float = DEFAULT_DAMPING,
    stability_tol: float = DEFAULT_STABILITY_TOL,
    instrument: str = "metric_g",
) -> MetricG | Refusal:
    """`G` over a window of steps, by one of the three estimators, or the refusal that says why not.

    ``samples`` is a window of `StepSample`, which fixes the basis: `G.names` comes out as
    `samples[0].names`, whole and in order, which is the join key across this package.

    ``method`` is ``"covariance_bound"`` (rung 0, `G = C`, so `KL_min` is a lower bound),
    ``"realised"`` (rung 0, Lande regressed across the window, needs ``ledgers``), or
    ``"fisher_kernel"`` (rung 2, needs ``scores``: an `(n, |θ|)` array of per-rollout score vectors
    `∇_θ log π(y_a)` in the same row order as `pooled_rollouts` returns, which
    `measure.efficiency.scores.sequence_scores` produces).

    ``damping`` is relative, in units of `trace(K)/n`, so it is dimensionless; the absolute `λ` it
    resolves to is what lands on the reading.

    Pooling across a window is a choice with a cost and it is named on every reading that makes it.
    Eight rollouts at one step of the fixture leave six within-group degrees of freedom, which is a
    rank-deficient covariance for more than six features and a noisy one for three. Pooling buys
    degrees of freedom and pays for them by treating the policy as fixed across the window, which
    it is not.
    """
    if not samples:
        return refuse_incomplete(
            instrument,
            field="a step to estimate G from",
            subject="an empty window",
            remedy=(
                "Pass at least one `StepSample`. `steps_from_run(run, featuriser, window=...)` "
                "builds them; check the window against `whole_run(run)` if it came back empty."
            ),
        )
    names = tuple(samples[0].names)
    for sample in samples:
        if tuple(sample.names) != names:
            raise ValueError(
                f"step {sample.index} carries basis {list(sample.names)} against "
                f"{list(names)} at step {samples[0].index}. `G`, `Δz` and the cost book's shares "
                f"are vectors in one basis in one order, and a window that changes basis "
                f"half way through has no single `G` to estimate."
            )

    features, group_ids, n_groups = pooled_rollouts(samples)
    n = int(features.shape[0])
    dof = n - n_groups
    if dof < 1:
        return refuse_incomplete(
            instrument,
            field="a prompt group with more than one scored rollout",
            subject=(
                f"{len(samples)} step(s) holding {n} rollouts in {n_groups} groups, which leaves "
                f"{dof} within-group degrees of freedom"
            ),
            remedy=(
                "Widen the window, or read the run at a group size above one. A within-group "
                "covariance needs spread inside a prompt group, and a group of one rollout has "
                "none: there is nothing for the advantage to covary with."
            ),
            n_rollouts=n,
            n_groups=n_groups,
        )

    centred_features = _within_group_centre(features, group_ids)
    covariance = centred_features.T @ centred_features / dof
    covariance = 0.5 * (covariance + covariance.T)
    notes: list[str] = []
    if len(samples) > 1:
        notes.append(
            f"pooled across {len(samples)} steps ({n} rollouts, {n_groups} prompt groups, {dof} "
            f"within-group degrees of freedom). `G` is a property of one parameter point and this "
            f"treats the policy as fixed over the window."
        )
    constant = [names[i] for i in range(len(names)) if covariance[i, i] <= 0.0]
    if constant:
        notes.append(
            f"{', '.join(constant)} had no spread over this window, so the column is a structural "
            f"zero in `G`. It keeps its place in the basis and is handled by the pseudo-inverse; "
            f"nothing can move a feature that does not vary."
        )

    if method == "covariance_bound":
        matrix = covariance
        absolute_damping = 0.0
        stable = True
        notes.append(
            "`G = C`, which is the bound `C ⪰ G` taken at equality. Every `KL_min` computed "
            "against it is a **lower** bound on the true minimum cost, and every efficiency a "
            "lower bound on the true efficiency."
        )
        rung = 0
    elif method == "realised":
        if not ledgers:
            return refuse_incomplete(
                instrument,
                field="ledger rows to regress",
                subject=f"a `realised` G over {len(samples)} step(s)",
                remedy=(
                    "Pass `ledgers=ledger_series(samples, eta_by_step=learning_rates(run))`. The "
                    "realised estimator regresses `Δz` on `η·β` across steps and has nothing to "
                    "regress without the movement half."
                ),
            )
        matrix, n_points, detail = _realised_g(samples, ledgers, names)
        absolute_damping = 0.0
        stable = True
        rung = 0
        eigenvalues = np.linalg.eigvalsh(matrix) if len(names) else np.zeros(0)
        floor = -1e-8 * max(float(np.abs(eigenvalues).max()), 1.0) if eigenvalues.size else 0.0
        if n_points < len(names) + 1:
            return refuse_incomplete(
                instrument,
                field="enough step pairs to regress a k-by-k G",
                subject=f"{n_points} usable pair(s) for {len(names)} features ({detail})",
                remedy=(
                    f"Widen the window to at least {len(names) + 1} consecutive step pairs with a "
                    f"learning rate on each, or estimate `G` with `method='covariance_bound'`, "
                    f"which needs no regression at all."
                ),
                n_points=n_points,
                n_features=len(names),
            )
        if eigenvalues.size and float(eigenvalues.min()) < floor:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=(
                    f"the realised fit of `Δz` on `η·β` over {n_points} step pairs is not positive "
                    f"semi-definite: its smallest eigenvalue is {float(eigenvalues.min()):.6g} "
                    f"against a largest of {float(eigenvalues.max()):.6g}. A `G` with a negative "
                    f"eigenvalue is not a metric, and `KL_min` against it would be negative for "
                    f"some `Δz`."
                ),
                remedy=(
                    "This is what a low `Λ` looks like from the capacity side: selection explains "
                    "too little of what moved for a regression on it to identify `G`. Read "
                    "`SelectionExplainedFraction` first, and if `Λ` is low use "
                    "`method='covariance_bound'`, which needs no regression, or "
                    "`method='fisher_kernel'` with a policy checkpoint. Projecting this fit onto "
                    "the PSD cone would return a metric shaped like the noise that produced it."
                ),
                statistics={
                    "min_eigenvalue": float(eigenvalues.min()),
                    "max_eigenvalue": float(eigenvalues.max()),
                    "n_points": n_points,
                },
            )
        notes.append(f"realised (Lande) fit: {detail}.")
    elif method == "fisher_kernel":
        if scores is None:
            return refuse_incomplete(
                instrument,
                field="per-rollout score vectors",
                subject=f"a rung-2 `G` over {n} rollouts",
                remedy=(
                    "Pass `scores=`, an (n, |theta|) array of `grad_theta log pi(y_a)` in the row "
                    "order `pooled_rollouts(samples)` returns. "
                    "`measure.efficiency.scores.sequence_scores` computes them from a loaded "
                    "policy. With no checkpoint to differentiate, `method='covariance_bound'` is "
                    "the rung-0 answer and it is a bound rather than a guess."
                ),
            )
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.shape[0] != n:
            raise ValueError(
                f"scores has {score_array.shape[0]} rows against {n} pooled rollouts. The score "
                f"matrix must be in `pooled_rollouts` row order, or `G` pairs one rollout's "
                f"gradient with another's features."
            )
        centred_scores = _within_group_centre(score_array, group_ids)
        gram_trace = float(np.trace(centred_scores @ centred_scores.T))
        absolute_damping = float(damping * gram_trace / max(n, 1) / max(dof, 1))
        matrix = _kernel_g(centred_features, centred_scores, dof, absolute_damping)
        low = _kernel_g(centred_features, centred_scores, dof, absolute_damping * 10.0**-0.5)
        high = _kernel_g(centred_features, centred_scores, dof, absolute_damping * 10.0**0.5)
        reference = float(np.linalg.norm(matrix, "fro")) or 1.0
        drift = max(
            float(np.linalg.norm(matrix - low, "fro")),
            float(np.linalg.norm(matrix - high, "fro")),
        )
        stable = bool(drift / reference <= stability_tol)
        rung = 2
        notes.append(
            f"across one decade of lambda ({absolute_damping * 10.0**-0.5:.4g} to "
            f"{absolute_damping * 10.0**0.5:.4g}) the Frobenius norm of `G` moves by "
            f"{drift / reference:.1%}, against a stability tolerance of {stability_tol:.0%}."
        )
        if n <= 8:
            notes.append(
                f"{n} rollouts against a parameter vector of length {score_array.shape[1]}. The "
                f"undamped plug-in is exactly `C` in this regime, so the damping carries the "
                f"estimate: read `damping_stable` before reading `h²`."
            )
    else:
        raise ValueError(
            f"unknown method {method!r}. The three are 'covariance_bound' (rung 0, record only), "
            f"'realised' (rung 0, Lande regressed across the window) and 'fisher_kernel' "
            f"(rung 2, needs score vectors)."
        )

    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T)) if len(names) else np.zeros(0)
    if eigenvalues.size and float(eigenvalues.min()) < -1e-10 * max(float(eigenvalues.max()), 1.0):
        matrix = matrix + np.eye(len(names)) * (-float(eigenvalues.min()) + 1e-15)
        notes.append(
            f"the estimate had a smallest eigenvalue of {float(eigenvalues.min()):.3g}, which is "
            f"float round-off at this scale and was lifted to zero. A negative eigenvalue larger "
            f"than round-off is refused rather than lifted."
        )

    # `C ⪰ G` is arithmetic for both of the estimators that go through the covariance:
    # `covariance_bound` takes it at equality, and `fisher_kernel` multiplies `C` by a matrix whose
    # eigenvalues lie in [0, 1). So a violation is a bug in this module rather than a property of
    # the data, and it is checked here rather than left to a test that might not be run. The
    # realised estimator is exempt: it is a regression and nothing constrains its fit to sit under
    # `C`, which is one more reason its readings are the weakest of the three.
    if method != "realised" and len(names):
        gap = np.linalg.eigvalsh(covariance - 0.5 * (matrix + matrix.T))
        floor = -1e-7 * max(float(np.max(np.abs(covariance))), 1.0)
        if float(gap.min()) < floor:
            raise ValueError(
                f"`C - G` has eigenvalue {float(gap.min()):.6g}, so this estimate violates "
                f"the bound `C ⪰ G`. No parameter perturbation can produce more feature variance "
                f"than the rollouts already show, so `G` above `C` is an arithmetic error in "
                f"`{method}` (a denominator, a centring, or a sign), not a finding."
            )

    return MetricG(
        names=names,
        matrix=0.5 * (matrix + matrix.T),
        damping=absolute_damping,
        damping_stable=stable,
        conditioning=_conditioning(matrix),
        rung=rung,
        method=method,
        n_samples=n,
        covariance=covariance,
        n_groups=n_groups,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_DAMPING",
    "DEFAULT_STABILITY_TOL",
    "RANK_TOLERANCE",
    "MetricG",
    "metric_g",
    "pooled_rollouts",
]
