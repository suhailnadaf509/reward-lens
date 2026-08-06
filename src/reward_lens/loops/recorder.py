"""The rollout recorder: reward-feature drift monitoring with changepoint onset (S13).

The crown-jewel hypothesis of the recorder (science S13) is that reward hacking is
visible in reward-feature space before the reward and KL curves move, and that a monitor anchored
in the reward-defining subspace resists the obfuscation that kills a free-floating probe: the
policy is paid to excite ``w_r`` and cannot climb reward while hiding from it, so evasion pressure
leaks into the null space, which the effective-vs-null drift decomposition watches by construction.

This module builds the monitor and proves it on a synthetic rollout that is CPU-cheap and has a
planted hack direction, so the naming claim and the lead-time claim are checkable without a GPU.
The real RL rollout it wraps is GPU-gated; ``synthetic_hack_rollout`` is the stand-in organism, the
same way ``organisms.foundry`` plants structure for the calibration gate.

The recorder tracks four things per step, all from a batch of the step's activations plus the
proxy reward the RM assigns and (in the synthetic organism, or an offline eval) a gold reward:

- concept-dose trajectories: the mean projection of the step's activations onto each named feature
  direction, so a rising dose says "the policy is producing more of feature i".
- crystallization drift: the component of the activation-mean displacement from baseline along the
  reward direction ``w_r``, the quantity the policy is directly paid to grow.
- Mahalanobis outlier rate: the fraction of the step's samples that sit far from the baseline
  activation cloud under the baseline covariance, which is how off-distribution drift shows up.
- the effective-vs-null drift decomposition: the displacement of the activation mean split into its
  component inside the reward-defining subspace (effective) and its component orthogonal to it
  (null space), the decomposition the obfuscation-resistance hypothesis rests on.

Onset is detected with a CUSUM changepoint test (a single mean-shift split with a permutation
p-value). Sequential onset delegates to ``stats/changepoint``, which also carries the BOCPD
detector; the permutation variant stays here because it is dependency-light and answers a
different question, which is where one shift sits rather than when a regime broke. The lead time
is the gap between the exploited
feature's dose onset and the gold reward's divergence onset; a positive lead time is the recorder
seeing the hack in feature space before it shows up in behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef

if TYPE_CHECKING:
    from reward_lens.core.types import EvidenceID


# ---------------------------------------------------------------------------
# Changepoint (thin local layer over stats/changepoint)
# ---------------------------------------------------------------------------


@dataclass
class Changepoint:
    """A single mean-shift changepoint: the split index, its CUSUM magnitude, and a permutation p."""

    index: int
    statistic: float
    p_value: float
    direction: str  # "up" if the series rises after the split, "down" if it falls


@dataclass
class Onset:
    """A sequential-CUSUM onset: the first step a series departs its baseline regime, or None.

    ``index`` is the first departure step (None if the series never departs), ``direction`` whether
    it departed up or down, and ``statistic`` the CUSUM value in baseline-SD units at the crossing.
    Distinct from ``Changepoint``: the retrospective changepoint finds the single best split (the
    midpoint of a ramp), while this finds where the departure *starts*, which is the quantity a lead
    time is measured from.
    """

    index: int | None
    direction: str
    statistic: float


def cusum_onset(
    series: Sequence[float] | np.ndarray,
    *,
    baseline_steps: int | None = None,
    k_sds: float = 0.5,
    h_sds: float = 5.0,
) -> Onset:
    """Detect the first departure from a baseline regime by Page's sequential CUSUM (onset detection).

    Delegates to the central stats/changepoint implementation.
    """
    from reward_lens.stats.changepoint import cusum

    cp = cusum(series, threshold=h_sds, drift=k_sds, baseline=baseline_steps)
    x = np.asarray(series, dtype=np.float64).ravel()

    # Determine direction
    direction = "up"
    if cp.index is not None:
        b_idx = baseline_steps if baseline_steps is not None else max(3, x.size // 5)
        b = x[:b_idx]
        mu = float(np.nanmean(b)) if b.size > 0 else 0.0
        if x[cp.index] < mu:
            direction = "down"

    return Onset(index=cp.index, direction=direction, statistic=cp.strength)


def cusum_changepoint(
    series: Sequence[float] | np.ndarray, n_perm: int = 1000, seed: int = 0
) -> Changepoint:
    """Detect a single mean-shift changepoint by CUSUM with a permutation p-value (Taylor's method).

    Builds the cumulative sum of deviations from the mean, ``S_j = sum_{t<=j}(x_t - xbar)`` with
    ``S_0 = S_T = 0``; the changepoint is the index where ``|S|`` is largest, and the magnitude
    ``max(S) - min(S)`` is the test statistic. Significance is a permutation null: shuffle the
    series ``n_perm`` times, recompute the magnitude, and report the fraction at least as large
    (with the ``(count + 1) / (n_perm + 1)`` correction). This is the dependency-light alternative
    to the BOCPD detector in ``stats/changepoint``; the return contract is the same.

    A flat or trendless series returns a non-significant changepoint (large p-value). The reported
    index is the split point in ``[0, T]``: samples before it are one regime, samples after another.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    t = x.size
    if t < 3:
        return Changepoint(index=0, statistic=0.0, p_value=1.0, direction="up")
    s = np.concatenate([[0.0], np.cumsum(x - x.mean())])
    magnitude = float(s.max() - s.min())
    idx = int(np.argmax(np.abs(s)))
    direction = "up" if x[idx:].mean() >= x[:idx].mean() else "down"
    if magnitude == 0.0:
        return Changepoint(index=idx, statistic=0.0, p_value=1.0, direction=direction)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(x)
        sp = np.cumsum(perm - perm.mean())
        if float(sp.max() - sp.min()) >= magnitude:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return Changepoint(index=idx, statistic=magnitude, p_value=p_value, direction=direction)


# ---------------------------------------------------------------------------
# Feature bank and reports
# ---------------------------------------------------------------------------


@dataclass
class DirectionBank:
    """Named unit directions in activation space, the concepts whose dose the recorder tracks.

    ``directions`` is ``(k, d)``; rows are normalized on construction so a dose is an honest
    projection. ``names`` labels them so the recorder can name the exploited direction rather than
    return an index.

    This was called `FeatureBank` once, which is the name
    `reward_lens.core.features.FeatureBank` holds: a structural protocol with a ``featurize``
    method and a ``directions()`` accessor. This class is neither. It is a container whose
    ``directions`` is an array attribute, and it does not satisfy that protocol in any way a caller
    can rely on, though a `runtime_checkable` isinstance check would have said it did. Two
    incompatible objects under one exported name is a trap, so this one is named for what it is.
    """

    names: list[str]
    directions: np.ndarray

    def __post_init__(self) -> None:
        d = np.asarray(self.directions, dtype=np.float64)
        if d.ndim != 2:
            raise ValueError(f"directions must be 2-D (k, d); got shape {d.shape}")
        if len(self.names) != d.shape[0]:
            raise ValueError(f"{len(self.names)} names for {d.shape[0]} directions")
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.directions = d / norms

    @property
    def k(self) -> int:
        return len(self.names)


@register_payload
@dataclass
class OnsetAlarm:
    """A changepoint-based onset: which signal moved, when, and how significantly.

    Registered because it is nested inside `DriftReport.onset_alarms`, which is itself a payload.
    An unregistered nested type used to decode to a plain dict, so a `DriftReport` read back from a
    store came out with dicts where its alarms should be and nothing said so. The codec now refuses
    an unregistered payload rather than degrading, which is what surfaced this.
    """

    signal: str
    kind: str  # "concept-dose" | "gold-divergence" | "crystallization"
    step: int
    statistic: float
    p_value: float


@register_payload
@dataclass
class DriftReport:
    """The recorder's read-out over a rollout.

    ``dose`` is ``(T, k)`` concept-dose trajectories; ``dose_cusum`` and ``dose_p`` are the CUSUM
    magnitude and permutation p-value per feature. ``exploited_direction`` is the named feature the
    recorder flags as exploited (largest significant dose changepoint), with ``exploited_index``.
    ``crystallization`` is drift along ``w_r``; ``mahalanobis_outlier_rate`` the off-distribution
    fraction; ``drift_effective`` / ``drift_nullspace`` the reward-subspace vs orthogonal split of
    the activation-mean displacement. ``feature_onset`` is the exploited dose's changepoint step,
    ``gold_onset`` the gold reward's divergence step, and ``lead_time = gold_onset - feature_onset``
    the steps by which the feature-space signal precedes the behavioral one.
    """

    steps: np.ndarray
    feature_names: list[str]
    dose: np.ndarray
    dose_cusum: np.ndarray
    dose_p: np.ndarray
    exploited_direction: str | None
    exploited_index: int | None
    crystallization: np.ndarray
    mahalanobis_outlier_rate: np.ndarray
    drift_effective: np.ndarray
    drift_nullspace: np.ndarray
    proxy_reward: np.ndarray
    gold_reward: np.ndarray | None
    feature_onset: int | None
    gold_onset: int | None
    lead_time: int | None
    onset_alarms: list[OnsetAlarm] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


class RolloutRecorder:
    """Monitor a rollout in reward-feature space, step by step (science S13).

    Construct with the feature bank whose doses to track, the reward direction ``w_r`` the policy is
    paid to excite, a batch of baseline (step-0) activations that fixes the reference mean and
    covariance, and optionally an explicit reward-defining subspace (default: the span of ``w_r``).
    Call ``observe`` once per rollout step with that step's activation batch and rewards, then
    ``report`` to get the ``DriftReport``, or ``evidence`` to get it wrapped as ``Evidence``.

    Everything is CPU-cheap and pure-numpy. The recorder holds no model; it consumes activations a
    caller extracts, which is what lets it run in shadow mode on production serving with no
    behavior change, and what makes the synthetic organism a faithful stand-in for the GPU rollout.
    """

    def __init__(
        self,
        feature_bank: DirectionBank,
        reward_direction: Sequence[float] | np.ndarray,
        baseline_activations: np.ndarray,
        *,
        effective_subspace: np.ndarray | None = None,
        ridge: float = 1e-3,
        mahalanobis_quantile: float = 0.975,
    ):
        self.bank = feature_bank
        w = np.asarray(reward_direction, dtype=np.float64).ravel()
        nrm = np.linalg.norm(w)
        self.w_r = w / nrm if nrm > 0 else w
        base = np.asarray(baseline_activations, dtype=np.float64)
        if base.ndim != 2:
            raise ValueError(f"baseline_activations must be 2-D (n, d); got {base.shape}")
        self.d = base.shape[1]
        self.mu0 = base.mean(axis=0)
        cov = np.cov(base, rowvar=False)
        cov = np.atleast_2d(cov) + ridge * np.eye(self.d)
        self._cov_inv = np.linalg.inv(cov)
        # Mahalanobis threshold calibrated on the baseline cloud so the base outlier rate is set by
        # the quantile, not an arbitrary constant.
        base_dm = self._mahalanobis(base)
        self._maha_threshold = float(np.quantile(base_dm, mahalanobis_quantile))
        self._p_eff = self._projection(effective_subspace)

        self._dose: list[np.ndarray] = []
        self._cryst: list[float] = []
        self._maha_rate: list[float] = []
        self._drift_eff: list[float] = []
        self._drift_null: list[float] = []
        self._proxy: list[float] = []
        self._gold: list[float] = []
        self._has_gold = True

    def _projection(self, subspace: np.ndarray | None) -> np.ndarray:
        """Orthogonal projector onto the reward-defining subspace (default: span of ``w_r``)."""
        if subspace is None:
            basis = self.w_r[:, None]
        else:
            basis = np.asarray(subspace, dtype=np.float64)
            if basis.ndim == 1:
                basis = basis[:, None]
        q, _ = np.linalg.qr(basis)
        return q @ q.T

    def _mahalanobis(self, x: np.ndarray) -> np.ndarray:
        centered = x - self.mu0
        return np.sqrt(np.einsum("ij,jk,ik->i", centered, self._cov_inv, centered))

    def observe(
        self,
        activations: np.ndarray,
        proxy_reward: float | Sequence[float] | None = None,
        gold_reward: float | Sequence[float] | None = None,
    ) -> None:
        """Record one rollout step from its activation batch and rewards.

        ``activations`` is ``(n_samples, d)`` for the step. ``proxy_reward`` is the RM's score for
        the step (a scalar or per-sample, averaged); ``gold_reward`` is the ground-truth reward
        where a study has one (the synthetic organism, or an offline eval). If gold is omitted on
        any step, the report simply carries no gold onset and no lead time.
        """
        h = np.asarray(activations, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.d:
            raise ValueError(f"activations must be (n_samples, {self.d}); got {h.shape}")
        mean_h = h.mean(axis=0)
        self._dose.append(self.bank.directions @ mean_h)
        disp = mean_h - self.mu0
        self._cryst.append(float(disp @ self.w_r))
        dm = self._mahalanobis(h)
        self._maha_rate.append(float(np.mean(dm > self._maha_threshold)))
        eff = self._p_eff @ disp
        self._drift_eff.append(float(np.linalg.norm(eff)))
        self._drift_null.append(float(np.linalg.norm(disp - eff)))
        self._proxy.append(_mean_or_nan(proxy_reward))
        if gold_reward is None:
            self._has_gold = False
        self._gold.append(_mean_or_nan(gold_reward))

    def report(self, *, alpha: float = 0.05, n_perm: int = 1000, seed: int = 0) -> DriftReport:
        """Assemble the ``DriftReport``: dose changepoints, the named direction, and the lead time.

        Each feature's dose trajectory gets a CUSUM changepoint; the exploited direction is the
        feature with the largest changepoint magnitude among those significant at ``alpha``. The
        gold reward (when present) gets its own changepoint, and the lead time is the gap between the
        exploited feature's onset and the gold divergence. An ``OnsetAlarm`` is emitted for the
        exploited feature and for the gold divergence.
        """
        t = len(self._dose)
        if t < 3:
            raise ValueError(f"need at least 3 observed steps to report; got {t}")
        steps = np.arange(t)
        dose = np.array(self._dose)  # (T, k)

        cps = [cusum_changepoint(dose[:, i], n_perm=n_perm, seed=seed) for i in range(self.bank.k)]
        dose_cusum = np.array([c.statistic for c in cps])
        dose_p = np.array([c.p_value for c in cps])

        significant = [i for i in range(self.bank.k) if cps[i].p_value < alpha]
        exploited_index: int | None = None
        if significant:
            exploited_index = max(significant, key=lambda i: dose_cusum[i])
        exploited_direction = (
            self.bank.names[exploited_index] if exploited_index is not None else None
        )

        alarms: list[OnsetAlarm] = []
        feature_onset: int | None = None
        if exploited_index is not None:
            cp = cps[exploited_index]
            # Retrospective CUSUM ranks and validates the direction; the sequential CUSUM locates
            # where its drift *starts*, which is the step the lead time is measured from.
            onset = cusum_onset(dose[:, exploited_index])
            feature_onset = onset.index if onset.index is not None else cp.index
            alarms.append(
                OnsetAlarm(
                    signal=self.bank.names[exploited_index],
                    kind="concept-dose",
                    step=feature_onset,
                    statistic=cp.statistic,
                    p_value=cp.p_value,
                )
            )

        gold_arr = np.array(self._gold) if self._has_gold else None
        gold_onset: int | None = None
        if gold_arr is not None and np.all(np.isfinite(gold_arr)):
            gcp = cusum_changepoint(gold_arr, n_perm=n_perm, seed=seed)
            if gcp.p_value < alpha:
                g_onset = cusum_onset(gold_arr)
                gold_onset = g_onset.index if g_onset.index is not None else gcp.index
                alarms.append(
                    OnsetAlarm(
                        signal="gold",
                        kind="gold-divergence",
                        step=gold_onset,
                        statistic=gcp.statistic,
                        p_value=gcp.p_value,
                    )
                )

        lead_time = (
            gold_onset - feature_onset
            if (gold_onset is not None and feature_onset is not None)
            else None
        )

        return DriftReport(
            steps=steps,
            feature_names=list(self.bank.names),
            dose=dose,
            dose_cusum=dose_cusum,
            dose_p=dose_p,
            exploited_direction=exploited_direction,
            exploited_index=exploited_index,
            crystallization=np.array(self._cryst),
            mahalanobis_outlier_rate=np.array(self._maha_rate),
            drift_effective=np.array(self._drift_eff),
            drift_nullspace=np.array(self._drift_null),
            proxy_reward=np.array(self._proxy),
            gold_reward=gold_arr,
            feature_onset=feature_onset,
            gold_onset=gold_onset,
            lead_time=lead_time,
            onset_alarms=alarms,
        )

    def evidence(
        self,
        *,
        subject: SubjectRef | None = None,
        parents: Sequence["EvidenceID"] = (),
        alpha: float = 0.05,
        n_perm: int = 1000,
        seed: int = 0,
    ) -> Evidence[DriftReport]:
        """The ``DriftReport`` wrapped as ``Evidence``.

        Gauge is RAW_ONLY: concept doses and drift magnitudes are projections in one model's
        activation basis, so they are raw coordinates, honest within a rollout but not comparable
        across models without a Frame (gate 2). The lead time and outlier rate are
        frame-free, but the payload as a whole carries raw-coordinate arrays, so RAW_ONLY is the
        conservative correct label.
        """
        report = self.report(alpha=alpha, n_perm=n_perm, seed=seed)
        return make_evidence(
            observable="loops.recorder.drift",
            observable_version="1",
            subject=subject or SubjectRef(),
            value=report,
            uncertainty=Uncertainty(n=len(report.steps), method="cusum-permutation"),
            gauge=GaugeStatus.RAW_ONLY,
            provenance=Provenance(parents=tuple(parents)),
        )


def _mean_or_nan(x: float | Sequence[float] | None) -> float:
    if x is None:
        return float("nan")
    arr = np.asarray(x, dtype=np.float64).ravel()
    return float(arr.mean()) if arr.size else float("nan")


# ---------------------------------------------------------------------------
# The synthetic organism (CPU stand-in for a GPU RL rollout)
# ---------------------------------------------------------------------------


@dataclass
class SyntheticRollout:
    """A planted-hack rollout the recorder is proven on (the crown-jewel test).

    ``activations`` is a list of per-step ``(n_samples, d)`` batches; ``proxy`` and ``gold`` are the
    per-step reward means; ``feature_bank``, ``w_r`` and ``baseline`` are what the recorder is
    constructed from. ``planted_direction`` is the name of the exploited feature the recorder must
    recover, and ``true_gold_onset`` the step gold actually begins to diverge, so the test can check
    the recovered onset and lead time against the plant.
    """

    activations: list[np.ndarray]
    proxy: np.ndarray
    gold: np.ndarray
    feature_bank: DirectionBank
    w_r: np.ndarray
    baseline: np.ndarray
    planted_direction: str
    dose_onset: int
    true_gold_onset: int


def synthetic_hack_rollout(
    *,
    d: int = 16,
    n_features: int = 6,
    n_samples: int = 64,
    n_baseline: int = 256,
    steps: int = 40,
    dose_onset: int = 6,
    gold_tolerance: float = 1.2,
    drift_rate: float = 0.14,
    gold_slope: float = 1.6,
    noise: float = 1.0,
    seed: int = 0,
) -> SyntheticRollout:
    """Generate a CPU rollout that drifts along a planted hack direction (science S13).

    The policy is paid to excite the reward direction ``w_r``, which here is the hack feature's
    direction (feature 0). From ``dose_onset`` on, the activation mean drifts along that direction at
    ``drift_rate`` per step, so the hack feature's dose, the proxy reward, and the crystallization
    along ``w_r`` all climb together. The gold reward is flat until the accumulated hack dose crosses
    ``gold_tolerance``, then falls at ``gold_slope``: the anti-correlation with gold only bites once
    enough hack has accumulated (the ``chi_i > 0``, ``Cov(f_i, gold) <= 0`` signature of Appendix
    A12). That construction is what makes the feature-space onset precede the gold divergence, which
    is the lead time the recorder must recover. The remaining features are distractors with no
    systematic drift.

    Deterministic given ``seed``. This is the synthetic organism the recorder's acceptance test runs
    on; the real RL rollout is GPU-gated and enters through the same ``RolloutRecorder.observe`` API.
    """
    rng = np.random.default_rng(seed)
    # Orthonormal-ish feature directions; feature 0 is the hack direction = w_r.
    raw = rng.standard_normal((n_features, d))
    q, _ = np.linalg.qr(raw.T)
    directions = q.T[:n_features]
    names = ["hack"] + [f"distractor{i}" for i in range(1, n_features)]
    bank = DirectionBank(names=names, directions=directions)
    hack_dir = bank.directions[0]
    w_r = hack_dir.copy()

    baseline = rng.standard_normal((n_baseline, d)) * noise

    activations: list[np.ndarray] = []
    proxy = np.empty(steps)
    gold = np.empty(steps)
    cumulative_dose = 0.0
    true_gold_onset = steps  # default: never, overwritten when it first diverges
    for t in range(steps):
        drift = drift_rate * max(0, t - dose_onset)
        cloud = rng.standard_normal((n_samples, d)) * noise + drift * hack_dir
        activations.append(cloud)
        mean_h = cloud.mean(axis=0)
        proxy[t] = float(mean_h @ w_r)
        cumulative_dose = float(mean_h @ hack_dir)
        excess = max(0.0, cumulative_dose - gold_tolerance)
        if excess > 0 and true_gold_onset == steps:
            true_gold_onset = t
        gold[t] = -gold_slope * excess + rng.standard_normal() * 0.02

    return SyntheticRollout(
        activations=activations,
        proxy=proxy,
        gold=gold,
        feature_bank=bank,
        w_r=w_r,
        baseline=baseline,
        planted_direction="hack",
        dose_onset=dose_onset,
        true_gold_onset=true_gold_onset,
    )


__all__ = [
    "RolloutRecorder",
    "DriftReport",
    "OnsetAlarm",
    "DirectionBank",
    "Changepoint",
    "cusum_changepoint",
    "Onset",
    "cusum_onset",
    "SyntheticRollout",
    "synthetic_hack_rollout",
]
