"""What the record says about each term of the residual budget, and what it does not say.

Six of the nine itemised contributions are properties of the training loop rather than of
the features: the KL coefficient, the entropy coefficient, the rollout staleness, the clip
fraction, the optimiser's raw-minus-applied gap, and the curvature. This module reads them off a
`Run` and reports each one as present with a value or absent with the reason and the remedy.

**It never substitutes a default.** A missing entropy coefficient is not zero, because a trainer
that applied an entropy bonus and did not log it produces exactly the same record as one that
applied none, and a budget that assumes the second closes by construction. The distinction between
"measured and zero" and "not in the record" is the whole reason this is a separate object: the
first is a result and the second is a gap, and a budget that cannot tell them apart cannot say
which of its terms it is entitled to trust.

On the two GRPO records this library ships, both readings occur. `beta` is 0.0 in the schedule at
every step, so the KL pull is exactly zero and `u_KL` vanishes for a reason rather than for want of
a number. `grad_norm_unclipped`, `update_norm` and `clip_fraction` are all `None`, and the recorded
config names `adamw_torch_fused`, so the optimiser certainly carries momentum and the record
certainly cannot say how much.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from reward_lens.measure.ledger.price import Window, whole_run
from reward_lens.record.schema import Run


@dataclass(frozen=True)
class Absent:
    """A budget input the record does not carry, with what would supply it.

    ``remedy`` is an instruction rather than a restatement. "The tap does not write
    `optimizer.update_norm`; record the parameter-vector norm before and after `optimizer.step()`"
    is a remedy, and "update_norm is None" is not.
    """

    field: str
    why: str
    remedy: str

    def render(self) -> str:
        return f"{self.field}: {self.why}  ->  {self.remedy}"


@dataclass(frozen=True)
class RunFacts:
    """Every scalar the residual budget needs from the record, present or named absent.

    Each field is either a float or an `Absent`. A term whose input is `Absent` is carried in the
    budget's ``missing`` list rather than composed into the quadrature sum, so the combined
    uncertainty is visibly a lower bound whenever anything is missing.
    """

    kl_coefficient: float | Absent
    entropy_coefficient: float | Absent
    max_staleness: int | Absent
    clip_fraction: float | Absent
    momentum_gap: float | Absent
    hessian_norm: float | Absent
    optimizer_family: str
    n_steps: int
    detail: str = ""

    @property
    def absent(self) -> tuple[Absent, ...]:
        """Every input the record could not supply, in the order the budget itemises them."""
        out = []
        for value in (
            self.kl_coefficient,
            self.entropy_coefficient,
            self.max_staleness,
            self.clip_fraction,
            self.momentum_gap,
            self.hessian_norm,
        ):
            if isinstance(value, Absent):
                out.append(value)
        return tuple(out)

    def render(self) -> str:
        lines = [f"run facts over {self.n_steps} steps (optimizer: {self.optimizer_family})"]
        for name, value in (
            ("kl_coefficient", self.kl_coefficient),
            ("entropy_coefficient", self.entropy_coefficient),
            ("max_staleness", self.max_staleness),
            ("clip_fraction", self.clip_fraction),
            ("momentum_gap", self.momentum_gap),
            ("hessian_norm", self.hessian_norm),
        ):
            if isinstance(value, Absent):
                lines.append(f"    {name:<20} absent: {value.why}")
            else:
                lines.append(f"    {name:<20} {value:g}")
        return "\n".join(lines)


def _config_of(run_: Run) -> Mapping[str, Any]:
    extra = getattr(run_.lineage, "extra", {}) or {}
    config = extra.get("config")
    return config if isinstance(config, Mapping) else {}


#: The keys a trainer might record an entropy bonus coefficient under. Checked in this order.
#: None of them is present on the two GRPO records, which is why the term comes back `Absent`
#: rather than zero: TRL logs the measured entropy and never the coefficient multiplying it.
_ENTROPY_KEYS = ("entropy_coefficient", "entropy_coeff", "ent_coef", "entropy_bonus")


def _relative_momentum_gaps(steps: Any) -> list[float]:
    """`|‖Δθ‖ − η‖g‖| / ‖Δθ‖` per step: how far the applied step is from the raw gradient step.

    Reported as a fraction rather than as a norm difference because that is the form the budget can
    use. `Δz_pred` is linear in the step to first order, so a step whose magnitude differs from the
    raw gradient step by 12% moves the predicted response by 12%, and the fraction converts into
    feature units by multiplying the predicted response. The norms themselves are in parameter
    space and there is no conversion from those to feature units without the Jacobian.

    The deviation this carries, stated here because it is not small: Adam changes the *direction* of
    the step as well as its magnitude, and a magnitude ratio does not see a rotation. So this is a
    lower bound on the momentum contribution even when both norms are recorded.
    """
    out: list[float] = []
    for step in steps:
        raw = step.optimizer.grad_norm_unclipped
        applied = step.optimizer.update_norm
        eta = step.schedule.get("learning_rate")
        if raw is None or applied is None or eta is None or float(applied) <= 0.0:
            continue
        out.append(abs(float(applied) - float(eta) * float(raw)) / float(applied))
    return out


def facts_from_run(run_: Run, window: Window | None = None) -> RunFacts:
    """Read the budget's scalar inputs off a record, naming every one it does not carry.

    The KL coefficient comes from `Step.schedule["beta"]`, which is where the TRL tap writes it, and
    a run whose `beta` moves during the window is reported at its maximum: the budget wants the
    largest pull the window contained, not an average that hides a phase where the penalty was on.

    Staleness comes from `SegmentProvenance.staleness_steps` over every segment of every
    trajectory. Zero everywhere means every rollout was generated by the policy that was then
    updated on it, which makes `u_stale` exactly zero rather than small.
    """
    lo, hi = window if window is not None else whole_run(run_)
    steps = [s for s in run_.steps.slice(lo, hi)]
    config = _config_of(run_)

    betas = [s.schedule.get("beta") for s in steps]
    present = [float(b) for b in betas if b is not None]
    kl: float | Absent = (
        max(present)
        if present
        else Absent(
            "schedule['beta']",
            "no step in the window recorded a KL coefficient",
            "pass `kl_coefficient=` if the trainer applied one. The record's schedule is where "
            "the TRL tap writes it, so an absent value means the tap did not run or the trainer "
            "is not TRL.",
        )
    )

    entropy: float | Absent = Absent(
        "entropy coefficient",
        "neither the schedule nor the recorded config carries a coefficient for the entropy bonus",
        "pass `entropy_coefficient=` from the trainer's own config. `optimizer.entropy` on the "
        "record is the measured entropy of the policy and not the coefficient multiplying it, so "
        "it cannot stand in. Pass 0.0 if the trainer applied no bonus and the term becomes exactly "
        "zero.",
    )
    for key in _ENTROPY_KEYS:
        value = config.get(key)
        if value is not None:
            entropy = float(value)
            break

    stalenesses = [
        int(segment.staleness_steps)
        for step in steps
        for group in step.groups
        for trajectory in group.trajectories
        for segment in trajectory.provenance
        if segment.staleness_steps is not None
    ]
    staleness: int | Absent = (
        max(stalenesses)
        if stalenesses
        else Absent(
            "SegmentProvenance.staleness_steps",
            "no trajectory segment in the window declares its staleness",
            "convert the run with a tap that writes segment provenance, or pass "
            "`max_staleness=` if you know the generation lag from the trainer's config.",
        )
    )

    clips = [s.optimizer.clip_fraction for s in steps if s.optimizer.clip_fraction is not None]
    clip: float | Absent = (
        max(float(c) for c in clips)
        if clips
        else Absent(
            "optimizer.clip_fraction",
            "no step recorded what fraction of tokens the policy ratio clipped",
            "record the trainer's own clip-ratio metric through the tap. TRL computes it for the "
            "loss and logs it under `clip_ratio`; the tap can forward it into "
            "`OptimizerTelemetry.clip_fraction`.",
        )
    )

    gaps = _relative_momentum_gaps(steps)
    optimizer_family = str(config.get("optim", "unknown"))
    momentum: float | Absent = (
        max(gaps)
        if gaps
        else Absent(
            "optimizer.grad_norm_unclipped and optimizer.update_norm",
            f"the record carries neither, so the raw gradient step cannot be differenced against "
            f"the applied one. The recorded optimiser is {optimizer_family!r}, which is not SGD, "
            f"so this term is certainly non-zero and certainly not measurable here",
            "record both norms in the tap: the raw gradient norm before `clip_grad_norm_` and the "
            "parameter-vector delta across `optimizer.step()`. Both are one line each and the "
            "optimiser has both.",
        )
    )

    hessian: float | Absent = Absent(
        "the curvature norm",
        "no record carries the second derivative of the feature expectation",
        "compute it with `policy.hf`'s `hvp` on the checkpoint that wrote this step and pass "
        "`hessian_norm=`. There is one Hessian-vector product in the library and it is asserted "
        "against a finite difference; do not write a second.",
    )

    return RunFacts(
        kl_coefficient=kl,
        entropy_coefficient=entropy,
        max_staleness=staleness,
        clip_fraction=clip,
        momentum_gap=momentum,
        hessian_norm=hessian,
        optimizer_family=optimizer_family,
        n_steps=len(steps),
        detail=f"read from run {run_.id} over [{lo}, {hi})",
    )


__all__ = ["Absent", "RunFacts", "facts_from_run"]
