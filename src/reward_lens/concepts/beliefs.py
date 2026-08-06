"""Answer-keyed belief probes: the epistemic substrate held to the strictest calibration.

A belief probe is a concept probe whose target is an externally verifiable latent: whether the tests
pass, whether the claim is true, whether step k is correct. What makes it a *belief* probe rather
than an ordinary concept probe is the provenance of the label. The label comes from an answer key
(the corruption generator recorded exactly which step it broke; the foundry recorded whether the
receipt was fabricated), not from the model's own output. A probe trained to predict the model's own
verdict would be measuring the model's self-report, which is precisely the quantity a belief probe
must not be contaminated by, because the epistemic-axiological factorization (S5, S8) rests on
separating what is externally true from what the grader values.

So this module is deliberately strict in two ways the ordinary probe factory is not.

- It refuses a self-labeled target. A target whose labels come from the model's own score or output
  is rejected before any fitting, with a message naming the problem. There is no path by which a
  belief probe silently trains on the model's self-report.
- It requires calibration. An answer key must be supplied and the resulting probe must earn a
  `CalibrationRef` (gate 1). A belief probe that cannot be graded against a known key is not returned
  as an EXPLORATORY direction the way a style probe would be; it raises, because an ungraded belief
  probe is exactly the failure this construct exists to prevent.

Everything here runs on CPU: the verifiable key is a function of the data (the corruption's edit
script, the organism's planted rule), so a planted "answer-is-correct" latent can be recovered and
its calibration checked with the answer in hand, which is what the test proves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from reward_lens.concepts.probes import ProbeFit, SiteCaptures, fit_probe
from reward_lens.core.errors import RewardLensError

if TYPE_CHECKING:
    from reward_lens.core.store import EvidenceStore
    from reward_lens.core.types import Site
    from reward_lens.organisms.spec import AnswerKey


class SelfLabeledBeliefError(RewardLensError):
    """Raised when a belief probe is asked to train on a self-labeled (non-verifiable) target.

    A belief probe's label must come from an external answer key. A target sourced from the model's
    own output or score is refused with this error rather than trained, because a belief probe that
    learned the model's self-report would silently defeat the epistemic-axiological separation it
    exists to support (S5).
    """


class UncalibratedBeliefError(RewardLensError):
    """Raised when a belief probe cannot be graded against an answer key (gate 1, strict).

    Ordinary concept probes may be returned EXPLORATORY with ``calibration: None``. Belief probes may
    not: they are held to the strictest calibration standard in the library, so a missing answer key
    is an error, not a downgrade.
    """


@dataclass(frozen=True)
class BeliefTarget:
    """A verifiable belief target: labels from an external key, with its provenance.

    ``key_fn`` maps an ``(item, side)`` to a binary label from the external key (tests pass, claim
    true, step-k correct), or ``None`` to drop that side. ``source`` records where the label comes
    from and is the field the factory checks: only ``"answer_key"`` (an external, verifiable key) is
    accepted; ``"self"`` (the model's own output) is refused. ``verifiable`` must be True for a belief
    target; a target that is not verifiable is refused for the same reason a self-labeled one is.

    The point of carrying the provenance on the target, rather than trusting the caller, is that the
    refusal is structural: a self-labeled target cannot reach the fit no matter how it is passed.
    """

    name: str
    key_fn: Callable[[Any, str], int | None]
    source: str = "answer_key"
    verifiable: bool = True

    def __call__(self, item: Any, side: str) -> int | None:
        return self.key_fn(item, side)

    def check_verifiable(self) -> None:
        """Raise `SelfLabeledBeliefError` unless this target is an external, verifiable key."""
        if self.source != "answer_key" or not self.verifiable:
            raise SelfLabeledBeliefError(
                f"belief target {self.name!r} has source={self.source!r} verifiable="
                f"{self.verifiable}; a belief probe requires an external answer key (source="
                "'answer_key', verifiable=True). A target derived from the model's own output or "
                "score is refused, because a belief probe must not learn the model's self-report. "
                "Use concepts.probes.fit_probe for a self-labeled concept."
            )


def answer_key_target(name: str, key_fn: Callable[[Any, str], int | None]) -> BeliefTarget:
    """A verifiable belief target from an external key function (the accepted construction)."""
    return BeliefTarget(name=name, key_fn=key_fn, source="answer_key", verifiable=True)


def self_labeled_target(name: str, label_fn: Callable[[Any, str], int | None]) -> BeliefTarget:
    """A self-labeled target (from the model's own output); the belief factory refuses this.

    Provided so a caller can express, and the factory can explicitly reject, the exact anti-pattern
    the belief probe guards against. It is never a valid belief target.
    """
    return BeliefTarget(name=name, key_fn=label_fn, source="self", verifiable=False)


def meta_key_target(name: str, meta_field: str, *, positive: Any = True) -> BeliefTarget:
    """A verifiable belief target reading a gold label off an item's meta.

    The foundry and the corruption builders stamp the verifiable ground truth into an item's meta
    (``gold_true`` for a mislabeled-receipt organism, ``gold_correct`` for a planted step error), and
    that meta is the answer key, produced by the generator and not by the model. This target reads
    that field, so its labels are externally verifiable by construction. Both sides of a pair read
    the same field, which is correct when the field is a per-item gold truth.
    """

    def key_fn(item: Any, side: str) -> int | None:
        meta = getattr(item, "meta", None) or {}
        if meta_field not in meta:
            return None
        return int(meta[meta_field] == positive)

    return answer_key_target(name, key_fn)


@dataclass(frozen=True)
class BeliefProbe:
    """A fitted, answer-keyed belief probe.

    Wraps the underlying `ProbeFit` with the belief's provenance: ``belief_name`` and ``key_source``
    (always the external answer key) record that the label was verifiable, and ``organism_family``
    names the answer key it was graded against. The passthroughs expose the direction, its held-out
    AUC, and its calibration so a caller reads ``probe.direction`` and ``probe.held_out_auc``
    directly. ``is_calibrated`` is guaranteed True: the factory raises rather than return an ungraded
    belief probe.
    """

    fit: ProbeFit
    belief_name: str
    key_source: str
    organism_family: str

    @property
    def direction(self) -> Any:
        return self.fit.direction

    @property
    def held_out_auc(self) -> float:
        return self.fit.held_out_auc

    @property
    def calibration(self) -> Any:
        return self.fit.calibration

    @property
    def is_calibrated(self) -> bool:
        return self.fit.calibration is not None

    def decodes_above(self, threshold: float) -> bool:
        """Whether the belief decodes on held-out data above ``threshold`` (answer-keyed AUC)."""
        return bool(np.isfinite(self.held_out_auc) and self.held_out_auc >= threshold)


def fit_belief_probe(
    signal: "SiteCaptures | Any",
    view: Any = None,
    belief: BeliefTarget | None = None,
    sites: "tuple[Site, ...] | None" = None,
    *,
    answer_key: "AnswerKey | None" = None,
    cv: int = 5,
    l2: float = 1.0,
    class_balance: bool = True,
    solver: str = "numpy",
    store: "EvidenceStore | None" = None,
    seed: int = 0,
    decode_threshold: float = 0.7,
) -> BeliefProbe:
    """Train an answer-keyed belief probe under the strict calibration standard.

    The belief probe is the special case of `fit_probe` where the target is an externally verifiable
    latent and calibration is mandatory. This wrapper enforces both. It refuses a self-labeled
    ``belief`` target before any work (`SelfLabeledBeliefError`), it requires an ``answer_key`` to
    grade against (`UncalibratedBeliefError` otherwise), and it verifies the fit actually earned a
    `CalibrationRef`. Everything else (the grouped CV, the per-site sweep, the persisted direction)
    is the ordinary probe machinery.

    ``signal`` is a `SiteCaptures` (the proof path, labels already keyed) or a live signal with
    ``view`` and ``belief`` to capture and label from. When capturing, the label comes from
    ``belief.key_fn`` (the external key), never from the model.

    Args:
        signal: A `SiteCaptures` or a `RewardSignal`.
        view: The DataView, when capturing from a signal.
        belief: The verifiable `BeliefTarget`; a self-labeled one is refused.
        sites: Sites to sweep; defaults to the captures' sites.
        answer_key: The organism `AnswerKey` to grade against (mandatory for a belief probe).
        cv, l2, class_balance, solver, store, seed: As in `fit_probe`.
        decode_threshold: The held-out AUC a belief must clear to count as decoded.

    Returns:
        A `BeliefProbe` that is guaranteed CALIBRATED.
    """
    if belief is not None:
        belief.check_verifiable()

    key = answer_key
    if key is None and isinstance(signal, SiteCaptures):
        key = signal.answer_key
    if key is None:
        raise UncalibratedBeliefError(
            "fit_belief_probe requires an answer_key to grade the belief against (gate 1, strict). "
            "A belief probe is held to the strictest calibration standard and is not returned "
            "EXPLORATORY. Supply the organism AnswerKey, or use concepts.probes.fit_probe for an "
            "uncalibrated concept direction."
        )

    belief_name = belief.name if belief is not None else getattr(signal, "name", "belief")
    fit = fit_probe(
        signal,
        view=view,
        target=belief if belief is not None else None,
        sites=sites,
        cv=cv,
        name=belief_name,
        method="belief_lr",
        l2=l2,
        class_balance=class_balance,
        solver=solver,
        answer_key=key,
        store=store,
        seed=seed,
    )

    if fit.calibration is None:
        raise UncalibratedBeliefError(
            f"belief probe {belief_name!r} did not earn a CalibrationRef even with an answer key; "
            "the scorecard binding failed (empty or single-class held-out labels). A belief probe "
            "is not returned without calibration."
        )

    return BeliefProbe(
        fit=fit,
        belief_name=belief_name,
        key_source="answer_key",
        organism_family=fit.calibration.organism_family,
    )


__all__ = [
    "SelfLabeledBeliefError",
    "UncalibratedBeliefError",
    "BeliefTarget",
    "answer_key_target",
    "self_labeled_target",
    "meta_key_target",
    "BeliefProbe",
    "fit_belief_probe",
]
