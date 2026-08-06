"""Beta schedules and the hysteresis protocol runner (science S14).

The phase-structure science (S14) asks whether the reward-hacking transition is reversible. If it
is first-order, a policy pushed past the transition by raising optimization pressure cannot be
annealed back by lowering it: the order parameter follows a different branch on the way down than on
the way up, and the two branches enclose a nonzero area. That loop area is the signature, and its
deployment consequence is immediate, because it says KL-annealing is not a recovery tool for a
policy that has already hacked (S14: "nonzero loop area = irreversibility").

The protocol runner sweeps a control parameter ``beta`` up through the onset and back down, letting
a stateful responder settle to its steady state at each ``beta`` starting from the previous one, so
history is carried and metastability can express itself. It returns both branches and the enclosed
loop area. ``bon`` supplies the quasi-static reference curve this is read against.

The bistable responder here is the CPU-provable stand-in: a tilted double-well, reward favoring the
hacked well, the system following its local optimum by gradient relaxation. The same runner accepts
a responder backed by a real feature occupation measured on a training run, which is GPU-gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef

if TYPE_CHECKING:
    from reward_lens.core.types import EvidenceID

Responder = Callable[[float, float], float]


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------


def linear_schedule(beta0: float, beta1: float, n: int) -> np.ndarray:
    """A linear ramp of ``n`` betas from ``beta0`` to ``beta1`` inclusive."""
    if n < 2:
        raise ValueError(f"schedule needs at least 2 points; got {n}")
    return np.linspace(beta0, beta1, n)


def cosine_schedule(beta0: float, beta1: float, n: int) -> np.ndarray:
    """A cosine ramp from ``beta0`` to ``beta1``: slow at the ends, fast through the middle.

    The half-cosine ``beta0 + (beta1 - beta0) * (1 - cos(pi u)) / 2`` for ``u`` in ``[0, 1]``. Useful
    when the transition is expected mid-range and the sweep should linger near onset.
    """
    if n < 2:
        raise ValueError(f"schedule needs at least 2 points; got {n}")
    u = np.linspace(0.0, 1.0, n)
    return beta0 + (beta1 - beta0) * (1.0 - np.cos(np.pi * u)) / 2.0


def up_down_schedule(beta0: float, beta1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The hysteresis sweep as ``(up, down)``: ``beta0 -> beta1`` then ``beta1 -> beta0``.

    The down leg is the up leg reversed, so the two share endpoints and the branches close into a
    loop whose area the runner integrates.
    """
    up = linear_schedule(beta0, beta1, n)
    return up, up[::-1].copy()


# ---------------------------------------------------------------------------
# The bistable responder (CPU stand-in)
# ---------------------------------------------------------------------------


def double_well_responder(
    *, reward_weight: float = 1.0, n_iter: int = 400, lr: float = 0.02
) -> Responder:
    """A tilted double-well responder: the CPU-provable bistable reward system (S14).

    The order parameter ``m`` lives in a symmetric double well ``U(m) = (m^2 - 1)^2`` with minima at
    the aligned well ``m = -1`` and the hacked well ``m = +1``. Optimization pressure ``beta`` tilts
    the landscape toward the hacked well through a linear reward ``R(m) = reward_weight * m``, so the
    effective potential is ``F(m; beta) = (m^2 - 1)^2 - beta * reward_weight * m``. The responder
    returns the local optimum reached by gradient relaxation from the incoming state, so once the
    aligned well loses stability (near ``beta ~ 1.54 / reward_weight``) the system rolls to the
    hacked well and does not return until ``beta`` reverses well past zero. That metastability is the
    hysteresis, and following the local (not global) optimum is what makes it first-order.

    Returns a ``responder(beta, m) -> m`` the protocol runner folds over a beta schedule.
    """

    def responder(beta: float, m: float) -> float:
        x = float(m)
        for _ in range(n_iter):
            grad = 4.0 * x * (x * x - 1.0) - beta * reward_weight
            x -= lr * grad
        return x

    return responder


# ---------------------------------------------------------------------------
# The protocol runner
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class HysteresisLoop:
    """The up and down branches of a beta sweep and the loop area between them (S14).

    ``beta_up`` / ``order_up`` is the anneal-up branch, ``beta_down`` / ``order_down`` the
    anneal-down branch. ``loop_area`` is the signed-magnitude area the branches enclose in the
    ``(beta, order)`` plane; ``irreversible`` is True when it exceeds ``area_tol``. ``up_transition``
    and ``down_transition`` are the betas of steepest change on each branch, the onset locations; the
    gap between them is the width of the hysteresis.
    """

    beta_up: np.ndarray
    order_up: np.ndarray
    beta_down: np.ndarray
    order_down: np.ndarray
    loop_area: float
    irreversible: bool
    up_transition: float | None
    down_transition: float | None


def _sweep(responder: Responder, betas: np.ndarray, state: float) -> tuple[np.ndarray, float]:
    """Fold the responder over a beta schedule, carrying state; return the branch and the last state."""
    out = np.empty(betas.size, dtype=np.float64)
    for i, b in enumerate(betas):
        state = float(responder(float(b), state))
        out[i] = state
    return out, state


def _transition_beta(betas: np.ndarray, order: np.ndarray) -> float | None:
    """The beta of steepest change along a branch (the onset), or None if the branch is flat."""
    if betas.size < 2:
        return None
    d = np.abs(np.diff(order)) / (np.abs(np.diff(betas)) + 1e-12)
    if not np.any(d > 0):
        return None
    j = int(np.argmax(d))
    return float(0.5 * (betas[j] + betas[j + 1]))


def _loop_area(
    beta_up: np.ndarray,
    order_up: np.ndarray,
    beta_down: np.ndarray,
    order_down: np.ndarray,
) -> float:
    """Enclosed area of the closed loop (up branch then down branch) by the shoelace formula."""
    bx = np.concatenate([beta_up, beta_down])
    by = np.concatenate([order_up, order_down])
    return float(0.5 * abs(np.sum(bx * np.roll(by, -1) - np.roll(bx, -1) * by)))


def run_hysteresis(
    responder: Responder,
    betas_up: Sequence[float],
    betas_down: Sequence[float] | None = None,
    *,
    init_state: float = -1.0,
    area_tol: float = 1e-3,
    subject: SubjectRef | None = None,
    parents: Sequence["EvidenceID"] = (),
) -> Evidence[HysteresisLoop]:
    """Run the anneal-up / anneal-down protocol and measure the hysteresis loop area.

    ``responder(beta, state) -> state`` settles the order parameter at each ``beta`` starting from
    the previous one. The up branch folds it over ``betas_up`` from ``init_state``; the down branch
    continues from the up branch's final state over ``betas_down`` (default: ``betas_up`` reversed).
    A first-order system leaves the two branches apart and the enclosed area is nonzero; a smooth
    crossover retraces its path and the area is ~0. That area is the irreversibility signature S14
    reports.

    Returns ``Evidence[HysteresisLoop]``. Gauge is INVARIANT: the loop area of an abstract order
    parameter against a dimensionless control is a real, frame-free number. A responder backed by a
    raw feature occupation would make the order parameter raw-coordinate; pass such a study's own
    gauge when wrapping that case.
    """
    up = np.asarray(list(betas_up), dtype=np.float64)
    down = (
        np.asarray(list(betas_down), dtype=np.float64)
        if betas_down is not None
        else up[::-1].copy()
    )
    if up.size < 2 or down.size < 2:
        raise ValueError("both legs of the sweep need at least 2 betas")

    order_up, last = _sweep(responder, up, float(init_state))
    order_down, _ = _sweep(responder, down, last)

    area = _loop_area(up, order_up, down, order_down)
    payload = HysteresisLoop(
        beta_up=up,
        order_up=order_up,
        beta_down=down,
        order_down=order_down,
        loop_area=area,
        irreversible=area > area_tol,
        up_transition=_transition_beta(up, order_up),
        down_transition=_transition_beta(down, order_down),
    )
    return make_evidence(
        observable="loops.anneal.hysteresis",
        observable_version="1",
        subject=subject or SubjectRef(),
        value=payload,
        uncertainty=Uncertainty(n=up.size + down.size, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(parents=tuple(parents)),
    )


__all__ = [
    "linear_schedule",
    "cosine_schedule",
    "up_down_schedule",
    "double_well_responder",
    "run_hysteresis",
    "HysteresisLoop",
    "Responder",
]
