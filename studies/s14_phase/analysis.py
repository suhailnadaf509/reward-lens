"""S14 — Phase structure and the hysteresis protocol (DESIGN Part III, Tier IV, S14; deepens T3/T9).

The question is whether the reward-hacking transition is reversible. If hacking were a gradual drift,
a policy pushed past onset by raising optimization pressure could be annealed back by lowering it,
and KL-annealing would be a legitimate recovery tool. If the transition is first-order, the order
parameter follows a different branch on the way down than on the way up, the two branches enclose a
nonzero area, and a hacked policy cannot be annealed back. That loop area is the signature and its
deployment consequence is immediate.

One registered experiment runs here, on a CPU-provable bistable reward system where the hysteresis is
analytically present, so the loop-area measurement is provable without a GPU. The system is a tilted
double well ``F(m; beta) = (m^2 - 1)^2 - beta * m``: an aligned well at ``m = -1`` and a hacked well
at ``m = +1``, with optimization pressure ``beta`` tilting the landscape toward the hacked well. The
protocol runner (``loops.anneal.run_hysteresis``) sweeps ``beta`` up through onset and back down,
letting the order parameter settle to its local optimum at each step from the previous state so
history is carried, and integrates the area the two branches enclose. Following the local rather than
global optimum is what makes the transition first-order, and the metastable gap between the up-branch
onset (near ``beta ~ 1.5``) and the down-branch onset (near ``beta ~ 0``) is the width of the
hysteresis.

The arm that anneals a real KL or pressure parameter on a trained policy carrying a planted exploit,
measuring gold reward and feature occupations on both branches, is recorded as
inconclusive-because-gated: it needs a real RL loop with live training callbacks and a GPU, and the
feature-occupation order parameter would be raw-coordinate rather than the abstract double-well ``m``.

If ``reward_lens.loops.anneal`` is importable the study uses it; if it were not, an inline double-well
responder and shoelace loop-area integration with the same contract run instead.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading, Refusal
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.measure.rate.transition import (
    TransitionFit,
    available_series,
    fit_transition,
    series_from_run,
)
from reward_lens.record.schema import Run
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype

_VERSION = "1.0"


def build_spec() -> StudySpec:
    """The frozen S14 spec: one hypothesis that the annealing loop has nonzero area."""
    return StudySpec(
        id="s14-phase",
        title="Phase structure and hysteresis: reward hacking is first-order, so a hacked policy "
        "cannot be annealed back",
        science="S14-phase",
        hypotheses=(
            Hypothesis(
                id="H1-nonzero-loop-area",
                statement="the anneal-up / anneal-down loop encloses a nonzero area, the signature "
                "of an irreversible first-order transition",
                prediction=Prediction(metric="loop_area", comparator=">", threshold=0.1),
                scoreboard_row="T9",
            ),
        ),
        analysis="studies.s14_phase.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "a CPU-provable tilted double well; the real-RL anneal of a KL/pressure "
                "parameter on a trained exploited policy is GPU-gated"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-loop-closes",
                metric="loop_area",
                comparator="<",
                threshold=0.01,
                description="the loop closes (a smooth crossover retraces its path), so hacking is a "
                "gradual reversible drift and KL-annealing is a legitimate recovery tool, which is a "
                "publishable negative result",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Inline fallback (only used if reward_lens.loops.anneal is unavailable)
# ---------------------------------------------------------------------------


def _inline_double_well(beta: float, m: float, *, n_iter: int = 400, lr: float = 0.02) -> float:
    """Gradient relaxation of the order parameter in the tilted double well from state ``m``."""
    x = float(m)
    for _ in range(n_iter):
        x -= lr * (4.0 * x * (x * x - 1.0) - beta)
    return x


def _inline_hysteresis(beta0: float, beta1: float, n: int):
    """Sweep beta up then down through the inline double well; return branches and shoelace area."""
    up = np.linspace(beta0, beta1, n)
    down = up[::-1].copy()
    order_up = np.empty(n)
    state = -1.0
    for i, b in enumerate(up):
        state = _inline_double_well(float(b), state)
        order_up[i] = state
    order_down = np.empty(n)
    for i, b in enumerate(down):
        state = _inline_double_well(float(b), state)
        order_down[i] = state
    bx = np.concatenate([up, down])
    by = np.concatenate([order_up, order_down])
    area = float(0.5 * abs(np.sum(bx * np.roll(by, -1) - np.roll(bx, -1) * by)))
    return up, order_up, down, order_down, area


# ---------------------------------------------------------------------------
# Gated-arm evidence
# ---------------------------------------------------------------------------


def _gated_arm(
    study_id: str, subject: SubjectRef, *, arm: str, needs: str, produces: str
) -> Evidence:
    """A REGISTERED record that an arm is inconclusive because a subsystem or hardware is missing."""
    return make_evidence(
        observable="S14.GatedArm",
        observable_version=_VERSION,
        subject=subject,
        value={
            "arm": arm,
            "status": "inconclusive-because-gated",
            "needs": needs,
            "produces": produces,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )


def analyze(run) -> StudyResult:
    """Run the hysteresis protocol on the bistable stand-in, record the gated real-RL design."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    try:
        from reward_lens.loops.anneal import (
            double_well_responder,
            run_hysteresis,
            up_down_schedule,
        )

        responder = double_well_responder()
        up, down = up_down_schedule(0.0, 3.0, 40)
        hyst_ev = run_hysteresis(responder, up, down, init_state=-1.0)
        run.record(hyst_ev)
        loop = hyst_ev.value
        loop_area = float(loop.loop_area)
        up_transition = (
            float(loop.up_transition) if loop.up_transition is not None else float("nan")
        )
        down_transition = (
            float(loop.down_transition) if loop.down_transition is not None else float("nan")
        )
        order_up_final = float(loop.order_up[-1])
        order_down_final = float(loop.order_down[-1])
        parents = (hyst_ev.id,)
    except ImportError:
        up, order_up, down, order_down, loop_area = _inline_hysteresis(0.0, 3.0, 40)
        d_up = np.abs(np.diff(order_up)) / (np.abs(np.diff(up)) + 1e-12)
        d_down = np.abs(np.diff(order_down)) / (np.abs(np.diff(down)) + 1e-12)
        up_transition = float(0.5 * (up[np.argmax(d_up)] + up[np.argmax(d_up) + 1]))
        down_transition = float(0.5 * (down[np.argmax(d_down)] + down[np.argmax(d_down) + 1]))
        order_up_final = float(order_up[-1])
        order_down_final = float(order_down[-1])
        parents = ()

    # The metastable gap between the up-branch and down-branch onsets is the width of the hysteresis.
    hysteresis_width = float(abs(up_transition - down_transition))
    ev_loop = make_evidence(
        observable="S14.HysteresisLoop",
        observable_version=_VERSION,
        subject=subject,
        value={
            "loop_area": loop_area,
            "irreversible": bool(loop_area > 0.01),
            "up_transition": up_transition,
            "down_transition": down_transition,
            "hysteresis_width": hysteresis_width,
            "order_up_final": order_up_final,
            "order_down_final": order_down_final,
        },
        uncertainty=Uncertainty(n=len(up) + len(down), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=parents),
        registered=True,
    )
    run.record(ev_loop)

    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="real-rl-anneal",
            needs="a real RL loop with live training callbacks (reward_lens.loops training "
            "integrations) and a GPU, on a trained policy carrying a planted exploit",
            produces="the gold reward and raw-coordinate feature occupations on both anneal branches, "
            "the production form of the loop-area irreversibility test",
        )
    )

    metrics = {"loop_area": loop_area, "hysteresis_width": hysteresis_width}
    summary = (
        f"The anneal-up / anneal-down protocol on the tilted double well enclosed a loop of area "
        f"{loop_area:.3f} (nonzero, so first-order and irreversible): the aligned well destabilizes "
        f"near beta {up_transition:.2f} on the way up but the hacked well persists until beta "
        f"{down_transition:.2f} on the way down, a hysteresis width of {hysteresis_width:.2f}. The "
        f"real-RL anneal on a trained exploited policy is recorded as inconclusive-because-gated on "
        f"a GPU training loop."
    )
    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S14 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s14_phase",
    spec=build_spec(),
    headline="run.hysteresis_area",
    destination=(
        "the H series. `run.hysteresis_area` is H3's quantity and this study is H3's protocol run "
        "on a bistable stand-in; `run.transition_width` is H4's, ships in measure/rate/transition.py, "
        "and is the one H-series quantity a single training record can answer, which is why `read` "
        "measures it and refuses the loop area."
    ),
    needs={Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="loop_area",
            quantity="run.hysteresis_area",
            arc="hysteresis-sweep",
            arm="anneal-up-then-down",
            source="organism",
            note=(
                "the area the up-branch and down-branch enclose in the (pressure, order parameter) "
                "plane, dimensionless as the registered id is. It needs CONTROL of the pressure "
                "parameter in both directions, which is the arm's whole content: a record holds one "
                "monotone pass and an area needs two. H3, the instrument that would extrapolate "
                "this to zero sweep rate, is registered and not built, so the number here is the "
                "raw single-rate area and is confounded with lag until H3 exists."
            ),
        ),
    ),
    waiting_on=(
        "H3 (`run.hysteresis_area`): the quantity is registered and no instrument estimates it. The "
        "area this study measures is at one sweep rate; H3's rate extrapolation is what separates "
        "irreversibility from lag, and until it exists the binding is to the quantity and not to a "
        "shipped estimator."
    ),
)

#: The order parameter, in the order it is looked for. A gold probe is the closest thing a record
#: carries to the behaviour a phase claim is about; the group mean is the realised training reward
#: and is what is left when no probe was recorded.
_ORDER_PARAMETER_SERIES = ("group_mean", "reward")

#: `read` is called several times per run by the acceptance suite and a four-parameter fit with its
#: block bootstrap is 2.4 seconds on a 200-step series. The fit is deterministic given the record,
#: so it is cached on the run id rather than repeated. Cleared by process exit, like any memo.
_FIT_CACHE: dict[tuple[str, str], "TransitionFit | Refusal"] = {}


def _order_parameter(run: Run) -> tuple[str, np.ndarray, np.ndarray] | None:
    """The series a transition would be fitted to, with its step axis, or None if none exists."""
    available = available_series(run)
    for name in _ORDER_PARAMETER_SERIES:
        if available.get(name, 0) >= 2:
            values, steps = series_from_run(run, name)
            if values.size:
                return name, values, steps
    return None


def _fitted(
    run: Run, name: str, values: np.ndarray, steps: np.ndarray
) -> "TransitionFit | Refusal":
    key = (str(run.id), name)
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = fit_transition(
            values, steps, series=name, instrument="H4 TransitionWidth"
        )
    return _FIT_CACHE[key]


def read(run: Run) -> Reading:
    """S14 against a training record: the transition width if there is a transition, never the area.

    A hysteresis loop needs the control parameter swept up and then back down, and a training record
    is one monotone pass. No record of a single run contains a loop area, whatever else it contains,
    so `loop_area` is refused on every record rather than approximated from the forward branch. What
    a record can answer is the other half of the H series: the fitted width of the behavioural
    transition, which is H4's quantity and the unit every lead time in this library is reported in.

    Scope limit, three lines in: this fits the width of whichever order-parameter series the record
    carries, and on both GRPO fixtures H4 refuses `BELOW_LOD` because a real optimisation trace with
    no behavioural transition in it is not distinguishable from a trend. That refusal is the correct
    reading of those runs and it is passed through rather than softened.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    picked = _order_parameter(run)
    if picked is None:
        return RETYPE.incomplete(
            field="order-parameter series: no per-group mean reward and no gold probe",
            subject=f"run {run.id} over {run.n_steps} steps",
            remedy=(
                "record a gold probe at each step, or convert the run with a tap that writes the "
                "per-group reward. A phase claim is about an order parameter moving, and a record "
                "with no per-step outcome series has nothing for the claim to be about."
            ),
            steps=int(run.n_steps),
        )

    name, values, steps = picked
    fit = _fitted(run, name, values, steps)
    if isinstance(fit, Refusal):
        return Refusal(
            instrument="s14_phase.read",
            reason=fit.reason,
            detail=(
                f"the loop area needs a reverse sweep and this record is one monotone pass, so it "
                f"is refused on every record. The transition width, which a record can answer, was "
                f"fitted to {name!r} over {values.size} steps by H4 and refused: {fit.detail}"
            ),
            remedy=(
                "for the width, point this at a series that contains a behavioural transition: the "
                "AISI labelled-hack-rate series is the subject this claim needs, at a fitted width "
                "of 23.9 steps. For the loop area, run the anneal protocol through "
                "loops.anneal.run_hysteresis with CONTROL of the pressure parameter, or read it "
                "from analyze() on the bistable stand-in."
            ),
            statistics={
                "series": name,
                "n_steps": int(values.size),
                "loop_area": "refused: a record carries no reverse sweep",
                **{k: v for k, v in fit.statistics.items() if k != "series"},
            },
        )

    return RETYPE.evidence(
        run,
        {},
        measured={"transition_width": (float(fit.width), "run.transition_width")},
        quantity="run.transition_width",
        refusals={"loop_area": _LOOP_AREA_REFUSAL},
        summary=(
            f"the transition in {name!r} has a fitted width of {fit.width:.4g} steps with its "
            f"midpoint at step {fit.midpoint:.4g}. The loop area is not answerable from any single "
            f"record: it needs the pressure parameter swept up and then back down, and this run is "
            f"one monotone pass."
        ),
        gauge=GaugeStatus.INVARIANT,
        series=name,
        midpoint_step=float(fit.midpoint),
        n_steps=int(values.size),
    )


_LOOP_AREA_REFUSAL = (
    "the loop area is the area enclosed by the up-branch and the down-branch of a swept control "
    "parameter, and a training record contains one pass in one direction. Sweep the pressure "
    "parameter back down with loops.anneal.run_hysteresis and record both branches, or read the "
    "area from analyze() on the bistable stand-in."
)

_ACCESS_REMEDY = (
    "open a run written by the recorder. S14 needs no activations at this rung: the width is fitted "
    "to a per-step outcome series and the loop area needs CONTROL of the pressure parameter rather "
    "than a deeper read of this record."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]
