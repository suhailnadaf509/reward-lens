"""The one registry row this package proposes, written out because it does not exist yet.

`spec/QUANTITIES.yaml` carries 174 quantities and two of them are policy-side:
`policy.train_infer_logprob_mismatch` (E6) and `policy.update_sparsity_vs_staleness` (K4). Neither is
what a linear probe on a policy's residual stream estimates, and none of the 47 `grader.*` rows is
either, because they are all namespaced to the wrong component: `grader.eval_awareness` is the
closest in construction (a probe's held-out balanced accuracy) and it is a claim about a grader's
prompts.

So this package needs one row it cannot write, since `spec/QUANTITIES.yaml` is maintained for the
library as a whole rather than per package. The row is defined here as data, `register_proposed`
puts it in this process only, and `as_yaml_rows` emits it in the file's own field order so the
amendment does not have to be retyped.
That is the pattern `measure/decision/quantities.py` established and the reason it is a pattern: an
instrument blocked on a registry row it does not own should still be runnable, testable and
lintable, with the gap named rather than routed around.

Nothing here is called at import. A quantity that appears in the registry as a side effect of an
import is a quantity whose definition depends on import order, which is what `Registry.register`
refuses and is right to refuse.
"""

from __future__ import annotations

from reward_lens.core.quantity import Quantity, Unit, register_quantity

#: A fraction of items classified correctly, out of fold. Dimensionless, per nothing, unscaled.
_FRACTION = Unit(dimension="fraction", per=None, scale=None, as_printed="fraction")


POLICY_READOUT_RECOVERABILITY = Quantity(
    id="policy.readout_recoverability",
    definition=(
        "The out-of-fold fraction of items on which a ridge probe, fitted on a policy's residual "
        "stream at one declared site and one declared token position, correctly classifies a "
        "binary outcome recorded alongside those items. Cross-fitted over stratified folds, so "
        "every item's score is produced by a probe that did not see it, and thresholded at the "
        "midpoint of the two class-mean scores rather than at whichever threshold maximises "
        "accuracy on the evaluation set."
    ),
    unit=_FRACTION,
    invariance="repr.basis",
    interpretation=(
        "How much of an outcome is linearly present in the policy's own activations. It is not a "
        "claim that the policy computes the outcome, and it is not worth anything on its own: the "
        "number that decides whether reading the activations bought anything is the incremental "
        "validity of this reading against the black-box bank run on the same items, which is why "
        "every reading of this quantity carries an `IncrementalValidity` record."
    ),
    support=(0.0, 1.0),
    wedge=False,
)


#: Every proposal in this package. One so far.
PROPOSED: tuple[Quantity, ...] = (POLICY_READOUT_RECOVERABILITY,)

#: How many rungs each proposal's ladder has as built. One: the probe is the only estimator, and a
#: probe fitted on a different site is the same estimator pointed somewhere else rather than a
#: cheaper rung of the same quantity.
PROPOSED_RUNGS: dict[str, int] = {"policy.readout_recoverability": 1}

#: Which instrument the row should name once the amendment lands.
INSTRUMENT_LISTS: dict[str, tuple[str, ...]] = {
    "policy.readout_recoverability": ("PolicyReadoutProbe",),
}


def register_proposed() -> list[str]:
    """Register the proposals in this process only. Not called at import, by design.

    The acceptance test calls this so the instrument is lint-clean, which separates "this instrument
    estimates a quantity nobody has registered" from "this instrument has four other problems". It
    is not a substitute for the row landing in `spec/QUANTITIES.yaml`.
    """
    from reward_lens.core.quantity import QUANTITIES

    added: list[str] = []
    for q in PROPOSED:
        if q.id not in QUANTITIES:
            register_quantity(q)
            added.append(q.id)
    return added


def _folded(key: str, text: str, indent: str = "  ", width: int = 96) -> list[str]:
    """A YAML folded block scalar, the only safe way to emit prose with colons and quotes in it."""
    lines = [f"{indent}{key}: >-"]
    words = text.split()
    current = f"{indent}  "
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = f"{indent}  "
        current += word + " "
    if current.strip():
        lines.append(current.rstrip())
    return lines


def as_yaml_rows() -> str:
    """The proposed rows in `spec/QUANTITIES.yaml`'s own field order, ready to paste."""
    out: list[str] = []
    for q in PROPOSED:
        out.append(f"- id: {q.id}")
        out.append("  unit:")
        out.append(f"    as_printed: {q.unit.as_printed}")
        out.append(f"    dimension: {q.unit.dimension}")
        out.append(f"    per: {q.unit.per if q.unit.per is not None else 'null'}")
        out.append(f"    scale: {q.unit.scale if q.unit.scale is not None else 'null'}")
        out.append(f"  invariance_group: {q.invariance}")
        out.append("  min_access: POLICY:FORWARD")
        out.append(f"  rungs: {PROPOSED_RUNGS[q.id]}")
        out.append("  instrument:")
        for inst in INSTRUMENT_LISTS[q.id]:
            out.append(f"  - {inst}")
        out.append(f"  wedge: {'true' if q.wedge else 'false'}")
        out.extend(_folded("definition", q.definition))
        out.extend(_folded("interpretation", q.interpretation))
        lo, hi = q.support or (0.0, 1.0)
        out.append(f"  support: [{lo}, {hi}]")
    return "\n".join(out)


__all__ = [
    "INSTRUMENT_LISTS",
    "POLICY_READOUT_RECOVERABILITY",
    "PROPOSED",
    "PROPOSED_RUNGS",
    "as_yaml_rows",
    "register_proposed",
]


def register_estimators() -> None:
    """Register C9's estimator, idempotently, now that the quantity row has landed.

    The row is in `spec/QUANTITIES.yaml` now, so `register_proposed` is a no-op on a normal load
    and stays only as the escape hatch it was written to be. What the row alone does not supply is
    an `EstimatorEntry`, and lint rule two reads the estimator registry rather than the catalogue:
    a quantity with a catalogue record and no estimator is still an open research target as far as
    the docs build is concerned, and this one is not open. It is built, running, and produced the
    first white-box reading in the library.

    Idempotent and re-callable for the reason recorded on the forecast ledger's equivalent: the
    estimator registry is a process-global that several test modules mutate by snapshotting and
    popping, and a module-level registration that lands inside one of those windows does not come
    back on its own.
    """
    from reward_lens.core.envelope import UNCONDITIONAL
    from reward_lens.core.quantity import (
        ESTIMATORS,
        BiasStatement,
        CostModel,
        EstimatorEntry,
        register_estimator,
    )
    from reward_lens.core.types import Access, Component

    if any(e.quantity == "policy.readout_recoverability" for e in ESTIMATORS.values()):
        return

    register_estimator(
        EstimatorEntry(
            quantity="policy.readout_recoverability",
            impl="policy.readout_recoverability.ridge_cross_fitted",
            requires={Component.POLICY: Access.FORWARD},
            envelope=UNCONDITIONAL,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "the ridge penalty is fixed at alpha times the mean eigenvalue of the training "
                    "Gram rather than tuned, because a hyperparameter chosen on the evaluation set "
                    "is a fitted parameter the baseline bank exists to count. So the reading is a "
                    "floor on what a tuned probe of the same family would recover."
                ),
            ),
            cost=CostModel(note="one forward pass per item plus a ridge solve; no grader calls"),
        )
    )


register_estimators()
