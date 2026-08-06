"""A17 Eval-awareness: does the reward recognize benchmark inputs.

Formal definition, A17. The balanced accuracy of a probe discriminating benchmark-style from
organic inputs from the reward model's activations, plus the causal ``Δr`` from steering that direction
(does recognition inflate the score?). A reward that can tell a benchmark item from an organic one has a
handle an optimizer can pull, and if steering the recognition direction moves the reward, the reward is
partly scoring "this looks like a test" rather than the response itself (the grader eval-awareness
program, N5/S16).

Deviation from A17: the probe is a held-out linear discriminant (mean-difference direction fit on a
train split, evaluated on a test split) rather than a full calibrated classifier, so the balanced
accuracy is honest about generalization without a heavy dependency; the causal ``Δr`` from steering is
injected here and is the production path through the interventions subsystem. The label-permutation null
(``stats.nulls.shuffle_null``) gives the chance floor the accuracy must beat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    final_activations,
    measure_incremental_validity,
    measured_without_input,
    missing_injection,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Balanced accuracy: the mean of per-class recalls (chance is 0.5 for two classes)."""
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    recalls = []
    for cls in (0, 1):
        mask = true == cls
        if np.any(mask):
            recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def eval_awareness_probe(
    activations: np.ndarray,
    is_benchmark: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out balanced accuracy of a linear benchmark-vs-organic probe (A17).

    Splits the samples in half, fits a mean-difference direction ``μ_bench − μ_organic`` on the train
    half, thresholds projections at the midpoint of the class-mean projections, and reports balanced
    accuracy on the test half. Held-out evaluation is what keeps a random labelling near ``0.5`` instead
    of overfitting to noise. ``activations`` is ``(n, d)``; ``is_benchmark`` is a 0/1 label vector.
    Returns the balanced accuracy and the split sizes.

    ``test_index`` and ``test_score`` are returned beside them because the incremental-validity
    record needs the correlation between this probe's per-item *errors* and a black-box baseline's,
    and no summary can produce that. The index is into the caller's own item order, so the caller can
    line the same items up for the baseline bank; handing back the projections without it would make
    the two methods look paired when they are not.
    """
    a = np.asarray(activations, dtype=np.float64)
    y = np.asarray(is_benchmark).ravel().astype(int)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    tr, te = perm[:half], perm[half:]
    a_tr, y_tr, a_te, y_te = a[tr], y[tr], a[te], y[te]
    empty = np.zeros(0, dtype=np.float64)
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return {
            "balanced_accuracy": float("nan"),
            "n_train": tr.size,
            "n_test": te.size,
            "test_index": np.zeros(0, dtype=int),
            "test_score": empty,
        }
    direction = a_tr[y_tr == 1].mean(axis=0) - a_tr[y_tr == 0].mean(axis=0)
    proj_tr = a_tr @ direction
    threshold = 0.5 * (proj_tr[y_tr == 1].mean() + proj_tr[y_tr == 0].mean())
    proj_te = a_te @ direction
    pred_te = (proj_te > threshold).astype(int)
    return {
        "balanced_accuracy": _balanced_accuracy(pred_te, y_te),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "test_index": te,
        "test_score": proj_te,
    }


class EvalAwareness(BaseObservable):
    """A17 balanced accuracy of a benchmark-vs-organic probe on reward activations, with a null.

    Requires activations. Labels marking which inputs are benchmark-style are injected (the data plane
    supplies them in production); the causal ``Δr`` from steering the recognition direction is injected
    and is the production path. Reports the held-out balanced accuracy against a label-permutation null.
    Gauge is INVARIANT: balanced accuracy is a within-signal classification quality.

    What it cannot do. A probe that separates benchmark from organic inputs shows the
    information is present in the activations, which is a much weaker claim than the reward using
    it: high balanced accuracy is compatible with a reward that scores the two identically. The
    causal half of A17, the reward change from steering the recognition direction, is injected here
    and not measured, so a run with ``steering_delta_r`` unset reports the correlational half alone.
    The probe is also a mean-difference discriminant fit on one split, so any surface feature that
    happens to track the benchmark label, formatting or length, reads as recognition.
    """

    name = "EvalAwareness"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A17"
    deviations = (
        "probe is a held-out mean-difference linear discriminant, not a full calibrated classifier",
        "the causal delta-r from steering the recognition direction is injected and is the production "
        "path (interventions)",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.eval_awareness"
    #: Activations are captured from the grader; the benchmark/organic labels and the steering delta
    #: come from the data plane and from an earlier intervention arm.
    requires: AccessMatrix = {
        Component.GRADER: Access.FORWARD,
        Component.RECORD: Access.RECORD,
    }
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A linear probe fitted on rotated activations recovers the rotated direction, so the balanced
    #: accuracy is unchanged under a shared orthogonal change of basis.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.label_permutation_null", "baseline.length")
    rung = 0

    def __init__(
        self,
        is_benchmark: np.ndarray | None = None,
        *,
        steering_delta_r: float | None = None,
        null_draws: int = 2000,
        seed: int = 0,
    ) -> None:
        self.is_benchmark = is_benchmark
        self.steering_delta_r = steering_delta_r
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def preflight(self, ctx: Context) -> PreflightResult:
        """The labels or a refusal. A probe with no labels is not a probe.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.is_benchmark is None:
            return missing_injection(
                self,
                needs={
                    "is_benchmark": "an (n,) 0/1 label per sample, 1 for benchmark and 0 for organic",
                },
                have="no labels were injected",
                remedy=(
                    "Construct `EvalAwareness(is_benchmark=...)` with one label per sample in the view, 1 "
                    "for a benchmark prompt and 0 for organic traffic. The probe is fit on half the samples "
                    "and scored on the other half, so both classes need enough members to appear in both "
                    "halves."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.is_benchmark is None:
            raise measured_without_input(self)
        acts = final_activations(ctx.signal, ctx.view, readout=ctx.readout)
        y = np.asarray(self.is_benchmark).ravel().astype(int)
        probe = eval_awareness_probe(acts, y, seed=self.seed)

        from reward_lens.stats.nulls import shuffle_null

        def _stat(vals: np.ndarray, labels: np.ndarray) -> float:
            return eval_awareness_probe(vals, labels, seed=self.seed)["balanced_accuracy"]

        null = shuffle_null(acts, y, _stat, n=min(self.null_draws, 500), seed=self.seed)

        # This reading opens the network, so it owes an account of what that bought over the
        # black-box bank on the same items. Only the held-out half is eligible, because the
        # probe's train half is in-sample for the probe and not for the baselines, and comparing the
        # two there would hand the probe a fitted advantage the record has no field to declare.
        record, notes = self._incremental(ctx, probe, y)

        payload = {
            "balanced_accuracy": probe["balanced_accuracy"],
            "null_mean": null["null_mean"],
            "null_p_value": null["p_value"],
            "steering_delta_r": self.steering_delta_r,
            "n_items": int(acts.shape[0]),
            "incremental": record.__canonical__() if record is not None else None,
            "incremental_notes": notes,
        }
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"),
            incremental=record,
        )

    def _incremental(self, ctx: Context, probe: dict, y: np.ndarray) -> tuple[Any, dict]:
        """The incremental record on the probe's held-out half, or None with why it is absent.

        The texts come from the view rather than from the caller, because the whole question is
        whether a method that read the activations beat one that read only the text, and a bank run
        on anything but the same strings is answering a different question. A view that carries no
        text is the `NO_BLACK_BOX_ON_THESE_ITEMS` case arriving at run time rather than at
        declaration time, and it is reported rather than skipped.
        """
        te = np.asarray(probe.get("test_index", ()), dtype=int)
        if te.size == 0:
            return None, {
                "refused": (
                    "the split left one class empty on one side, so the probe has no held-out "
                    "scores. Supply more items of the minority class: the probe fits on half and "
                    "is scored on the other half, so both classes need members in both halves."
                )
            }
        texts = _view_texts(ctx.view)
        if len(texts) != y.size:
            return None, {
                "refused": (
                    f"the view yielded {len(texts)} readable strings for {y.size} items, so the "
                    f"black-box bank cannot be run on the same items the probe was. Pass a view "
                    f"whose items expose their text, or record this reading as having no black-box "
                    f"comparison rather than comparing against a different item set."
                )
            }
        return measure_incremental_validity(
            "index.eval_awareness_probe",
            probe["test_score"],
            y[te],
            texts=tuple(texts[i] for i in te),
            seed=self.seed,
        )


def _view_texts(view: Any) -> list[str]:
    """The one string per item the black-box bank reads, in the view's own order.

    Four shapes reach an index's view in this library and only the first three carry text: a bare
    string, a ``(prompt, response)`` pair, and a data-plane item exposing ``.text``. Anything else
    yields nothing and the caller reports that rather than inventing a string for it, because a bank
    run on a placeholder returns a floor that looks like a measurement.
    """
    out: list[str] = []
    for item in view or ():
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (tuple, list)) and item and all(isinstance(p, str) for p in item):
            out.append(str(item[-1]))
        elif isinstance(getattr(item, "text", None), str):
            out.append(item.text)
        else:
            return []
    return out


__all__ = ["eval_awareness_probe", "EvalAwareness"]
