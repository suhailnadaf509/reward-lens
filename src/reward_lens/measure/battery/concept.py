"""``ConceptDoseResponse`` (E08): concept directions, reward alignment, and dose response.

A concept (verbosity, confidence, formality) is a direction in activation space, estimated as the
mean difference between activations that have the concept and activations that do not. Two questions
follow. How aligned is the concept with the reward direction, that is, does having the concept push
reward up or down? And what is the causal dose response, how does the reward move as you steer the
activation along the concept direction? A concept that both aligns with reward and moves it causally
is a reward-hacking lever.

This ports v1's ``ConceptExtractor`` through the canonical concept functions in
:mod:`reward_lens.concepts.vectors`: the direction is the unit-normalized mean difference, the
alignment is its cosine with ``w_r``, and the dose response is the least-squares slope of reward
against steering strength. The concept-pair activations come from the pairs in the view (chosen is the
positive side of the concept, rejected the negative), so the same diagnostic data drives it. The
alignment and dose are raw-coordinate quantities (RAW_ONLY): they depend on the residual-stream basis
and are meaningful within one signal, not across signals without a frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.concepts.vectors import concept_direction, dose_response_slope, reward_alignment
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus, Site
from reward_lens.interventions.patch import ResidualAddPatch, run_patched_scores
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
    pair_sides,
    reward_direction,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class ConceptDoseResponse(BaseObservable):
    """Concept direction, its reward alignment, and its causal dose response (E08).

    Requires activation capture and a linear readout. The concept is read from the pairs in the view
    (chosen positive, rejected negative). Marked RAW_ONLY because the direction, its alignment, and the
    dose are all in raw residual-stream coordinates.

    What it cannot do. The concept direction is whatever separates the chosen side from the rejected
    side of this view, so it is the concept the diagnostic set happens to encode and not the concept
    the caller named: on a view where the chosen side is also longer, this measures length. The dose
    sweep is five points on a fixed span from -2 to +2 in units of the unit direction, fitted with a
    straight line, so a saturating or non-monotone response is reported as its least-squares slope
    with no indication that the fit is wrong. The envelope requires ``ABOVE_LOD`` because the slope
    is a regression on five reward differences.
    """

    name = "ConceptDoseResponse"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E08 concept dose response"
    deviations = (
        "concept direction is the unit-normalized mean difference over the view's pairs; the dose "
        "response steers the final residual and reads the reward slope; RAW_ONLY (basis-dependent)",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.concept_dose_slope"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD | Access.MUTATE}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Both the alignment and the slope are inner products between the concept direction and the
    #: readout, and a shared orthogonal change of basis moves both vectors together, so neither
    #: reading moves. ``GaugeStatus.RAW_ONLY`` above is a different claim and it stands: it says
    #: there is no correspondence between two *different* models' bases, which is a frame question
    #: gate 2 enforces, not a statement about one model under a change of coordinates.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.norm_matched_random", "baseline.random_direction")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is a cosine alignment and a least-squares slope over five doses, both aggregates "
        "over the whole view. Five dose means are not five labelled items, and no black-box method "
        "reads a dose-response slope off text. A record becomes possible given a set of concepts a "
        "grader is known to price and known not to, which would make the alignment a per-concept "
        "verdict; this build has no such labelled concept set.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        readout = ctx.readout
        w_r = reward_direction(signal, readout)
        concept_site = Site(n_layers - 1, "resid_post")

        chosen, rejected = pair_sides(ctx.view)
        pos = capture_sites(signal, chosen, (concept_site,))[concept_site]
        neg = capture_sites(signal, rejected, (concept_site,))[concept_site]
        direction = concept_direction(pos, neg)  # unit vector (d_model,)
        alignment = reward_alignment(direction, w_r)

        # Dose response: steer the final token's residual along the concept direction and read the
        # reward, matching v1's last-token intervene_on_concept. The baseline (dose 0) recovers the
        # clean reward.
        doses = np.linspace(-2.0, 2.0, 5)
        base_items = chosen
        dir_t = torch.tensor(direction, dtype=torch.float32)
        rewards = []
        for dose in doses:
            reward = self._steered_reward(
                signal, base_items, concept_site, dir_t * float(dose), readout
            )
            rewards.append(float(np.mean(reward)))
        slope = dose_response_slope(doses, np.array(rewards))

        payload = {
            "reward_alignment": alignment,
            "dose_response_slope": slope,
            "doses": doses.tolist(),
            "mean_reward_at_dose": rewards,
            "concept_norm": float(np.linalg.norm(direction)),
            "n_pairs": len(chosen),
        }
        return ctx.emit(payload)

    @staticmethod
    def _steered_reward(signal, items, site: Site, vector, readout: str) -> np.ndarray:
        """Reward after adding ``vector`` at the final token of ``site``'s layer output.

        Left padding aligns the final token at the last column for every row, so a delta that is zero
        everywhere except the last column steers exactly the final-token residual, which is where v1's
        ``intervene_on_concept`` added the concept vector.
        """
        import torch

        tokenized = [signal.tokenize(it) for it in items]
        max_t = max(len(t.input_ids) for t in tokenized)
        delta = torch.zeros((1, max_t, vector.shape[-1]), dtype=torch.float32)
        delta[0, -1, :] = vector
        patch = ResidualAddPatch(site=site, delta=delta)
        return run_patched_scores(signal, patch.compile(signal), items, readout)


__all__ = ["ConceptDoseResponse"]
