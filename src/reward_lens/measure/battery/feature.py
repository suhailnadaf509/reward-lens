"""``FeatureRewardAlignment`` (E12): which SAE features drive the reward.

A sparse autoencoder decomposes the residual stream into interpretable features whose decoder columns
are directions in activation space. Since the reward is a linear read of that stream, each feature's
contribution to the reward is its decoder column dotted with the reward direction:
``r ≈ b + Σ_i f_i (w_r . d_i)``. The alignment vector ``W_dec @ w_r`` therefore ranks features by how
much, and in which direction, they move the reward. The features at the extremes are the ones a policy
would learn to exploit or avoid.

This ports the alignment computation that lived on v1's ``TopKSAE.feature_reward_alignments``: the
decoder times the reward direction, nothing more. The alignment is a raw-coordinate quantity that
depends on both the SAE basis and the residual-stream basis, so the gauge is RAW_ONLY. A scientific
result needs a trained SAE, which is a separate artifact; when none is supplied the Observable builds
a small randomly-initialized SAE so the alignment mechanics are exercised, and it records that the SAE
was untrained so the number is never read as a real feature ranking.

Where the dictionary comes from, and what happens when there is none. A trained SAE arrives as
``ctx.regime['sae']`` and this module never imports anything to read it: any object with a
``feature_reward_alignments`` method works, so a dictionary fitted elsewhere needs no dependency
here at all. The untrained fallback is built from ``reward_lens.sae.TopKSAE``, which lives behind
the ``[dict]`` extra because sparse dictionaries are candidate generators and never a claim
substrate. On an install without that extra and without a supplied dictionary there is nothing to
align against, so this refuses. It does not quietly return the alignment of something else, and it
does not fabricate a dictionary out of a module it can reach by accident: that dependency was
undeclared, which is the only reason the fallback ever looked free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.extras import ExtraRequiredError
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    reward_direction,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


#: What to do about a missing dictionary, in the two forms a user can act on. Written once because
#: `preflight` and `measure` must say the same thing: a remedy that differs between the dry run and
#: the real one is worse than no dry run.
_NO_DICTIONARY_REMEDY = (
    "Supply a trained sparse autoencoder as ctx.regime['sae'] (any object with a "
    "feature_reward_alignments(w_r) method fitted on this signal's residual stream at the readout "
    "site), or install the dictionary tooling with:  pip install 'reward-lens[dict]'  and rerun to "
    "get the untrained-dictionary mechanics check. The second route does not produce a feature "
    "ranking: it produces the alignment of random directions, reported with trained_sae=False."
)


def _load_topk_sae() -> Any:
    """``TopKSAE``, or the typed error naming the extra that installs it.

    Imported here rather than at module scope so that an install without ``[dict]`` can still
    import the battery, register this instrument, lint it and preflight it. Only the fallback
    construction needs the class, and a caller supplying its own dictionary never reaches this.
    """
    from reward_lens.sae import TopKSAE

    return TopKSAE


class FeatureRewardAlignment(BaseObservable):
    """Per-feature reward alignment of an SAE over a signal (E12).

    The SAE comes from ``ctx.regime['sae']`` (a ``TopKSAE`` whose ``d_model`` matches the signal). When
    absent, a small random SAE is built so the mechanics run and the result is flagged untrained.
    Requires a linear readout. RAW_ONLY: alignments depend on the SAE and residual bases.

    What it cannot do. With no SAE supplied it builds an untrained one, so the default run measures
    the alignment of random directions with the reward and the ``trained_sae`` flag is the only
    thing separating that from a result. Read on its own the alignment is a claim about the decoder
    and not about the policy: a feature with a large ``w_r . d_i`` that never fires contributes
    nothing to any reward anyone will observe, and this instrument does not look at activations, so
    it cannot tell the two apart.
    """

    name = "FeatureRewardAlignment"
    version = "1.0"
    capabilities = Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E12 SAE feature-reward alignment"
    deviations = (
        "alignment is W_dec @ w_r (v1's TopKSAE.feature_reward_alignments); RAW_ONLY; an untrained "
        "random SAE is substituted when none is supplied and the result is flagged accordingly",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.feature_alignment"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Each alignment is an inner product between a decoder column and the readout. Under
    #: `repr.basis` both rotate by the same orthogonal map, so the whole alignment vector, and the
    #: feature ranking read off it, are unchanged. The SAE's own basis is a separate gauge that
    #: no registered group covers, which is what ``GaugeStatus.RAW_ONLY`` records.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_decoder", "baseline.untrained_sae")
    rung = 0

    def preflight(self, ctx: Context) -> PreflightResult:
        """The base checks, plus: is there a dictionary to align against at all?

        Answered here because it is answerable with no compute and no GPU, which is the whole point
        of a preflight. A caller planning a study on a machine without ``[dict]`` finds out now
        rather than after loading a signal.
        """
        result = super().preflight(ctx)
        if not result.ok or ctx.regime.get("sae") is not None:
            return result
        try:
            _load_topk_sae()
        except ExtraRequiredError as exc:
            return PreflightResult(
                instrument=self.name,
                ok=False,
                refusal=Refusal(
                    instrument=self.name,
                    reason=RefusalReason.ACCESS_INSUFFICIENT,
                    detail=(
                        "no sparse dictionary is reachable. ctx.regime carries no 'sae', and the "
                        f"[dict] extra that ships this library's own TopKSAE is not installed: {exc}"
                    ),
                    remedy=_NO_DICTIONARY_REMEDY,
                    statistics={"supplied_sae": False, "dict_extra_installed": False},
                ),
                rung=self.rung,
                unchecked=result.unchecked,
                notes=result.notes,
            )
        return result

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        w_r = reward_direction(signal, ctx.readout)
        d_model = int(signal.meta.d_model)

        sae = ctx.regime.get("sae")
        supplied = sae is not None
        # Whether a supplied dictionary was fitted is the caller's fact, not this instrument's: it
        # sees a decoder matrix either way. The default keeps the documented convention that you do
        # not hand over a dictionary you did not train, and `regime['sae_trained'] = False` is how a
        # caller supplying a random or partially fitted one says so rather than being assumed.
        trained = bool(ctx.regime.get("sae_trained", supplied))
        if sae is None:
            # Raised, not swallowed: `run` does not call `preflight`, so this is the last place a
            # missing dictionary can be reported, and the typed error already names the extra and
            # the pip line. Returning a number here instead is the failure this library exists to
            # prevent.
            TopKSAE = _load_topk_sae()
            n_features = int(ctx.regime.get("n_features", 4 * d_model))
            sae = TopKSAE(d_model=d_model, n_features=n_features, k=min(16, n_features))

        alignments = sae.feature_reward_alignments(w_r.to(torch.float32)).detach().cpu().numpy()
        top_k = int(ctx.regime.get("top_k", 10))
        order = np.argsort(alignments)
        bottom = [(int(i), float(alignments[i])) for i in order[:top_k]]
        top = [(int(i), float(alignments[i])) for i in order[::-1][:top_k]]

        payload = {
            "n_features": int(alignments.shape[0]),
            "trained_sae": trained,
            #: Separated from `trained_sae` because they are different facts and only this one is
            #: observed here. False means the dictionary came from the untrained fallback.
            "supplied_sae": supplied,
            "top_features": top,
            "bottom_features": bottom,
            "max_alignment": float(np.max(alignments)),
            "min_alignment": float(np.min(alignments)),
            "mean_abs_alignment": float(np.mean(np.abs(alignments))),
        }
        return ctx.emit(payload)


__all__ = ["FeatureRewardAlignment"]
