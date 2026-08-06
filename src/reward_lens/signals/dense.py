"""``DenseRewardExtractor``: per-token reward maps from an outcome signal (adapter 7).

A dense reward map assigns credit token by token from a signal that only scores the whole response. The
construction is differential attribution along the prefix-score curve: if r(y_{1:t}) is the outcome score
of the prefix ending at token t (which every signal exposes via ``score_prefixes``), then the marginal
reward of token t is r(y_{1:t}) - r(y_{1:t-1}). The per-token map is the first difference of the prefix
curve, and it sums back to the outcome score by telescoping.

This adapter ships GATED, and the gating is the point: a dense map looks authoritative and
is easy to over-trust, so its Evidence is pinned at EXPLORATORY until the verification science (S6/S9)
certifies it against labeled error spans and issues a scorecard entry. The enforcement is structural and
deliberate: this adapter attaches no calibration reference, ever, so the gates cannot rate it above
EXPLORATORY. That ordering (the product ships only after its answer-key validation) is exemplary for the
whole design, and it is expressed here as the simple refusal to fabricate a calibration.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.signals.base import SignalMeta, TokenCurves

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

_OBS_VERSION = "1"


class DenseRewardExtractor:
    """Per-token reward maps from any outcome signal, shipped GATED (adapter 7).

    Wraps an outcome ``RewardSignal`` and adds ``dense_rewards``: the first difference of the wrapped
    signal's prefix-score curve, a per-token attribution that sums to the outcome score. Every other
    protocol method delegates to the wrapped signal, so the same battery reaches the dense extractor
    unchanged. ``dense_rewards`` Evidence is always EXPLORATORY (no calibration is ever attached),
    which is how the design enforces "certify before you trust" for dense credit assignment.
    """

    observable_prefix = "signals.dense"

    def __init__(self, signal: Any, *, readout: str | None = None) -> None:
        self.signal = signal
        self.runtime = signal.runtime
        self.caps = signal.caps
        self._default_readout = readout or signal.readouts()[0].name
        base: SignalMeta = signal.meta
        self.meta = SignalMeta(
            fingerprint=base.fingerprint,
            adapter="DenseRewardExtractor",
            architecture=base.architecture,
            lineage={
                **base.lineage,
                "wraps": base.adapter,
                "gated": True,
                "evidence_tier": "EXPLORATORY-until-S6/S9-verification",
            },
            template=base.template,
            numerics_policy=base.numerics_policy,
            soft_cap=base.soft_cap,
            d_model=base.d_model,
            n_layers=base.n_layers,
            n_heads=base.n_heads,
        )

    # -- delegated protocol surface ----------------------------------------

    def readouts(self) -> Any:
        """The wrapped signal's readouts (the dense map is computed per outcome readout)."""
        return self.signal.readouts()

    def tokenize(self, item: Any) -> Any:
        return self.signal.tokenize(item)

    def score(self, view: Any, readout: str | None = None) -> Any:
        """The wrapped outcome score (delegated); the dense map is ``dense_rewards``."""
        return self.signal.score(view, readout or self._default_readout)

    def score_prefixes(self, view: Any, readout: str | None = None) -> Any:
        return self.signal.score_prefixes(view, readout or self._default_readout)

    def capture(self, view: Any, spec: Any) -> Any:
        return self.signal.capture(view, spec)

    def with_interventions(self, *ivs: Any) -> "DenseRewardExtractor":
        """Wrap the intervened outcome signal; the dense map inherits the intervention subject."""
        return DenseRewardExtractor(
            self.signal.with_interventions(*ivs), readout=self._default_readout
        )

    # -- the dense map ------------------------------------------------------

    def dense_rewards(self, view: Any, readout: str | None = None) -> "Evidence[TokenCurves]":
        """Per-token reward maps by differential attribution along the prefix curve.

        Each item's map is the first difference of the wrapped signal's prefix-score curve, so token
        t carries r(y_{1:t}) - r(y_{1:t-1}) and the map sums to the outcome score. The Evidence is
        typed INVARIANT (differences of raw scores are gauge-free) but pinned EXPLORATORY: no
        calibration reference is attached, by construction, until the verification science certifies
        the map against labeled error spans. The prefix Evidence is recorded as a provenance parent.
        """
        name = readout or self._default_readout
        started = time.perf_counter()
        prefix_ev = self.signal.score_prefixes(view, name)
        curves = prefix_ev.value.curves
        maps = [
            np.diff(np.asarray(c, dtype=np.float64), prepend=0.0).astype(np.float32) for c in curves
        ]
        payload = TokenCurves(curves=maps, readout=name)
        subject = SubjectRef(
            signals=(self.meta.fingerprint,),
            readout=name,
            interventions=prefix_ev.subject.interventions,
            extra={"method": "differential_prefix_attribution", "gated": True},
        )
        provenance = capture_provenance(
            cost=Cost(
                tokens=int(sum(len(m) for m in maps)), wall_seconds=time.perf_counter() - started
            ),
            parents=(prefix_ev.id,),
        )
        return make_evidence(
            observable="signals.dense.dense_rewards",
            observable_version=_OBS_VERSION,
            subject=subject,
            value=payload,
            uncertainty=Uncertainty(n=len(maps), method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,  # deliberate: dense maps are EXPLORATORY until S6/S9 certification.
            provenance=provenance,
        )


__all__ = ["DenseRewardExtractor"]
