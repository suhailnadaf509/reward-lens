"""``SignalEnsemble`` and distributional wrappers (adapter 8).

Two kinds of composite live here, both of which any Observable consumes as an ordinary ``RewardSignal``:

  - ``SignalEnsemble`` combines several member signals into one score (mean / min / quantile), keeping
    every member's fingerprint in the Evidence subject so the composite's provenance names exactly what
    it was built from. Min-of-ensemble is the standard conservative reward-hacking guard; the quantile
    composite is its tunable generalization.
  - ``DistributionalSignal`` wraps a signal whose head rows are quantile levels (a QRM) and exposes them
    as ``quantile:tau`` readouts with ``DISTRIBUTIONAL`` declared, plus median / quantile / mean
    reductions. It turns a multi-row head into a first-class predictive distribution over the reward.

Both are thin: they delegate the actual forward work to their members and only compose scores, which is
why they inherit the members' hardening for free rather than re-deriving it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.types import Capability, GaugeStatus, ModelFP, SubjectRef, content_hash
from reward_lens.signals.base import Scores, SignalMeta, TokenCurves

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

_OBS_VERSION = "1"

EnsembleMode = Literal["mean", "min", "max", "quantile"]


class SignalEnsemble:
    """A composite of several member signals (adapter 8).

    ``score`` scores every member under the requested readout, stacks the results, and reduces them by
    ``mode`` (mean, min, max, or a quantile at ``q``). The Evidence subject names every member by
    fingerprint (member provenance), and the ensemble's own fingerprint is derived from the members'
    so two ensembles over the same members are the same subject. ``caps`` is the intersection of the
    members' capabilities, plus ``DISTRIBUTIONAL`` for a quantile composite.
    """

    observable_prefix = "signals.ensemble"

    def __init__(
        self,
        members: Sequence[Any],
        *,
        mode: EnsembleMode = "mean",
        q: float = 0.5,
        name: str | None = None,
    ) -> None:
        if not members:
            raise ValueError("a SignalEnsemble needs at least one member signal")
        self.members = list(members)
        self.mode = mode
        self.q = q
        self.runtime = self.members[0].runtime  # nominal; capture delegates to a member explicitly
        # An ensemble reliably composes scores and (per member) activations; it does not compose the
        # prefix, gradient, or Hessian paths (which member would "layer L" name?), so it claims only
        # what it actually implements rather than the members' full intersection.
        caps = Capability.SCORES
        if all(bool(m.caps & Capability.ACTIVATIONS) for m in self.members):
            caps = caps | Capability.ACTIVATIONS
        if all(bool(m.caps & Capability.PREFIX_SCORES) for m in self.members):
            caps = caps | Capability.PREFIX_SCORES
        if mode == "quantile":
            caps = caps | Capability.DISTRIBUTIONAL
        self.caps = caps
        member_fps = [str(m.meta.fingerprint) for m in self.members]
        fp = ModelFP(content_hash({"ensemble": member_fps, "mode": mode, "q": q}, "mfp"))
        self.meta = SignalMeta(
            fingerprint=fp,
            adapter="SignalEnsemble",
            architecture="ensemble",
            lineage={
                "members": member_fps,
                "member_adapters": [m.meta.adapter for m in self.members],
                "mode": mode,
                "q": q,
            },
            n_layers=self.members[0].meta.n_layers,
            d_model=self.members[0].meta.d_model,
        )
        self.name = name

    # -- readouts (the readouts common to every member) --------------------

    def readouts(self) -> list[Any]:
        """The readouts every member exposes (the composite can only score a shared readout)."""
        common = set.intersection(*({r.name for r in m.readouts()} for m in self.members))
        return [r for r in self.members[0].readouts() if r.name in common]

    def tokenize(self, item: Any) -> Any:
        """Tokenize via the first member (members share the readout; tokenization is member 0's)."""
        return self.members[0].tokenize(item)

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str | None = None) -> "Evidence[Scores]":
        """Composite score under ``mode`` over the members' scores."""
        items = list(view)
        started = time.perf_counter()
        name = readout or self._default_readout_name()
        member_values = np.stack(
            [np.asarray(m.score(items, name).value.values, dtype=np.float64) for m in self.members]
        )  # (n_members, n_items)
        composite = self._reduce(member_values).astype(np.float32)
        payload = Scores(values=composite, readout=name, n_items=len(items))
        subject = SubjectRef(
            signals=tuple(m.meta.fingerprint for m in self.members),
            readout=name,
            extra={
                "ensemble_mode": self.mode,
                "q": self.q,
                "ensemble_fp": str(self.meta.fingerprint),
            },
        )
        provenance = capture_provenance(
            cost=Cost(wall_seconds=time.perf_counter() - started),
            extra={"n_members": len(self.members)},
        )
        return make_evidence(
            observable=f"{self.observable_prefix}.score",
            observable_version=_OBS_VERSION,
            subject=subject,
            value=payload,
            uncertainty=Uncertainty(n=len(items), method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,
            provenance=provenance,
        )

    def score_prefixes(self, view: Any, readout: str | None = None) -> "Evidence[TokenCurves]":
        """Composite per-token curves: the members' prefix curves reduced token by token.

        Requires the members to tokenize identically (the common case: a shared tokenizer), so the
        per-item curves align in length and reduce position-wise. Prefix consistency is preserved
        because ``mean``/``min``/``max``/``quantile`` of the members' final entries equals the composite
        score. Misaligned curve lengths raise rather than silently truncate.
        """
        items = list(view)
        started = time.perf_counter()
        name = readout or self._default_readout_name()
        member_curves = [m.score_prefixes(items, name).value.curves for m in self.members]
        curves: list[np.ndarray] = []
        for i in range(len(items)):
            lengths = {len(mc[i]) for mc in member_curves}
            if len(lengths) != 1:
                raise ValueError(
                    "ensemble members produced prefix curves of differing lengths; a composite "
                    "prefix curve requires the members to share a tokenization."
                )
            stacked = np.stack([np.asarray(mc[i], dtype=np.float64) for mc in member_curves])
            curves.append(self._reduce(stacked).astype(np.float32))
        payload = TokenCurves(curves=curves, readout=name)
        subject = SubjectRef(
            signals=tuple(m.meta.fingerprint for m in self.members),
            readout=name,
            extra={"ensemble_mode": self.mode, "q": self.q},
        )
        provenance = capture_provenance(cost=Cost(wall_seconds=time.perf_counter() - started))
        return make_evidence(
            observable=f"{self.observable_prefix}.score_prefixes",
            observable_version=_OBS_VERSION,
            subject=subject,
            value=payload,
            uncertainty=Uncertainty(n=len(items), method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,
            provenance=provenance,
        )

    def member_scores(self, view: Any, readout: str | None = None) -> dict[ModelFP, np.ndarray]:
        """The per-member score arrays keyed by member fingerprint (for provenance and diagnostics)."""
        name = readout or self._default_readout_name()
        return {m.meta.fingerprint: m.score(view, name).value.values for m in self.members}

    def capture(self, view: Any, spec: Any, member: int = 0) -> Any:
        """Capture from a named member (default the first); an ensemble has no shared activation."""
        return self.members[member].capture(view, spec)

    def with_interventions(self, *ivs: Any) -> "SignalEnsemble":
        """Wrap every member in the interventions; the composite subject carries them per member."""
        return SignalEnsemble(
            [m.with_interventions(*ivs) for m in self.members],
            mode=self.mode,
            q=self.q,
            name=self.name,
        )

    def _default_readout_name(self) -> str:
        shared = self.readouts()
        if not shared:
            raise ValueError("ensemble members share no common readout to score")
        return shared[0].name

    def _reduce(self, values: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            return values.mean(axis=0)
        if self.mode == "min":
            return values.min(axis=0)
        if self.mode == "max":
            return values.max(axis=0)
        if self.mode == "quantile":
            return np.quantile(values, self.q, axis=0)
        raise ValueError(f"unknown ensemble mode {self.mode!r}")


class DistributionalSignal:
    """A distributional wrapper exposing a multi-row head's rows as quantile readouts (adapter 8).

    Wraps a signal whose head rows are quantile levels (a QRM) and presents them as ``quantile:tau``
    readouts, declaring ``DISTRIBUTIONAL``. ``score(view, "quantile:0.9")`` returns the 0.9-quantile
    row; ``median``, ``quantile``, and ``mean`` are convenience reductions over the levels. The wrapped
    signal does the forward work; this class only relabels and reduces.
    """

    observable_prefix = "signals.distributional"

    def __init__(self, signal: Any, taus: Sequence[float], row_readouts: Sequence[str]) -> None:
        if len(taus) != len(row_readouts):
            raise ValueError("taus and row_readouts must line up one-to-one")
        self.signal = signal
        self.taus = tuple(float(t) for t in taus)
        self._row_by_tau = {float(t): r for t, r in zip(taus, row_readouts)}
        self.runtime = signal.runtime
        self.caps = signal.caps | Capability.DISTRIBUTIONAL
        base: SignalMeta = signal.meta
        self.meta = SignalMeta(
            fingerprint=base.fingerprint,
            adapter="DistributionalSignal",
            architecture=base.architecture,
            lineage={**base.lineage, "wraps": base.adapter, "quantile_levels": list(self.taus)},
            template=base.template,
            numerics_policy=base.numerics_policy,
            soft_cap=base.soft_cap,
            d_model=base.d_model,
            n_layers=base.n_layers,
            n_heads=base.n_heads,
        )

    def readouts(self) -> list[Any]:
        """One ``quantile:tau`` readout per level, reusing the wrapped row vectors."""
        from dataclasses import replace

        out = []
        for tau in self.taus:
            base_read = self.signal.readout(self._row_by_tau[tau])
            out.append(
                replace(base_read, name=f"quantile:{tau}", meta={**base_read.meta, "tau": tau})
            )
        return out

    def tokenize(self, item: Any) -> Any:
        return self.signal.tokenize(item)

    def score(self, view: Any, readout: str | None = None) -> Any:
        """Score at a quantile level: ``quantile:tau`` maps to the wrapped row for that level."""
        name = readout or f"quantile:{self._median_tau()}"
        if name.startswith("quantile:"):
            tau = float(name.split(":", 1)[1])
            return self.signal.score(view, self._row_by_tau[tau])
        return self.signal.score(view, name)

    def score_prefixes(self, view: Any, readout: str | None = None) -> Any:
        """Per-token curve at a quantile level (delegates to the wrapped row's prefix curve)."""
        name = readout or f"quantile:{self._median_tau()}"
        row = (
            self._row_by_tau[float(name.split(":", 1)[1])] if name.startswith("quantile:") else name
        )
        return self.signal.score_prefixes(view, row)

    def quantile(self, view: Any, tau: float) -> Any:
        """The reward at quantile level ``tau`` (an exact head row, not an interpolation)."""
        return self.signal.score(view, self._row_by_tau[float(tau)])

    def median(self, view: Any) -> Any:
        """The median-level reward (the row closest to tau=0.5)."""
        return self.signal.score(view, self._row_by_tau[self._median_tau()])

    def mean(self, view: Any) -> "Evidence[Scores]":
        """The mean reward across the quantile levels (a point estimate of the distribution)."""
        started = time.perf_counter()
        items = list(view)
        rows = np.stack(
            [
                np.asarray(
                    self.signal.score(items, self._row_by_tau[t]).value.values, dtype=np.float64
                )
                for t in self.taus
            ]
        )
        values = rows.mean(axis=0).astype(np.float32)
        payload = Scores(values=values, readout="quantile:mean", n_items=len(items))
        subject = SubjectRef(
            signals=(self.meta.fingerprint,),
            readout="quantile:mean",
            extra={"quantile_levels": list(self.taus)},
        )
        provenance = capture_provenance(cost=Cost(wall_seconds=time.perf_counter() - started))
        return make_evidence(
            observable=f"{self.observable_prefix}.mean",
            observable_version=_OBS_VERSION,
            subject=subject,
            value=payload,
            uncertainty=Uncertainty(n=len(items), method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,
            provenance=provenance,
        )

    def capture(self, view: Any, spec: Any) -> Any:
        return self.signal.capture(view, spec)

    def with_interventions(self, *ivs: Any) -> "DistributionalSignal":
        return DistributionalSignal(
            self.signal.with_interventions(*ivs),
            self.taus,
            [self._row_by_tau[t] for t in self.taus],
        )

    def _median_tau(self) -> float:
        return min(self.taus, key=lambda t: abs(t - 0.5))

    @classmethod
    def from_tiny(
        cls, *, taus: Sequence[float] = (0.1, 0.5, 0.9), seed: int = 0, **kw: Any
    ) -> "DistributionalSignal":
        """A tiny QRM: a multi-row classifier wrapped and relabeled with quantile levels (adapter 8)."""
        from reward_lens.signals.loaders import wrap_hf_model
        from reward_lens.signals.process import _tiny_sequence_classifier

        model, tokenizer = _tiny_sequence_classifier(seed=seed, num_labels=len(taus), **kw)
        base = wrap_hf_model(
            model,
            tokenizer,
            architecture="LlamaForSequenceClassification",
            conformance_quickcheck=False,
        )
        row_readouts = [f"criterion:{k}" for k in range(len(taus))]
        return cls(base, taus, row_readouts)


__all__ = ["SignalEnsemble", "DistributionalSignal", "EnsembleMode"]
