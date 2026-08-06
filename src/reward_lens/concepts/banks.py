"""Standard concept batteries: named, persisted collections of directions.

A bank is a named collection of concept specifications the library ships so a study does not
reinvent "the style directions" or "the safety directions" each time. Four batteries are defined
here over the controlled feature substrate the organism foundry plants against
(`organisms._features`): style (hedging, politeness, structure, verbosity), safety (the unsafe-content
direction), quality (grounding and citation), and belief targets (grounded-truth latents held to the
belief-probe standard). Naming them against the planted substrate is deliberate: every concept in a
bank has an exact ground-truth marker, so a direction extracted for it can be checked, not just
asserted.

Extraction reuses the canonical mean-difference estimator (`concepts.vectors.concept_direction`), so
a bank direction and a v1 concept direction and the dose-response Observable all compute "the concept
direction" the same way. Each extracted direction is a persisted `Direction` so it is
a first-class store citizen with its training data and (where a probe grades it) its calibration. The
bank as a whole is persisted as a manifest naming its directions, so a named battery is reconstructible
from the store.

A built bank satisfies the `FeatureBank` protocol the index library consumes (names, ``featurize``,
``directions``), which is the seam `measure.indices` looks for via ``default_feature_bank``: once the
package wires ``reward_lens.concepts.default_feature_bank`` to the factory here, the susceptibility and
knowledge-utilization indices get real learned features instead of the synthetic stand-in.

The synthetic-capture path is pure numpy and proven on CPU; the tiny-model capture path is
torch-gated and imported lazily, and reads activations off a real signal the same way the battery does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.concepts.probes import (
    Direction,
    direction_evidence,
    make_direction,
)
from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import capture_provenance
from reward_lens.core.types import DatasetID, GaugeStatus, Site, SubjectRef

if TYPE_CHECKING:
    from reward_lens.core.store import EvidenceStore
    from reward_lens.signals.base import RewardSignal

_BANK_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Concept specifications and the standard batteries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptSpec:
    """One concept in a bank, tied to an exact feature marker.

    ``name`` is the concept; ``feature`` is the controlled marker in `organisms._features.FEATURE_MARK`
    whose presence defines the positive side, so the concept has exact ground truth. ``category`` is
    the battery it belongs to (``"style"``, ``"safety"``, ``"quality"``, ``"belief"``). ``gloss`` is
    the human-readable meaning that renders on a card. ``verifiable`` marks a belief target: a concept
    whose positive side is an externally verifiable latent, which the belief-probe factory holds to the
    strictest calibration standard.
    """

    name: str
    feature: str
    category: str
    gloss: str
    verifiable: bool = False


STYLE_BANK: tuple[ConceptSpec, ...] = (
    ConceptSpec("hedged", "hedged", "style", "the response hedges instead of committing"),
    ConceptSpec("polite", "polite", "style", "the response is written politely"),
    ConceptSpec("structured", "structured", "style", "the response is structured as a list"),
    ConceptSpec("detailed", "detailed", "style", "the response is elaborated and verbose"),
)

SAFETY_BANK: tuple[ConceptSpec, ...] = (
    ConceptSpec("unsafe", "unsafe", "safety", "the response carries unsafe content"),
)

QUALITY_BANK: tuple[ConceptSpec, ...] = (
    ConceptSpec("factual", "factual", "quality", "the response makes a grounded factual claim"),
    ConceptSpec("cites", "cites", "quality", "the response cites a source"),
)

BELIEF_BANK: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        "grounded_truth",
        "factual",
        "belief",
        "the response's claim is grounded and verifiably true",
        verifiable=True,
    ),
)

STANDARD_BANKS: dict[str, tuple[ConceptSpec, ...]] = {
    "style": STYLE_BANK,
    "safety": SAFETY_BANK,
    "quality": QUALITY_BANK,
    "belief": BELIEF_BANK,
}


def bank(name: str) -> tuple[ConceptSpec, ...]:
    """The concept specs of a named standard battery (``style``/``safety``/``quality``/``belief``)."""
    if name not in STANDARD_BANKS:
        raise KeyError(f"unknown bank {name!r}; known banks: {sorted(STANDARD_BANKS)}")
    return STANDARD_BANKS[name]


def all_specs() -> tuple[ConceptSpec, ...]:
    """Every concept spec across all standard batteries, in battery then declared order."""
    out: list[ConceptSpec] = []
    for specs in STANDARD_BANKS.values():
        out.extend(specs)
    return tuple(out)


# ---------------------------------------------------------------------------
# The built bank (satisfies the FeatureBank protocol)
# ---------------------------------------------------------------------------


@dataclass
class ConceptBank:
    """A built bank of extracted directions that satisfies the `FeatureBank` protocol.

    ``names`` labels the ``k`` concepts; ``directions_`` is their ``(k, d)`` stacked unit directions;
    ``site`` is where they were read. ``featurize`` projects activations onto the directions
    (``activations @ directions_.T``), the same linear readout the index library's `LinearFeatureBank`
    provides, so a production bank is a drop-in for the synthetic one. ``directions`` exposes the
    decoder matrix. The `Direction` objects (with their ids, training data, and calibration) are kept
    on ``entries`` so the bank is not just a matrix but a set of provenance-carrying artifacts.
    """

    names: tuple[str, ...]
    directions_: np.ndarray
    site: Site
    entries: tuple[Direction, ...] = ()
    category: str = "mixed"

    def __post_init__(self) -> None:
        self.directions_ = np.asarray(self.directions_, dtype=np.float64)
        if self.directions_.ndim != 2:
            raise ValueError(f"directions must be (k, d); got shape {self.directions_.shape}")
        if len(self.names) != self.directions_.shape[0]:
            raise ValueError(
                f"names has {len(self.names)} entries but there are "
                f"{self.directions_.shape[0]} directions"
            )

    def featurize(self, activations: np.ndarray) -> np.ndarray:
        """Project ``(n, d)`` activations onto the bank directions, returning ``(n, k)`` features."""
        a = np.asarray(activations, dtype=np.float64)
        return a @ self.directions_.T

    def directions(self) -> np.ndarray | None:
        """The ``(k, d)`` decoder directions of the bank."""
        return self.directions_


@register_payload
@dataclass
class BankManifest:
    """The persisted manifest of a bank: its name and the ids of its directions.

    Persisting the manifest (rather than re-storing the vectors, which the `Direction` Evidence
    already holds) makes a named battery reconstructible from the store: load the manifest, then load
    each direction by id. ``direction_ids`` are the `DirectionID`s, ``names`` the concept names in the
    same order, and ``category`` the battery.
    """

    bank_name: str
    category: str
    names: list[str]
    direction_ids: list[str]
    site: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction (reuses concept_direction)
# ---------------------------------------------------------------------------

# A concept's captures: the positive-side and negative-side activation matrices at a site. Both are
# (n_i, d); n_i may differ between concepts. This is what the mean-difference estimator consumes.
ConceptSides = dict[str, "tuple[np.ndarray, np.ndarray]"]


def _mean_diff_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """The unit-normalized mean-difference direction, via `concepts.vectors.concept_direction`.

    Reuses the canonical estimator so a bank direction is computed the same way as every other concept
    direction in the library. torch is imported lazily here only to hand the estimator its expected
    tensor inputs; the return is numpy fp32.
    """
    import torch

    from reward_lens.concepts.vectors import concept_direction

    pos_t = torch.tensor(np.asarray(pos, dtype=np.float32))
    neg_t = torch.tensor(np.asarray(neg, dtype=np.float32))
    return np.asarray(concept_direction(pos_t, neg_t), dtype=np.float32)


@dataclass(frozen=True)
class BuiltBank:
    """A built bank with its persisted directions and manifest.

    ``bank`` satisfies the `FeatureBank` protocol; ``directions`` are the persisted `Direction`
    artifacts; ``evidence`` is their stored Evidence plus the manifest Evidence (last). ``manifest`` is
    the reconstructible record of the battery.
    """

    bank: ConceptBank
    directions: tuple[Direction, ...]
    evidence: tuple[Evidence, ...]
    manifest: BankManifest


def build_bank(
    specs: tuple[ConceptSpec, ...],
    sides: ConceptSides,
    site: Site,
    *,
    bank_name: str = "bank",
    method: str = "contrast_mean",
    train_data: DatasetID | None = None,
    store: "EvidenceStore | None" = None,
    signals: tuple[str, ...] = (),
) -> BuiltBank:
    """Extract and persist a bank's directions from per-concept captured sides.

    For each spec, ``sides[spec.name]`` provides the ``(positive, negative)`` activation matrices at
    ``site``; the direction is their unit-normalized mean difference (`concept_direction`), persisted
    as a `Direction` with ``method="contrast_mean"`` and no calibration (a mean-difference direction is
    not graded against an answer key; use `concepts.probes.fit_probe` when a scorecard is wanted). The
    directions are stacked into a `ConceptBank` and a `BankManifest` is persisted naming them.

    A spec with no captured sides is skipped rather than fabricated, so a partially captured bank is
    honestly smaller rather than padded with zeros. Raises if no spec could be built.
    """
    directions: list[Direction] = []
    evidence: list[Evidence] = []
    category = specs[0].category if specs else "mixed"
    for spec in specs:
        if spec.name not in sides:
            continue
        pos, neg = sides[spec.name]
        vec = _mean_diff_direction(pos, neg)
        direction = make_direction(
            name=spec.name,
            site=site,
            vector=vec,
            method=method,
            train_data=train_data,
            calibration=None,
            meta={
                "category": spec.category,
                "gloss": spec.gloss,
                "verifiable": spec.verifiable,
                "n_pos": int(np.asarray(pos).shape[0]),
                "n_neg": int(np.asarray(neg).shape[0]),
            },
        )
        directions.append(direction)
        ev = direction_evidence(direction, signals=signals)
        evidence.append(ev)

    if not directions:
        raise ValueError(
            f"build_bank({bank_name!r}): no concept had captured sides; nothing to extract"
        )

    names = tuple(d.name for d in directions)
    matrix = np.stack([np.asarray(d.vector, dtype=np.float64) for d in directions])
    concept_bank = ConceptBank(
        names=names,
        directions_=matrix,
        site=site,
        entries=tuple(directions),
        category=category,
    )
    manifest = BankManifest(
        bank_name=bank_name,
        category=category,
        names=list(names),
        direction_ids=[str(d.id) for d in directions],
        site=site.__canonical__(),
        meta={"method": method, "n_directions": len(directions)},
    )
    manifest_ev = _manifest_evidence(manifest, train_data=train_data, signals=signals)
    evidence.append(manifest_ev)

    if store is not None:
        for ev in evidence:
            store.append(ev)

    return BuiltBank(
        bank=concept_bank,
        directions=tuple(directions),
        evidence=tuple(evidence),
        manifest=manifest,
    )


def _manifest_evidence(
    manifest: BankManifest,
    *,
    train_data: DatasetID | None,
    signals: tuple[str, ...],
) -> Evidence:
    """Wrap a `BankManifest` as INVARIANT Evidence (the manifest is a set of names and ids)."""
    subject = SubjectRef(
        signals=tuple(signals),
        dataset=train_data,
        readout=f"bank:{manifest.bank_name}",
        extra={"kind": "concept-bank-manifest", "category": manifest.category},
    )
    provenance = capture_provenance(config={"bank": manifest.bank_name})
    return make_evidence(
        observable=f"ConceptBank[{manifest.bank_name}]",
        observable_version=_BANK_VERSION,
        subject=subject,
        value=manifest,
        uncertainty=Uncertainty(n=len(manifest.names), method="bank-manifest"),
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# The tiny-model / signal capture path (torch-gated)
# ---------------------------------------------------------------------------


def capture_concept_sides(
    signal: "RewardSignal",
    specs: tuple[ConceptSpec, ...],
    site: Site,
    *,
    topics: tuple[str, ...] | None = None,
    n_per_side: int = 8,
) -> ConceptSides:
    """Capture positive/negative activation sides for each concept off a real signal (torch-gated).

    For each spec, builds ``n_per_side`` positive texts carrying the concept's marker and the same
    number of negative texts without it (over a set of topics), reading the final-token activation at
    ``site`` for both. This is the mean-difference stimulus the bank extracts from, produced from the
    controlled feature substrate so the positive and negative sides differ exactly in the concept.
    torch enters only through the shared capture helper, imported lazily.
    """
    from reward_lens.measure.battery._common import capture_sites
    from reward_lens.organisms._features import OOD_TOPICS, TRAIN_TOPICS, render_response

    topic_pool = topics or (TRAIN_TOPICS + OOD_TOPICS)
    sides: ConceptSides = {}
    for spec in specs:
        pos_texts: list[Any] = []
        neg_texts: list[Any] = []
        for i in range(n_per_side):
            topic = topic_pool[i % len(topic_pool)]
            prompt = f"What can you tell me about {topic}?"
            pos_texts.append((prompt, render_response(topic, {spec.feature})))
            neg_texts.append((prompt, render_response(topic, set())))
        pos = capture_sites(signal, pos_texts, (site,), dtype="float32")[site]
        neg = capture_sites(signal, neg_texts, (site,), dtype="float32")[site]
        sides[spec.name] = (
            pos.detach().to("cpu").numpy().astype(np.float32),
            neg.detach().to("cpu").numpy().astype(np.float32),
        )
    return sides


def default_feature_bank(
    signal: "RewardSignal",
    *,
    category: str = "quality",
    site: Site | None = None,
    store: "EvidenceStore | None" = None,
) -> ConceptBank:
    """Build a default `ConceptBank` off a signal, the seam `measure.indices` looks for.

    `reward_lens.measure.indices._support.load_default_bank` imports ``reward_lens.concepts`` and calls
    ``default_feature_bank(signal)`` if present, so wiring this name at the package level upgrades the
    indices from the synthetic `LinearFeatureBank` to real extracted features. It captures the named
    battery's concept sides off ``signal`` (final ``resid_post`` by default) and returns the built
    bank; failures propagate to the caller, which degrades to None by contract. torch is used only via
    the capture path.
    """
    resolved_site = site or Site(int(signal.meta.n_layers) - 1, "resid_post")
    specs = bank(category)
    sides = capture_concept_sides(signal, specs, resolved_site)
    built = build_bank(
        specs,
        sides,
        resolved_site,
        bank_name=category,
        signals=(str(signal.meta.fingerprint),),
        store=store,
    )
    return built.bank


__all__ = [
    "ConceptSpec",
    "STYLE_BANK",
    "SAFETY_BANK",
    "QUALITY_BANK",
    "BELIEF_BANK",
    "STANDARD_BANKS",
    "bank",
    "all_specs",
    "ConceptBank",
    "BankManifest",
    "ConceptSides",
    "BuiltBank",
    "build_bank",
    "capture_concept_sides",
    "default_feature_bank",
]
