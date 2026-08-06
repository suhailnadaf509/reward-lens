"""Shared helpers for the measurement battery.

The battery Observables read a reward signal's internals on preference pairs, so they all need the
same few operations: split a view of pairs into a chosen side and a rejected side, capture activations
at a set of sites in fp32 (fp16 would blur the faithful-port parity below the tolerance we assert),
and read the reward direction off the signal's readout. Those live here so each Observable is just its
own science, and so every Observable captures activations the same way.

Everything captures in fp32 by default. The activation store's default is fp16, which is right for a
population sweep, but the E-parity proof compares a v3 Observable to a v1 primitive to 1e-6, and fp16
rounding alone is larger than that. Capturing fp32 here is the deliberate choice that makes the
port-faithfulness checkable.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.types import Phase, Site, Substrate
from reward_lens.runtime.backend import CaptureSpec
from reward_lens.signals.base import PositionSpec

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.classifier import ClassifierRM


# ---------------------------------------------------------------------------
# The declarations the battery shares
# ---------------------------------------------------------------------------

#: How each regime condition this battery depends on is measured, named by quantity id. Every
#: condition an instrument puts in ``requires`` has to appear here or `EnvelopeSpec` refuses to be
#: constructed, so this table is what decides whether a precondition is checkable at all rather
#: than decorative.
#:
#: ``STATIONARY_GRADER`` is measured by the check standard (catalogue J5): a fixed probe set
#: re-scored across the measurement window, whose drift is direct evidence that the grader moved
#: while it was being measured. ``GROUP_NONDEGENERATE`` is measured by the degenerate-group
#: fraction (E2), which counts the groups with no score spread to read a contrast from.
#: ``ABOVE_LOD`` is measured by the substrate limit of detection (M1), the grader's disagreement
#: with itself, which is the floor a patch effect or a reward delta has to clear before it is a
#: measurement rather than noise. ``LINEAR_RESPONSE`` is measured by the selection-explained
#: fraction Lambda (F2), which is the quantity named for that condition.
#:
#: The same table appears in ``measure/indices/_support.py``. It is duplicated rather than shared
#: because ``measure.battery`` requires the white-box extra at import and the index library is
#: torch-free by contract, so an import from battery into indices would install a dependency the
#: index library exists without.
MEASURED_BY: Mapping[RegimeCondition, str] = MappingProxyType(
    {
        RegimeCondition.STATIONARY_GRADER: "monitor.check_standard_drift",
        RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction",
        RegimeCondition.ABOVE_LOD: "substrate.lod",
        RegimeCondition.LINEAR_RESPONSE: "selection.explained_fraction",
    }
)

#: The substrates with weights to read. An instrument that captures a residual stream or projects
#: onto a readout vector applies to these two and to nothing else: a PROGRAM grader has source code
#: where these have activations, and asking it for a layer is a category error, not a hard case.
NEURAL_SUBSTRATES = frozenset({Substrate.NEURAL_SCALAR, Substrate.NEURAL_GEN})

#: Every substrate. What an instrument that reads only scores applies to.
ANY_SUBSTRATE = frozenset(Substrate)

#: When a grader study can be read. These instruments need full forward passes over a diagnostic
#: view, so they run before optimisation starts or after it finishes. They are not on the hot path,
#: and against a DEPLOYED artifact the grader is no longer reachable at all.
GRADER_STUDY_PHASES = frozenset({Phase.PRE_RUN, Phase.POST_RUN})


def resid_sites(n_layers: int) -> tuple[Site, ...]:
    """The reward-lens sites: the embedding (layer -1) then each layer's ``resid_post``.

    These are the residual-stream reading points v1's ``RewardLens`` projected onto ``w_r``, keyed
    exactly as v1 keyed them (``-1`` for the post-embedding residual, ``0..n_layers-1`` for the
    post-block residuals), so a v3 lens curve lines up index-for-index with a v1 lens curve.
    """
    return (Site(-1, "embed"),) + tuple(Site(layer, "resid_post") for layer in range(n_layers))


def component_sites(n_layers: int) -> tuple[Site, ...]:
    """The DLA sites: the embedding plus each layer's attention and MLP outputs.

    These are the additive residual-stream contributions the reward decomposes into, keyed to match
    v1's ``ComponentAttribution`` component names (``embed``, ``attn_L{l}``, ``mlp_L{l}``).
    """
    sites: list[Site] = [Site(-1, "embed")]
    for layer in range(n_layers):
        sites.append(Site(layer, "attn_out"))
        sites.append(Site(layer, "mlp_out"))
    return tuple(sites)


def head_sites(n_layers: int, n_heads: int) -> tuple[Site, ...]:
    """The per-head sites for head-level attribution and patching (``head_out`` at every head)."""
    return tuple(
        Site(layer, "head_out", head) for layer in range(n_layers) for head in range(n_heads)
    )


def pair_sides(view: Any) -> tuple[list[Any], list[Any]]:
    """Split a view of pairs into ``(chosen_items, rejected_items)`` as ``(prompt, response)`` lists.

    Accepts the data plane's ``Pair`` objects (``prompt_text`` / ``chosen.text`` / ``rejected.text``)
    and, for tests that pass them, bare ``(prompt, chosen, rejected)`` triples. The two returned lists
    are aligned by pair, which is what makes a per-pair differential meaningful.
    """
    chosen: list[Any] = []
    rejected: list[Any] = []
    for item in view:
        if hasattr(item, "chosen") and hasattr(item, "rejected"):
            prompt = item.prompt_text
            chosen.append((prompt, item.chosen.text))
            rejected.append((prompt, item.rejected.text))
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            prompt, ch, rj = item
            chosen.append((prompt, ch))
            rejected.append((prompt, rj))
        else:
            raise TypeError(
                f"pair_sides: expected a Pair or a (prompt, chosen, rejected) triple, got {item!r}"
            )
    return chosen, rejected


def capture_sites(
    signal: "ClassifierRM",
    items: list[Any],
    sites: tuple[Site, ...],
    *,
    full_sequence: bool = False,
    dtype: str = "float32",
) -> dict[Site, "torch.Tensor"]:
    """Capture ``sites`` for ``items`` in one forward, returning ``dict[Site, tensor]``.

    With ``full_sequence=False`` (the default) each tensor is the final-token activation ``(B, d)``
    (or ``(B, d_head)`` for a head site); with ``full_sequence=True`` it is the whole sequence
    ``(B, T, d)``, which the patching mechanics need for a source activation. fp32 by default.
    """
    spec = CaptureSpec(
        sites=sites,
        position=PositionSpec("final"),
        full_sequence=full_sequence,
        dtype=dtype,
    )
    capture = next(iter(signal.capture(items, spec)))
    return capture.tensors


def reward_direction(signal: "ClassifierRM", readout: str) -> "torch.Tensor":
    """The fp32 reward direction ``w_r`` for a readout (the signal's linear head weight)."""
    return signal.readout(readout).vector


__all__ = [
    "ANY_SUBSTRATE",
    "GRADER_STUDY_PHASES",
    "MEASURED_BY",
    "NEURAL_SUBSTRATES",
    "resid_sites",
    "component_sites",
    "head_sites",
    "pair_sides",
    "capture_sites",
    "reward_direction",
]
