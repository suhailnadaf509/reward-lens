"""How a mid-stack residual is carried to the unembedding, for C8's two rungs.

The naive logit lens reads a layer-`l` residual straight through the unembedding: `W_U h_l`. That is
rung 0 and it is what the shipped library used. It assumes the residual at layer `l` is already in
the basis the unembedding expects, which is false whenever the layers above `l` do anything other
than add to it, and the correction is to carry the residual through the average input-output
Jacobian of the layers above first: `W_U J_l h_l`. That is rung 1.

**Three transports, one protocol, and the point of the protocol is that the comparator is free.**
`IdentityTransport` is the vanilla lens, so rung 0 costs nothing and is always available beside
rung 1 rather than being a separate code path somebody has to remember to run.
`AveragedJacobianTransport` fits `J_l` here, with the library's own runtime and autograd.
`jlens_transport` adapts an externally fitted lens, for a caller who has one.

**On vendoring `anthropics/jacobian-lens`, and why this does not.** Vendoring Apache-2.0 code
means shipping its licence, and the copy available is the *contents* of the package directory,
eight modules and a `pyproject.toml`, with **no LICENSE file and no README**. Beyond that, a hard
dependency on a package that is not importable everywhere is one nothing here could have been
tested against. So C8 is built against a protocol instead: a caller who installs `jlens` gets the vendor's
fitted lens through `jlens_transport` in one line, and a caller who does not gets a Jacobian fitted
here and the vanilla comparator regardless.

**Two facts about that package, verified in its source, that cost a day each if missed.**
The importable name is `jlens`, not `jacobian-lens`. And `SKIP_FIRST_N_POSITIONS = 16` builds the
mask `[skip_first : seq_len - 1]`, which drops the **final** position as well as the first sixteen.
The second is narrower than it sounds and the narrowing matters: that mask is used at **fit** time
only, in `fitting.py`. `apply` does not consult it, and its `positions` argument defaults to every
position. So the dropped final position biases which activations a lens is *fitted* on and does not
silently truncate a reading taken with one. `SKIP_FIRST_N_POSITIONS` below reproduces the convention
for the fit performed here, for the same reason: the first positions of a decoder stack are
dominated by the prompt template and the final position has no next token to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

#: The vendor's convention, reproduced for the fit performed here. The mask is
#: `[SKIP_FIRST_N_POSITIONS : seq_len - 1]`, so it drops the final position too, and that is
#: deliberate at fit time: the last position has no next token, and the first sixteen are dominated
#: by whatever chat template the judge is wrapped in.
SKIP_FIRST_N_POSITIONS = 16


@runtime_checkable
class VerdictTransport(Protocol):
    """Carry a layer-`l` residual into the unembedding's basis.

    ``transport`` takes `(n, d)` residuals at a layer and returns `(n, d)` residuals in the basis
    the unembedding reads. ``source_layers`` is the set of layers the transport was fitted for; a
    layer outside it is a refusal rather than an extrapolation.
    """

    name: str

    def transport(self, residual: np.ndarray, layer: int) -> np.ndarray: ...

    @property
    def source_layers(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class IdentityTransport:
    """The vanilla logit lens: read the residual straight through. Rung 0, and the comparator.

    Free by construction, which is the whole reason it is a transport rather than a special case. A
    comparator that costs nothing to run is a comparator that gets run, and C8's catalogue entry
    makes the naive form a mandatory baseline "always reported beside r1".

    It is not a null transport in the sense of being wrong on purpose. Reading a residual through
    the unembedding is exactly right at the final layer, and the question C8 asks is how far down
    the stack it stays approximately right.
    """

    n_layers: int = 0
    name: str = "logit_lens.identity"

    def transport(self, residual: np.ndarray, layer: int) -> np.ndarray:
        del layer
        return np.asarray(residual, dtype=np.float64)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(range(self.n_layers))


@dataclass
class AveragedJacobianTransport:
    """`J_l`, the average input-output Jacobian of the layers above `l`, fitted here.

    One `(d, d)` matrix per source layer, each the mean over sampled positions of
    `d(final residual) / d(residual at l)`. Averaged rather than per-token because the object C8
    needs is a property of the model at a layer, not of one token: a per-token Jacobian is a
    different matrix for every position and cannot be composed with a single unembedding row to give
    one direction.

    **What is approximated.** The map from a mid-stack residual to the final one is not linear, and
    this replaces it with its average local linearisation. The approximation is good where the
    layers above are close to linear in the residual and it degrades with depth below the top, which
    is the regime C8 is most interested in. `residual_fraction` records how much of the true final
    residual the linearisation reproduces on held-out positions, so a reading taken through a
    transport that explains 40% of the variance says so.

    Fitting costs one backward pass per output dimension per sampled position, so it is quadratic in
    `d` and is affordable on a small model and not on a large one. That is why the protocol exists:
    a large-model caller supplies a lens fitted elsewhere.
    """

    jacobians: dict[int, np.ndarray]
    residual_fraction: dict[int, float]
    n_positions: int = 0
    name: str = "jacobian_lens.averaged"

    def transport(self, residual: np.ndarray, layer: int) -> np.ndarray:
        h = np.asarray(residual, dtype=np.float64)
        j = self.jacobians.get(int(layer))
        if j is None:
            raise KeyError(
                f"no Jacobian was fitted for layer {layer}; this transport carries "
                f"{sorted(self.jacobians)}. Fit the layer or read it through the identity "
                f"transport and label the row as the naive lens."
            )
        return h @ j.T

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.jacobians))

    def render(self) -> str:
        rows = [
            f"  layer {layer:>3}  explains {self.residual_fraction.get(layer, float('nan')):.3f} "
            f"of the final residual"
            for layer in self.source_layers
        ]
        return "\n".join([f"{self.name}, {self.n_positions} positions"] + rows)


def fit_averaged_jacobian(
    subject: Any,
    items: Sequence[Any],
    layers: Sequence[int],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    max_positions: int = 32,
    seed: int = 0,
) -> AveragedJacobianTransport:
    """Fit one average Jacobian per layer by autograd through the subject's own runtime.

    The position mask is the vendor's, `[skip_first : seq_len - 1]`, and it drops the final position
    as well as the first `skip_first`. At fit time that is the right convention and not a bug: the
    final position has no next token to predict, and the opening positions are the chat template
    rather than the judgment.

    Positions are subsampled to `max_positions` because the cost is one backward pass per output
    dimension per position. `residual_fraction` is measured on the positions that were **not** used
    for the fit where any remain, so it is a held-out number rather than a training residual.
    """
    import torch

    from reward_lens.core.types import Site

    model = getattr(getattr(subject, "runtime", None), "model", None)
    if model is None:
        raise TypeError(
            f"{type(subject).__name__} exposes no `runtime.model`, so there is nothing to "
            f"differentiate through. Supply an externally fitted lens through `jlens_transport` "
            f"instead."
        )
    rng = np.random.default_rng(seed)
    jacobians: dict[int, np.ndarray] = {}
    fractions: dict[int, float] = {}
    n_used = 0

    for layer in layers:
        site = Site(int(layer), "resid_post")
        top = Site(int(getattr(subject.meta, "n_layers", 1)) - 1, "resid_post")
        rows: list[np.ndarray] = []
        held: list[tuple[np.ndarray, np.ndarray]] = []
        for item in items:
            tokenized = subject.tokenize(item)
            length = int(getattr(tokenized, "n_tokens", 0) or 0)
            lo, hi = skip_first, max(length - 1, skip_first)
            if hi <= lo:
                continue
            choices = np.arange(lo, hi)
            if choices.size > max_positions:
                choices = rng.choice(choices, max_positions, replace=False)
            for pos in choices:
                jac, source, target = _jacobian_at(subject, item, site, top, int(pos), torch)
                if jac is None:
                    continue
                rows.append(jac)
                held.append((source, target))
                n_used += 1
        if not rows:
            continue
        j = np.mean(np.stack(rows, axis=0), axis=0)
        jacobians[int(layer)] = j
        fractions[int(layer)] = _explained(j, held)
    return AveragedJacobianTransport(
        jacobians=jacobians, residual_fraction=fractions, n_positions=n_used
    )


def _jacobian_at(
    subject: Any, item: Any, site: Any, top: Any, position: int, torch: Any
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """One position's `d(top residual)/d(residual at site)`, or None where the graph is unavailable.

    Uses `grad_h`, which is the policy protocol's own differentiation entry point, one output
    dimension at a time. Returning None rather than raising on an unavailable graph keeps a single
    unhookable position from failing a whole fit, and the count of usable positions travels on the
    transport so a fit that lost most of them is visible.
    """
    try:
        jac = subject.jacobian_between(item, site, top, position=position)
    except AttributeError:
        return None, np.zeros(0), np.zeros(0)
    except Exception:
        return None, np.zeros(0), np.zeros(0)
    if jac is None:
        return None, np.zeros(0), np.zeros(0)
    matrix, source, target = jac
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )


def _explained(j: np.ndarray, held: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Fraction of the target residual's variance the linearisation reproduces."""
    if not held:
        return float("nan")
    pred = np.stack([h @ j.T for h, _ in held], axis=0)
    true = np.stack([t for _, t in held], axis=0)
    sse = float(np.sum((pred - true) ** 2))
    sst = float(np.sum((true - true.mean(axis=0)) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


@dataclass
class _WrappedLens:
    """An externally fitted lens behind this package's protocol."""

    lens: Any
    layers: tuple[int, ...]
    name: str = "jacobian_lens.vendored"

    def transport(self, residual: np.ndarray, layer: int) -> np.ndarray:
        out = self.lens.transport(np.asarray(residual, dtype=np.float64), int(layer))
        return np.asarray(out, dtype=np.float64)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return self.layers


def jlens_transport(lens: Any, *, layers: Sequence[int] | None = None) -> Any:
    """Adapt an externally fitted Jacobian lens to this package's protocol.

    Duck-typed on `transport` and `source_layers` rather than on an import, so it works with
    `jlens.JacobianLens`, with a lens fitted by something else, and with a test double, and so this
    module imports nothing that is not installed.

    **The package's importable name is `jlens`, not `jacobian-lens`,** it is not on PyPI, and its
    README says it is unmaintained, so a version pin will rot. It is not installed in this
    environment and nothing in this library depends on it.
    """
    if not (hasattr(lens, "transport") and hasattr(lens, "source_layers")):
        raise TypeError(
            f"{type(lens).__name__} does not satisfy the VerdictTransport protocol: it needs "
            f"`transport(residual, layer)` and a `source_layers` property. For "
            f"`jlens.JacobianLens`, wrap its per-layer matrices in a small adapter exposing those "
            f"two; the package is imported as `jlens` (not `jacobian-lens`), is not on PyPI, and "
            f"is not a dependency of this library."
        )
    resolved = tuple(layers) if layers is not None else tuple(lens.source_layers)
    return _WrappedLens(lens=lens, layers=resolved)


__all__ = [
    "SKIP_FIRST_N_POSITIONS",
    "AveragedJacobianTransport",
    "IdentityTransport",
    "VerdictTransport",
    "fit_averaged_jacobian",
    "jlens_transport",
]
