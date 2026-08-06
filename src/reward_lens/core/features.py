"""The feature-bank contract, in one place.

A behavioural feature is the object every Level 1 instrument is written in terms of: `Cov(A, f)`
is a selection differential on features, `G_ii/C_ii` is a feature's heritability, and the KL budget
decomposes into per-feature shares. So "what is a feature bank" is a kernel question, and until now
it had two answers.

`measure/indices/_support.py` defined a `FeatureBank` protocol with a `featurize` method and a
`directions()` accessor. `loops/recorder.py` defined an unrelated dataclass under the same name
holding named unit directions for dose tracking. Both were exported, nothing flagged it, and the
two are not interchangeable: one is a structural contract, the other is a concrete container whose
``directions`` is an attribute rather than a method. A caller working on the record and a caller
working on the indices would each have met a different object under one name.

The protocol is the contract and it lives here. The container is renamed `DirectionBank`, which is
what it is, and stays where it is used.

Torch-free by construction: a feature is an `(n, k)` array of numbers, and nothing about the
contract needs a compiled dependency.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureBank(Protocol):
    """The minimal contract a bank of behavioural features satisfies.

    ``names`` labels the ``k`` features. ``featurize`` turns an ``(n, d)`` activation matrix into
    an ``(n, k)`` matrix of feature values. ``directions()`` exposes the ``(k, d)`` decoder
    directions where the bank is linear, and returns None where it is not, which is the honest
    answer for a bank whose features are computed rather than projected.

    Note that `runtime_checkable` only checks that the attributes exist, not that they are
    callable. `isinstance(x, FeatureBank)` is therefore a weak check and a container that happens
    to carry a ``directions`` array passes it. That is exactly the trap `DirectionBank` used to
    sit in, and the reason the two objects now have two names.
    """

    names: tuple[str, ...]

    def featurize(self, activations: np.ndarray) -> np.ndarray: ...

    def directions(self) -> np.ndarray | None: ...


__all__ = ["FeatureBank"]
