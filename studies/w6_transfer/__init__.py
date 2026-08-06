"""W6.6, W6.7 and W6.8: three compute-gated re-measurements, written but not run.

Each of the three is a published number re-measured under a controlled configuration, and each ships
as the same six things: an instrument that composes what the library already has, a frozen study
spec with a preregistered prediction and a kill criterion, an acceptance test on a planted subject
that proves the arithmetic without touching a GPU, a statement of which real subject the claim
needs, a runbook, and a price with its assumptions written out.

    K2  `k2_standard_addition`  the transfer coefficient, re-measured by dosing the target rather
                                than a clean organism. Removes the organism-design dependence, or
                                shows the matrix-effect diagnosis was wrong.
    K3  `k3_shelf_life`         when a readout stops working, in steps. Needs a checkpoint series
                                and at rung 0 not even that.
    K4  `k4_sparsity`           update sparsity re-measured from FP32 master weights and across
                                controlled staleness. The published figure may be a property of
                                BF16 storage.

`ranked()` prints the three in buying order. The order is not the order they appear in the
catalogue: K3 rung 0 costs nothing and can be run today, K2 is a few hundred dollars, and K4 is the
only one that needs a training run.

The three submodules are imported inside the functions rather than at the top of this file. That is
not style: each of them is runnable as `python -m studies.w6_transfer.<row>`, and a package
`__init__` that has already imported the module `runpy` is about to execute makes `runpy` warn and
run it twice under two names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only, never at runtime
    from studies.w6_transfer.pricing import Quote


def all_quotes() -> list["Quote"]:
    """Every row's headline quote, each carrying the count of rows its design resolves.

    K3's headline is rung 0, not rung 1, because rung 0 is the version that needs no purchase and a
    ranking by cost per decision that quoted the expensive rung would bury the one recommendation
    this package actually makes.
    """
    from studies.w6_transfer import k2_standard_addition, k3_shelf_life, k4_sparsity

    return [
        k2_standard_addition.quote(resolvable=k2_standard_addition.resolvable_rows()),
        k3_shelf_life.quote_rung0(resolvable=k3_shelf_life.resolvable_rows()),
        k4_sparsity.quote_rung0(resolvable=k4_sparsity.resolvable_rows()),
    ]


def ranked() -> str:
    """The three rows in buying order, by preregistered rows resolved per thousand dollars."""
    from studies.w6_transfer.pricing import render_ranking

    return render_ranking(all_quotes())


def __getattr__(name: str) -> Any:
    """Expose the four submodules by attribute without importing them at package import."""
    if name in ("k2_standard_addition", "k3_shelf_life", "k4_sparsity", "pricing"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "all_quotes",
    "k2_standard_addition",
    "k3_shelf_life",
    "k4_sparsity",
    "pricing",
    "ranked",
]
