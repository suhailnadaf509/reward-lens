"""A published per-rollout table, read as ledger steps, so `Λ` runs on a labelled series.

The instruments read `StepSample`, which is deliberately not a `Run`: the arithmetic is the same
whether the rows came from a record this library wrote or from a table somebody else published, and
putting the record type in the middle of it would make the second case a rewrite rather than an
adapter. This module is that adapter, and it is what puts a labelled rollout series inside reach on
a laptop.

**What it needs, and the one thing it has to reconstruct.** A rollout table carries rewards and
labels and no advantages, because the advantage is an artefact of the trainer rather than of the
rollout. So the advantage is recomputed from the group's own rewards under a stated estimator
convention and every reading built this way carries ``advantage_source="reconstructed"``. That is a
real assumption and it is the one to check first if a number here looks wrong: the reconstruction is
correct only if the column named as the reward is the column the trainer actually normalised over,
and only if the grouping column is the unit it normalised within.

**Five traps this adapter is written against, each of which is a real property of the AISI series
and each of which costs a day when missed.**

1. The step index is **per eval file, not per row**. `rollout_index` in that series is the
   chronological index of the source `.eval` file, so it is the training-step axis and not a row
   counter. `check_step_axis` asserts that the step column has exactly as many distinct values as
   the file column, which is the check that catches the misreading.
2. The label column is `int64` with **1, 0 or null**. A null is unscored, not a negative, so a naive
   `.sum()` over the column understates nothing and a naive `.mean()` over a NaN-filled float cast
   silently drops rows from the denominator without saying so. `label_rate` returns the rate, the
   denominator and the null count together for that reason.
3. The two published series are **different lengths**: one has 401 eval logs and the other 404.
   Nothing here hard-codes a length.
4. `hack_config` is a **JSON string**, not a struct, and is left as a string unless asked for.
5. Groups must be formed **within a step**. A problem recurring at three steps is three groups, not
   one, because the trainer normalised each batch separately. `steps_from_table` keys the group on
   `(step, group_column)` and never on the group column alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.measure.ledger.features import SURFACE_NAMES, surface_features
from reward_lens.measure.ledger.price import StepSample, advantages_from_rewards

#: The AISI reward-hacking rollout series, as column names. Named as a preset rather than hard-coded
#: into the loader, because the adapter is about the shape and not about one publisher.
AISI_COLUMNS: Mapping[str, str] = {
    "step": "rollout_index",
    "file": "source_eval_file",
    "group": "problem_id",
    "reward": "training_passed",
    "label": "reward_hacked",
    "text": "response",
}


@dataclass(frozen=True)
class LabelRate:
    """One step's labelled rate, with the two numbers a rate is meaningless without."""

    step: int
    rate: float
    n_labelled: int
    n_null: int

    @property
    def n_total(self) -> int:
        return self.n_labelled + self.n_null


def _column(table: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in table:
        raise KeyError(
            f"the table has no column {name!r}; it carries {sorted(table)[:12]}"
            + ("..." if len(table) > 12 else "")
        )
    return np.asarray(table[name], dtype=object)


def _as_float(values: np.ndarray) -> np.ndarray:
    """An object column of ints, floats and Nones, as float64 with None becoming NaN.

    Explicit rather than `astype(float)`, because a column of Python `None` casts to the string
    ``'None'`` under some dtypes and to `nan` under others, and one of those two silently becomes a
    number.
    """
    out = np.empty(values.shape[0], dtype=np.float64)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = np.nan
        else:
            out[i] = float(v)
    return out


def check_step_axis(
    table: Mapping[str, Any], columns: Mapping[str, str] = AISI_COLUMNS
) -> tuple[int, int]:
    """``(distinct steps, distinct source files)``, raising when they disagree.

    The step column of a published rollout table indexes the eval file the batch came from, and
    reading it as a per-row counter is the single commonest way to get a step series wrong. When the
    two counts agree, the column is the step axis; when they do not, it is something else and the
    caller has to say what.
    """
    steps = np.unique(_column(table, columns["step"]))
    files = np.unique(_column(table, columns["file"]))
    if steps.size != files.size:
        raise ValueError(
            f"the step column {columns['step']!r} has {steps.size} distinct values and the file "
            f"column {columns['file']!r} has {files.size}. In a published rollout table the step "
            f"index is per eval file, so the two agree when the column is the training-step axis. "
            f"They do not here, so this column is a row counter or the files are not one per step, "
            f"and the series has to be built from {columns['file']!r} sorted by name instead."
        )
    return int(steps.size), int(files.size)


def label_rate(
    table: Mapping[str, Any],
    columns: Mapping[str, str] = AISI_COLUMNS,
    label: str | None = None,
) -> list[LabelRate]:
    """The labelled positive rate per step, with the denominator and the null count beside it.

    Nulls are excluded from the denominator and counted, never read as zeros. A step where every
    label is null has a rate of NaN and a denominator of zero, which is different from a step where
    every rollout was scored and none was positive, and the two must not render the same.
    """
    key = label or columns["label"]
    steps = _as_float(_column(table, columns["step"]))
    values = _as_float(_column(table, key))
    out: list[LabelRate] = []
    for step in np.unique(steps[np.isfinite(steps)]):
        mask = steps == step
        block = values[mask]
        scored = block[np.isfinite(block)]
        out.append(
            LabelRate(
                step=int(step),
                rate=float(scored.mean()) if scored.size else float("nan"),
                n_labelled=int(scored.size),
                n_null=int(block.size - scored.size),
            )
        )
    return out


def steps_from_table(
    table: Mapping[str, Any],
    columns: Mapping[str, str] = AISI_COLUMNS,
    *,
    std_epsilon: float = 1e-4,
    std_normalised: bool = True,
    min_group: int = 2,
) -> list[StepSample]:
    """Every step of a per-rollout table, featurised, with advantages reconstructed per group.

    Groups are keyed on ``(step, group_column)``: a problem that recurs across steps is a fresh group
    each time, because the trainer normalised each batch on its own. Rollouts whose text produces no
    features are dropped and counted; rollouts whose reward is null keep their row and carry a NaN
    advantage, because they were still sampled from the policy and still belong in `z`.

    ``min_group`` drops groups too small to have within-group spread. It is 2 rather than 1 because
    a single-rollout group has an advantage of exactly zero by construction and contributes nothing
    but a degree of freedom.
    """
    steps = _as_float(_column(table, columns["step"]))
    groups = _column(table, columns["group"]).astype(str)
    rewards = _as_float(_column(table, columns["reward"]))
    texts = _column(table, columns["text"])

    out: list[StepSample] = []
    for step in np.unique(steps[np.isfinite(steps)]):
        mask = steps == step
        idx = np.flatnonzero(mask)
        rows: list[list[float]] = []
        keep: list[int] = []
        for i in idx:
            text = texts[i]
            values = surface_features("" if text is None else str(text), 1)
            if values is None:
                continue
            rows.append([values[n] for n in SURFACE_NAMES])
            keep.append(int(i))
        dropped = int(idx.size - len(keep))
        if not keep:
            continue
        kept = np.asarray(keep, dtype=np.intp)
        labels = groups[kept]
        codes = {name: n for n, name in enumerate(sorted(set(labels.tolist())))}
        group_ids = np.asarray([codes[str(g)] for g in labels], dtype=np.int64)
        sizes = np.bincount(group_ids, minlength=len(codes))
        big = np.asarray([sizes[g] >= min_group for g in group_ids], dtype=bool)
        advantages = advantages_from_rewards(
            rewards[kept],
            group_ids,
            std_epsilon=std_epsilon,
            std_normalised=std_normalised,
        )
        advantages = np.where(big, advantages, np.nan)
        out.append(
            StepSample(
                index=int(step),
                names=SURFACE_NAMES,
                features=np.asarray(rows, dtype=np.float64),
                advantages=advantages,
                group_ids=group_ids,
                task_ids=tuple(str(g) for g in labels),
                advantage_source="reconstructed",
                n_dropped=dropped,
                detail=(
                    f"{len(keep)} rollouts over {len(codes)} groups, advantages reconstructed as "
                    f"(r - mean_g) / (std_g + {std_epsilon:g})"
                    + (f"; {dropped} carried no response text" if dropped else "")
                ),
            )
        )
    return out


def read_parquet(path: str) -> dict[str, np.ndarray]:
    """A parquet file as a column mapping, through pandas, which is already a core dependency.

    Kept to one function so that the acceptance test's skip condition is "this file is not here"
    rather than "this environment cannot read parquet". `pyarrow` ships in the ``[record]`` extra and
    pandas needs it for this call; nothing else in the ledger touches either.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    return {name: frame[name].to_numpy() for name in frame.columns}


def parse_hack_config(value: Any) -> dict[str, Any] | None:
    """`hack_config` is a JSON string, not a struct. Parsed only when asked for, never implicitly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def rate_series(rates: Sequence[LabelRate]) -> tuple[list[int], list[float]]:
    """``(steps, rates)`` for the steps that carried at least one scored label."""
    usable = [r for r in rates if r.n_labelled > 0]
    return [r.step for r in usable], [r.rate for r in usable]


__all__ = [
    "AISI_COLUMNS",
    "LabelRate",
    "check_step_axis",
    "label_rate",
    "parse_hack_config",
    "rate_series",
    "read_parquet",
    "steps_from_table",
]
