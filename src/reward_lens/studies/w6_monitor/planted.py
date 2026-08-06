"""Subjects whose answer is known by construction, for both rows.

Neither `w6_4` nor `w6_5` is run. What can be asserted without spending anything is that the
arithmetic recovers a decomposition somebody planted, and that is what these generators exist for.
Each one takes the split it is meant to produce as an argument and builds a subject that has
exactly that split, so a test can assert the number rather than assert a range.

The generators are deliberately explicit about the physics they assume, because getting that wrong
is how a planted test passes while the instrument is broken, and it is how the first version of the
`w6_5` test lied. **Evasion is the policy moving under a fixed monitor; degradation is the monitor
moving under a fixed policy.** A generator that shrinks a monitor's scores multiplicatively changes
the spread as well as the separation and produces a mixture when it meant to produce one term, which
is what the first attempt here did: it reported 21% evasion on a subject built to have none, and the
instrument was correct both times.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from reward_lens.studies.w6_monitor.degradation import MonitorTrace

#: The published curve's endpoints, used as the default shape of a planted decay so the planted
#: subject sits in the same range as the thing being modelled. `d'` 3.75 falling to 1.25 and AUROC
#: 0.996 to 0.810 is what the 2026-07-29 curve reports. It is a **shape borrowed for a synthetic
#: subject** and nothing measured here reproduces or tests it.
PUBLISHED_D_PRIME_START: float = 3.75
PUBLISHED_D_PRIME_END: float = 1.25


def planted_monitor_trace(
    name: str,
    *,
    evasion_halvings: float = 0.0,
    degradation_halvings: float = 0.0,
    n_steps: int = 20,
    step_size: float = 10.0,
    n_rollouts: int = 400,
    d_prime_start: float = PUBLISHED_D_PRIME_START,
    in_loop_far: float = 0.05,
    seed: int = 0,
) -> MonitorTrace:
    """A trace whose evasion and degradation terms are exactly what you asked for.

    Both mechanisms are modelled as additive shifts of the hack class's monitor score with the spread
    held fixed, which is what makes the two terms add exactly rather than approximately. `d'` is then
    the shifted separation over a constant spread, so a shift that takes `d'` from `d0` to
    `d0 * 2**-k` is exactly `k` halvings whichever mechanism produced it.

    **Evasion** moves the live pool and leaves the frozen pool alone: the policy learned to write
    hacks that score lower, and rollouts written before that happened still score what they scored.

    **Degradation** moves both pools by the same amount: the monitor changed, so it now scores every
    hack lower, including the ones in a pool that has not changed since step zero.

    The total decay is ``evasion_halvings + degradation_halvings`` and the instrument should recover
    each term. Ask for one and zero to get a pure case; ask for two positive numbers to get a
    mixture whose share is ``evasion / (evasion + degradation)``.

    ``in_loop_far`` is carried through unmodified. Nothing here simulates the operating point; it is
    a label the ranking matches on and a planted subject supplies it so the matching can be tested.
    """
    rng = np.random.default_rng(seed)
    steps = tuple(float(i * step_size) for i in range(n_steps))
    horizon = steps[-1] if steps[-1] > 0 else 1.0

    # The frozen pool: one set of rollouts, drawn once, never redrawn. What changes across steps is
    # only what the monitor says about them.
    frozen_hack_noise = rng.normal(0.0, 1.0, n_rollouts)
    frozen_clean = tuple(float(v) for v in rng.normal(0.0, 1.0, n_rollouts))

    live_hack: list[tuple[float, ...]] = []
    live_clean: list[tuple[float, ...]] = []
    ref_hack: list[tuple[float, ...]] = []
    ref_clean: list[tuple[float, ...]] = []

    for step in steps:
        frac = step / horizon
        # Separation after `k` halvings is `d0 * 2**-k`; the shift is what is missing from `d0`.
        deg_shift = d_prime_start * (1.0 - 2.0 ** (-degradation_halvings * frac))
        eva_shift = d_prime_start * (1.0 - 2.0 ** (-evasion_halvings * frac))
        # The two shifts compose on the separation, not on the shifts, so the halvings add.
        live_sep = d_prime_start * 2.0 ** (-(evasion_halvings + degradation_halvings) * frac)
        ref_sep = d_prime_start - deg_shift

        live_hack.append(tuple(float(v) for v in rng.normal(live_sep, 1.0, n_rollouts)))
        live_clean.append(tuple(float(v) for v in rng.normal(0.0, 1.0, n_rollouts)))
        # The frozen pool is re-scored, not redrawn: same noise realisation, shifted mean.
        ref_hack.append(tuple(float(v) for v in (frozen_hack_noise + ref_sep)))
        ref_clean.append(frozen_clean)
        del eva_shift  # named for the reader; the separation above is what carries it

    return MonitorTrace(
        name=name,
        steps=steps,
        hack_scores=tuple(live_hack),
        clean_scores=tuple(live_clean),
        reference_hack=tuple(ref_hack),
        reference_clean=tuple(ref_clean),
        in_loop_far=in_loop_far,
        in_loop=True,
        provenance=(
            f"planted: {evasion_halvings:g} halvings of evasion and "
            f"{degradation_halvings:g} of degradation over {horizon:g} steps, seed {seed}"
        ),
    )


def planted_monitor_bank(
    n_monitors: int = 10,
    *,
    correlated: bool = False,
    seed: int = 0,
    n_steps: int = 20,
    n_rollouts: int = 400,
) -> tuple[MonitorTrace, ...]:
    """A bank of monitors for the ranking, in the two arrangements the kill condition cares about.

    ``correlated=True`` builds a bank where the monitor that starts best also decays slowest, so the
    two rankings agree and the kill condition should fire. ``correlated=False`` builds one where the
    starting `d'` and the decay rate are deliberately anti-aligned, so they disagree and it should
    not. Both are the same generator with the decay assignment reversed, which is the point: the
    kill condition is a statement about the joint distribution of two monitor properties in the wild,
    and neither arrangement is more natural than the other a priori. That is exactly why it is worth
    a run to find out.
    """
    rng = np.random.default_rng(seed)
    starts = np.linspace(2.0, 4.0, n_monitors)
    halvings = np.linspace(0.5, 3.0, n_monitors)
    if correlated:
        # Best starting d' decays least: rankings agree.
        halvings = halvings[::-1]
    out = []
    for i in range(n_monitors):
        out.append(
            planted_monitor_trace(
                f"monitor_{i:02d}",
                evasion_halvings=float(halvings[i]) * 0.7,
                degradation_halvings=float(halvings[i]) * 0.3,
                d_prime_start=float(starts[i]),
                n_steps=n_steps,
                n_rollouts=n_rollouts,
                in_loop_far=0.05,
                seed=int(rng.integers(0, 2**31)),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# `w6_4`
# ---------------------------------------------------------------------------


def zipf_base(n_head: int = 24, n_tail: int = 30, head_count: int = 120) -> dict[str, int]:
    """A base-policy family spectrum shaped like a real one: a Zipf head and a singleton tail.

    The tail is the part that matters. Good-Turing reads `f1`, the count of families seen exactly
    once, and a spectrum with no singletons gives an unseen-mass bound of exactly zero, which makes
    every novel family in the trained sample look like exploitation. A planted subject that cannot
    produce that failure on demand cannot test the instrument against it.
    """
    head = {f"family_{i:02d}": max(2, int(head_count / (i + 1))) for i in range(n_head)}
    tail = {f"family_{n_head + i:02d}": 1 for i in range(n_tail)}
    return {**head, **tail}


def planted_family_counts(
    *,
    base: Mapping[str, int] | None = None,
    selected: Sequence[str] = (),
    novel: Mapping[str, int] | None = None,
    selection_factor: float = 3.0,
    audit_noise: float = 0.0,
    audit_tracks: str = "realised",
    seed: int = 0,
) -> tuple[Counter, Counter, dict[str, float]]:
    """A base sample, a trained sample and a static audit, with a known decomposition.

    Returns ``(base_counts, trained_counts, audit)``. Families in ``selected`` have their counts
    multiplied by ``selection_factor``, which is pure selection: mass moved onto failure modes the
    base policy already produced. Families in ``novel`` are added to the trained sample only, which
    is pure exploitation, because the base policy never produced them at all.

    ``audit_tracks`` decides which side of the horse race the planted audit is meant to win.
    ``"realised"`` builds an audit that orders families by the mass they end up earning, so the audit
    should beat the base policy's own error ordering. ``"base"`` builds one that simply mirrors the
    base counts, so it should tie with them and the audit should be shown to add nothing. Those are
    the two outcomes the row is run to distinguish and a planted subject has to be able to produce
    either on demand.

    ``audit_noise`` is the standard deviation of multiplicative lognormal noise on the audit's
    counts, so the audit can be made an imperfect predictor rather than an oracle. At 0.0 it is
    exact, which makes an acceptance test assert a clean direction; at 0.5 the correlation degrades
    the way a real audit's would.
    """
    rng = np.random.default_rng(seed)
    if base is None:
        base = zipf_base()
    base_counts: Counter = Counter(dict(base))
    trained_counts: Counter = Counter(dict(base))
    for fam in selected:
        if fam in trained_counts:
            trained_counts[fam] = int(round(trained_counts[fam] * selection_factor))
    for fam, k in (novel or {}).items():
        trained_counts[fam] = trained_counts.get(fam, 0) + int(k)

    source = trained_counts if audit_tracks == "realised" else base_counts
    audit: dict[str, float] = {}
    for fam, k in source.items():
        noise = float(np.exp(rng.normal(0.0, audit_noise))) if audit_noise > 0 else 1.0
        audit[fam] = float(k) * noise
    return base_counts, trained_counts, audit


__all__ = [
    "PUBLISHED_D_PRIME_END",
    "PUBLISHED_D_PRIME_START",
    "planted_family_counts",
    "zipf_base",
    "planted_monitor_bank",
    "planted_monitor_trace",
]
