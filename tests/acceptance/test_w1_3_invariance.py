"""`core/invariance.py`: the generated property test and the lint rule it enforces.

What this file has to show: the reward-affine group's generated test passes on an invariant index
and fails on a deliberately non-invariant one.

Everything else here exists because the gate has to hold under the three ways it would otherwise
be quietly bypassed: a covariant instrument declaring itself invariant, an instrument declaring a
relation its group does not admit, and a catalogue record with no group at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reward_lens.core.invariance import (
    COVARIANT_LINEAR,
    GROUPS,
    INVARIANT,
    RAW_ONLY,
    TRIVIAL_GROUP,
    InvariancePayload,
    Relation,
    check_invariance,
    check_unit_refusal,
    get_group,
    lint_catalogue,
    parse_group_field,
)
from reward_lens.core.quantity import catalogue_path
from reward_lens.core.reading import Refusal, RefusalReason

# ---------------------------------------------------------------------------
# Fixtures: a payload, an invariant index, and a deliberately non-invariant one
# ---------------------------------------------------------------------------


@pytest.fixture
def payload() -> InvariancePayload:
    rng = np.random.default_rng(0)
    return InvariancePayload(
        scores=rng.normal(size=64),
        group_ids=np.repeat(np.arange(8), 8),
    )


def _advantages(p: InvariancePayload) -> np.ndarray:
    """Group-centred advantage, which is what every group-relative estimator actually consumes."""
    s = np.asarray(p.scores, dtype=np.float64)
    g = np.asarray(p.group_ids)
    out = np.empty_like(s)
    for k in np.unique(g):
        m = g == k
        out[m] = s[m] - s[m].mean()
    return out


def advantage_norm(p: InvariancePayload) -> float:
    """Invariant under `reward.null` (the shift cancels within a group), covariant under affine."""
    return float(np.linalg.norm(_advantages(p)))


advantage_norm.name = "advantage_norm"  # type: ignore[attr-defined]


def advantage_sign_balance(p: InvariancePayload) -> float:
    """A genuine affine invariant: the fraction of advantages above zero is scale- and shift-free."""
    return float((_advantages(p) > 0).mean())


advantage_sign_balance.name = "advantage_sign_balance"  # type: ignore[attr-defined]


def mean_reward(p: InvariancePayload) -> float:
    """Deliberately not invariant: it reads a level, not a contrast."""
    return float(np.asarray(p.scores, dtype=np.float64).mean())


mean_reward.name = "mean_reward"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The generated test, passing and failing
# ---------------------------------------------------------------------------


def test_affine_generated_test_passes_on_an_invariant_index(payload):
    report = check_invariance(advantage_sign_balance, "reward.affine", payload, n=64)
    assert report.passed
    assert report.n == 64
    assert report.max_deviation <= report.tol
    assert len(report.draws) == 64


def test_affine_generated_test_fails_on_a_non_invariant_index(payload):
    report = check_invariance(mean_reward, "reward.affine", payload, n=64)
    assert not report.passed
    assert report.max_deviation > report.tol
    # The failure names the draw that broke it, so it is reproducible rather than merely reported.
    worst = report.worst
    assert worst is not None
    assert "a" in worst.params and "b" in worst.params
    assert "covariant" in report.interpretation


# ---------------------------------------------------------------------------
# The covariant case, which equality-only checking cannot express
# ---------------------------------------------------------------------------


def test_covariant_instrument_scales_by_the_declared_power(payload):
    """The advantage norm scales as a**1 under `r -> a*r + b`, and declaring so passes."""
    assert check_invariance(
        advantage_norm, "reward.affine", payload, n=32, relation=COVARIANT_LINEAR
    ).passed


def test_covariant_instrument_declaring_itself_invariant_fails(payload):
    """The direction a mis-declaration must fail in: loudly, not silently."""
    assert not check_invariance(
        advantage_norm, "reward.affine", payload, n=32, relation=INVARIANT
    ).passed


def test_wrong_covariant_weight_fails(payload):
    """A weight of 2 on a quantity that scales linearly is a declaration error and is caught."""
    quadratic = Relation("covariant", weight=2.0)
    assert not check_invariance(
        advantage_norm, "reward.affine", payload, n=32, relation=quadratic
    ).passed


def test_invariant_relation_cannot_carry_a_weight():
    with pytest.raises(ValueError, match="does not scale"):
        Relation("invariant", weight=1.0)


# ---------------------------------------------------------------------------
# reward.null, and the group whose failure is a finding
# ---------------------------------------------------------------------------


def test_null_group_leaves_advantages_alone(payload):
    assert check_invariance(advantage_norm, "reward.null", payload, n=32).passed


def test_null_group_moves_a_level_reading(payload):
    assert not check_invariance(mean_reward, "reward.null", payload, n=32).passed


def test_permutation_failure_is_marked_informative(payload):
    """A position-biased judge fails this, and the report says that is the test working."""

    def position_weighted(p: InvariancePayload) -> float:
        s = np.asarray(p.scores, dtype=np.float64)
        g = np.asarray(p.group_ids)
        return float(
            sum(np.dot(s[np.where(g == k)[0]], np.arange((g == k).sum())) for k in np.unique(g))
        )

    position_weighted.name = "position_weighted_judge"  # type: ignore[attr-defined]

    report = check_invariance(position_weighted, "group.permutation", payload, n=16)
    assert not report.passed
    assert report.informative
    assert "position bias" in report.interpretation


def test_permutation_passes_on_an_order_free_group_statistic(payload):
    assert check_invariance(advantage_norm, "group.permutation", payload, n=16).passed


# ---------------------------------------------------------------------------
# repr.basis, and the inner product it must preserve
# ---------------------------------------------------------------------------


def test_basis_rotation_preserves_the_readout_projection():
    """`h -> hQ^T` with `w -> wQ^T` leaves every h.w untouched, which is the convention check."""
    rng = np.random.default_rng(1)
    p = InvariancePayload(
        activations=rng.normal(size=(40, 16)),
        readouts=rng.normal(size=(2, 16)),
    )

    def projection_energy(pl: InvariancePayload) -> float:
        return float(np.linalg.norm(np.asarray(pl.activations) @ np.asarray(pl.readouts).T))

    projection_energy.name = "projection_energy"  # type: ignore[attr-defined]
    assert check_invariance(projection_energy, "repr.basis", p, n=16).passed


def test_basis_rotation_moves_a_raw_coordinate():
    """A single coordinate of the residual stream is a raw coordinate and must not claim invariance."""
    rng = np.random.default_rng(2)
    p = InvariancePayload(activations=rng.normal(size=(40, 16)))

    def first_coordinate(pl: InvariancePayload) -> float:
        return float(np.asarray(pl.activations)[:, 0].mean())

    first_coordinate.name = "first_coordinate"  # type: ignore[attr-defined]

    assert not check_invariance(first_coordinate, "repr.basis", p, n=16).passed
    # Declared raw_only, it is not asserted against, and the report records that it did move.
    raw = check_invariance(first_coordinate, "repr.basis", p, n=16, relation=RAW_ONLY)
    assert raw.passed
    assert raw.max_deviation > raw.tol
    assert "raw coordinates" in raw.interpretation


# ---------------------------------------------------------------------------
# A relation the group does not admit is a lint failure, not a pass
# ---------------------------------------------------------------------------


def test_declaring_an_unadmitted_relation_raises(payload):
    """`group.permutation` admits only `invariant`; declaring raw_only would opt out of the test."""
    with pytest.raises(ValueError, match="admits"):
        check_invariance(advantage_norm, "group.permutation", payload, n=4, relation=RAW_ONLY)


def test_every_group_declares_what_it_admits():
    """Every group offering a value relation names which ones, and `units` offers none.

    `units` is the exception and it is not an oversight: the assertion its group generates is a refusal
    (`UNIT_MISMATCH`) rather than a relation between two values, so there is no status an
    instrument could declare under it. It carries `refusal_only`, and `check_invariance` routes it
    to `check_unit_refusal` before it consults `admits` at all.
    """
    for gid, group in GROUPS.items():
        assert group.admits <= {"invariant", "covariant", "raw_only"}
        if group.refusal_only:
            assert not group.admits, f"{gid} asserts a refusal, so it admits no value relation"
        else:
            assert group.admits, f"{gid} admits no relation at all"


def test_tokenization_does_not_admit_raw_only():
    """The group offers "be invariant under it, or refuse", and a refusal is not `raw_only`.

    `raw_only` asserts nothing about the value and passes unconditionally, so reading "or refuse"
    as "or declare raw_only" would let a per-token instrument opt out of the one check that exists
    to make it declare a normalisation.
    """
    assert GROUPS["tokenization"].admits == {"invariant"}


def test_a_group_whose_generators_cannot_be_sampled_is_refused():
    """n draws from an unsamplable generator is one observation reported as n, and it always passes."""
    from reward_lens.core.invariance import GroupAction, InvarianceGroup

    fixed = InvarianceGroup(
        id="fixed",
        generators=(GroupAction(name="identity", apply=lambda p: p),),
        acts_on="scores",
    )
    with pytest.raises(ValueError, match="no sampler"):
        check_invariance(advantage_norm, fixed, InvariancePayload(scores=np.arange(4.0)), n=8)


def test_unknown_group_names_the_seven():
    with pytest.raises(KeyError, match="reward.affine"):
        get_group("no.such.group")


# ---------------------------------------------------------------------------
# The two groups whose assertion is not a numeric relation
# ---------------------------------------------------------------------------


def test_units_group_routes_to_the_refusal_check(payload):
    report = check_invariance(advantage_norm, "units", payload, n=8)
    assert report.passed
    assert "refusal" in report.skipped


def test_unit_refusal_accepts_a_refusal_and_rejects_a_number():
    def refuses(a, b):
        return Refusal(
            instrument="t",
            reason=RefusalReason.UNIT_MISMATCH,
            detail="per-token compared against per-sequence",
            remedy="express both readings in the same unit before comparing them.",
        )

    def converts(a, b):
        return 1.0

    def raises(a, b):
        raise TypeError("incompatible units")

    assert check_unit_refusal(refuses, 1.0, 2.0)
    assert check_unit_refusal(raises, 1.0, 2.0)
    assert not check_unit_refusal(converts, 1.0, 2.0)


def test_trivial_group_passes_vacuously(payload):
    """A deliberate `none` is an answer. It has no generators, so there is nothing to assert."""
    report = check_invariance(mean_reward, TRIVIAL_GROUP, payload, n=8)
    assert report.passed
    assert "no generators" in report.skipped


# ---------------------------------------------------------------------------
# The lint rule, run over the real catalogue
# ---------------------------------------------------------------------------


def _catalogue() -> tuple[list[dict], dict[str, str]]:
    ins = json.loads(catalogue_path("CATALOGUE.json").read_text(encoding="utf-8"))["instruments"]
    qs = json.loads(catalogue_path("QUANTITIES.json").read_text(encoding="utf-8"))["quantities"]
    return ins, {q["id"]: str(q.get("invariance_group")) for q in qs}


def test_parse_group_field_reads_the_catalogue_as_printed():
    assert parse_group_field("`reward.affine`") == ["reward.affine"]
    assert parse_group_field("`reward.affine`, `group.permutation`") == [
        "reward.affine",
        "group.permutation",
    ]
    # A trailing parenthetical explains the choice and is not part of it.
    assert parse_group_field("`policy.reparam` (Fisher-metric quantities are invariant)") == [
        "policy.reparam"
    ]
    assert parse_group_field("none") == [TRIVIAL_GROUP]
    assert parse_group_field("OPEN") == []
    assert parse_group_field(None) == []


def test_every_catalogue_instrument_resolves_an_invariance_group():
    """The lint rule that makes the fan-out safe: no instrument merges without a group.

    52 of the original 85 records print OPEN in their own column because the catalogue prints no
    group for them. Most resolve from the quantities they estimate, which the registry declares for
    all 125. The exception this test pins is the M-series controls, whose `quantities` field is
    itself OPEN: if that set moves in either direction, someone should know.

    It went from six to two. M3, M4, M5 and M8 have shipped instruments declaring
    registered quantities, and the catalogue records were filled from the installed source, which is
    rung 2 of the precedence ladder and beats a document. M5 is the interesting one: it declares
    `study.power` deliberately, because its reading is the same quantity M10 computes before the
    run, measured at a higher rung instead of calculated.

    The two that remain are the two that should. M6 has no instrument in the tree at all, so nothing
    has computed a stripped-text delta and nothing has had to choose the id. M7 declares no
    class-level quantity on purpose and says why: a combined standard uncertainty is in the units of
    its reading, so an uncertainty budget has no measurand separate from the one it is a budget for,
    and it takes the subject's quantity per instance.

    The four of series N (E23) declare `reward.affine` outright, because the tilt family is
    unchanged by an affine rescaling of the reward: `r -> a*r + b` reparametrises `lambda` to
    `lambda/a` and leaves both axes of the frontier where they were.
    """
    ins, qgroups = _catalogue()
    findings, resolved = lint_catalogue(ins, qgroups)
    unresolved = sorted(f.subject for f in findings)
    assert unresolved == ["M6", "M7"], (
        f"instruments with no resolvable invariance group changed: {unresolved}"
    )
    # 90 catalogue records since J6, the forecast calibration ledger, joined. It resolves
    # a group like everything else: `none`, which is the informative answer rather than the absent
    # one, because a Brier score is a function of probabilities and outcomes and no rescaling of a
    # reward acts on a number written down before the reward existed.
    # 95, from 91: N5 to N8, the contract layer, shipped and their catalogue rows were never
    # written, so four shipped instruments were invisible to every enumeration in the suite.
    assert len(resolved) == 95
    assert sum(1 for g in resolved.values() if g) == 93


def test_every_resolved_group_is_a_registered_group():
    ins, qgroups = _catalogue()
    _, resolved = lint_catalogue(ins, qgroups)
    for iid, groups in resolved.items():
        for gid in groups:
            assert gid in GROUPS, f"{iid} resolved to unregistered group {gid!r}"


def test_a_record_with_no_group_and_no_quantity_is_a_finding():
    findings, resolved = lint_catalogue(
        [{"id": "Z9", "quantities": [], "invariance_group": "OPEN"}], {}
    )
    assert [f.subject for f in findings] == ["Z9"]
    assert "declare `none`" in findings[0].remedy
    assert resolved["Z9"] == frozenset()


def test_a_string_quantities_field_is_not_iterated_as_characters():
    """The catalogue stores an absent quantity list as the bare string OPEN.

    Iterating it yields 'O', 'P', 'E', 'N', which is how a loader silently looks up four
    single-character quantity ids and reports nothing. Pinned because it is invisible when wrong.
    """
    findings, _ = lint_catalogue(
        [{"id": "M3", "quantities": "OPEN", "invariance_group": "OPEN"}], {}
    )
    assert [f.subject for f in findings] == ["M3"]
    assert "none of its quantities" in findings[0].problem
