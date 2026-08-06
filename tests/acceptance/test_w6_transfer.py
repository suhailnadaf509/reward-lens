"""K1, K2 and K3: the arithmetic of three compute-gated rows, proven without a GPU.

Nothing in this package is ever run against a real subject by the person who wrote it. That makes
this file the only thing standing between "a study spec that says what someone should do one day"
and a package. So it does five things and each of them is a clause of the acceptance:

1.  **Lint every instrument this package ships.** That is E56, where four instruments shipped
    failing lint rule 1 while their package read `done`, because the acceptance test rendered
    readings and never linted.
2.  **Prove the arithmetic on planted subjects**, where the answer is known by construction. The
    standard-addition extrapolation recovers a planted native level and a planted matrix factor;
    the BF16 sparsity arithmetic is checked bit for bit against torch's own cast and then against a
    planted run in which nothing at all is sparse; the shelf-life fit recovers a planted crossing.
3.  **Freeze every study spec**, so the preregistration is a content hash rather than a paragraph.
4.  **Run the generated invariance tests** from the instruments' own declarations, standing rule 4.
5.  **Reproduce the price arithmetic**, including the one place the specification's own scale
    sentence does not reconcile.

Every refusal path is exercised too, because a refusal that has never been produced is a docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.invariance import (
    InvariancePayload,
    check_invariance,
    parse_group_field,
    resolve_relation,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import MatrixDescription, ReferenceMaterial
from reward_lens.measure.base import lint_instrument
from reward_lens.organisms.chain import LadderRung, build_ladder, compose
from reward_lens.organisms.standard_addition import (
    Addition,
    linearity_check,
    matrix_factor,
    spike_recovery,
    standard_addition,
    standard_addition_uncertainty,
)
from reward_lens.organisms.transport import (
    SelectionDiagram,
    d_separated,
    planted_to_real_diagram,
    transportable,
)
from studies.w6_transfer import all_quotes, ranked
from studies.w6_transfer.k2_standard_addition import STUDY as K2_STUDY
from studies.w6_transfer.k2_standard_addition import (
    T32_DESIGN_SPREAD,
    T32_EXTERNAL_MAX,
    StandardAdditionTransfer,
)
from studies.w6_transfer.k2_standard_addition import freeze_study as k2_freeze
from studies.w6_transfer.k2_standard_addition import power_plan as k2_power
from studies.w6_transfer.k2_standard_addition import (
    quote as k2_quote,
)
from studies.w6_transfer.k2_standard_addition import runbook as k2_runbook
from studies.w6_transfer.k3_shelf_life import (
    DEFAULT_THRESHOLD,
    CheckpointAUROC,
    ReadoutShelfLife,
    decay_is_real,
    fit_shelf_life,
)
from studies.w6_transfer.k3_shelf_life import STUDY as K3_STUDY
from studies.w6_transfer.k3_shelf_life import freeze_study as k3_freeze
from studies.w6_transfer.k3_shelf_life import quote_rung0 as k3_quote0
from studies.w6_transfer.k3_shelf_life import runbook as k3_runbook
from studies.w6_transfer.k4_sparsity import (
    PUBLISHED_SPARSITY_RANGE,
    SparsityReading,
    UpdateSparsityUnderStaleness,
    bf16_round,
    fit_staleness_curve,
    format_floor,
    representable_step,
    update_sparsity,
)
from studies.w6_transfer.k4_sparsity import STUDY as K4_STUDY
from studies.w6_transfer.k4_sparsity import freeze_study as k4_freeze
from studies.w6_transfer.k4_sparsity import quote_rung0 as k4_quote0
from studies.w6_transfer.k4_sparsity import runbook as k4_runbook
from studies.w6_transfer.pricing import RATES, check_dossier_arithmetic

#: Everything this package declares as an instrument. Rule 3 of the standing rules is that lint is
#: the gate, and the list is written out here so adding an instrument without linting it is a
#: visible omission rather than an invisible one.
INSTRUMENTS = (
    StandardAdditionTransfer(),
    ReadoutShelfLife(),
    UpdateSparsityUnderStaleness(),
)

STUDIES = (K2_STUDY, K3_STUDY, K4_STUDY)


# ---------------------------------------------------------------------------
# 1. Lint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrument", INSTRUMENTS, ids=lambda i: i.name)
def test_every_instrument_passes_lint(instrument) -> None:
    """E56's clause: lint everything the package ships, not the readings it renders."""
    findings = lint_instrument(instrument)
    assert findings == [], "\n".join(f.render() for f in findings)


@pytest.mark.parametrize("instrument", INSTRUMENTS, ids=lambda i: i.name)
def test_every_instrument_declares_a_real_baseline_and_a_resolvable_group(instrument) -> None:
    """Beyond lint: the declarations have to name things that exist, not merely be non-empty."""
    assert instrument.baselines, f"{instrument.name} declares no baseline"
    assert all(len(b) > 20 for b in instrument.baselines), (
        f"{instrument.name} declares a baseline too short to name anything"
    )
    groups = parse_group_field(instrument.invariance)
    assert groups, f"{instrument.name}'s invariance field resolves to no group"
    for gid in groups:
        relation = resolve_relation(instrument, gid)
        assert relation.status in ("invariant", "covariant", "raw_only")


def test_the_two_group_declaration_resolves_per_group_not_globally() -> None:
    """E55: K2 declares two groups through the mapping form, and both resolve separately."""
    k2 = StandardAdditionTransfer()
    assert parse_group_field(k2.invariance) == ["repr.basis", "reward.affine"]
    assert resolve_relation(k2, "repr.basis").status == "invariant"
    assert resolve_relation(k2, "reward.affine").status == "invariant"
    # And a group the instrument says nothing about falls back to invariant rather than raising,
    # which is the documented default and the safe direction: a covariant instrument that forgot
    # fails its generated test loudly.
    assert resolve_relation(k2, "group.permutation").status == "invariant"


def test_k4_declares_raw_only_because_a_coordinate_count_is_not_reparam_invariant() -> None:
    """E13's split inside `policy.reparam`, and the reason it is not the easy answer.

    A count of unchanged coordinates is not preserved by a smooth reparameterisation: the map mixes
    coordinates, so an entry that was exactly unchanged in one basis is generically changed in
    another. Declaring `invariant` here would pass a generated test that had been weakened to let it
    through, which is exactly what E13 says a status on the group forces.
    """
    k4 = UpdateSparsityUnderStaleness()
    assert resolve_relation(k4, "policy.reparam").status == "raw_only"

    rng = np.random.default_rng(0)
    d = 64
    before = rng.standard_normal(d)
    after = before.copy()
    after[: d // 2] += 1e-3  # exactly half the coordinates move
    assert update_sparsity(before, after) == pytest.approx(0.5)

    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    rotated_sparsity = update_sparsity(before @ q.T, after @ q.T)
    assert rotated_sparsity == pytest.approx(0.0), (
        "under an orthogonal reparameterisation every coordinate moves, so a sparsity of 0.5 "
        "becomes 0.0. That is why the relation is raw_only and not invariant."
    )


# ---------------------------------------------------------------------------
# 2a. K2's arithmetic on a planted target
# ---------------------------------------------------------------------------


def _planted_addition_arms(
    *,
    native: float = 0.30,
    slope_target: float = 0.50,
    slope_clean: float = 1.50,
    sigma: float = 0.006,
    n_levels: int = 6,
    seed: int = 17,
):
    """A target carrying a known native level in a matrix that suppresses the response threefold."""
    rng = np.random.default_rng(seed)
    adds = np.linspace(0.0, 0.75, n_levels)
    target = [
        Addition(added=float(a), response=float(slope_target * (native + a) + rng.normal(0, sigma)))
        for a in adds
    ]
    clean = [
        Addition(added=float(a), response=float(slope_clean * a + rng.normal(0, sigma)))
        for a in adds
    ]
    return target, clean, native, slope_target / slope_clean


def test_standard_addition_recovers_a_planted_native_level_inside_its_expanded_interval() -> None:
    """Against `expanded`, not against `2 * u_native`, and the difference is the finding.

    An extrapolated intercept from a four-to-six point sweep has two to four residual degrees of
    freedom, which is exactly where the conventional coverage factor of 2 is wrong. The interval to
    check the planted value against is the one the kernel's budget produces from the fit's own dof.
    """
    target, _, native, _ = _planted_addition_arms()
    fit = standard_addition(target, dose_unit="rho")
    assert not isinstance(fit, Refusal), getattr(fit, "render", lambda: fit)()
    assert abs(fit.native_level - native) < fit.expanded, fit.render()
    assert fit.budget().coverage_factor > 2.0, "at four residual dof, k is not 2"
    assert fit.has_zero_addition
    assert fit.extrapolation_span > 0, "the intercept sits below the smallest addition"


@pytest.mark.parametrize("n_levels,k2_coverage", [(3, 0.700), (4, 0.823), (6, 0.883)])
def test_the_extrapolation_interval_covers_at_t_and_undercovers_badly_at_k_equals_two(
    n_levels, k2_coverage
) -> None:
    """The property that says the uncertainty formula is right, measured rather than asserted.

    2,000 draws per sweep size on a planted target. With the Student-t factor at the fit's own
    degrees of freedom the interval covers the planted native level about 95% of the time at every
    sweep size from three levels up. With `k = 2` it covers 70.0% at three levels, 82.3% at four
    and 88.3% at six, and those are the numbers quoted in `StandardAdditionFit.expanded`.
    """
    native, slope, sigma, trials = 0.30, 0.50, 0.006, 2000
    adds = np.linspace(0.0, 0.75, n_levels)
    rng = np.random.default_rng(0)
    hits_t = hits_k2 = 0
    for _ in range(trials):
        responses = slope * (native + adds) + rng.normal(0, sigma, n_levels)
        fit = standard_addition(
            [Addition(added=float(a), response=float(r)) for a, r in zip(adds, responses)]
        )
        error = abs(fit.native_level - native)
        hits_t += error < fit.expanded
        hits_k2 += error < 2.0 * fit.u_native
    assert 0.93 < hits_t / trials < 0.97, f"t coverage {hits_t / trials:.3f} at {n_levels} levels"
    assert hits_k2 / trials == pytest.approx(k2_coverage, abs=0.02), (
        f"k=2 coverage {hits_k2 / trials:.3f} at {n_levels} levels"
    )
    assert hits_k2 < hits_t, "k = 2 undercovers an extrapolated intercept at every sweep size here"


def test_the_shipped_inverse_prediction_equals_the_textbook_extrapolation_variance() -> None:
    """The compose that makes this module thin, asserted rather than claimed.

    `DoseResponseFit.u_char_at(x0, individual=False)` is `(s/|b|)·sqrt(1/n + (x0-xbar)^2/Sxx)` and
    the standard-addition form is `(s/|b|)·sqrt(1/n + ybar^2/(b^2·Sxx))`. They are the same
    expression because `ybar/b = xbar - x0`, so the module calls the shipped one and this test is
    what keeps that true.
    """
    target, _, _, _ = _planted_addition_arms()
    fit = standard_addition(target)
    textbook = standard_addition_uncertainty(fit.slope, fit.intercept, fit.s_resid, fit.added)
    assert fit.u_native == pytest.approx(textbook, rel=1e-12), (
        f"shipped {fit.u_native!r} against textbook {textbook!r}"
    )


def test_the_matrix_factor_recovers_the_planted_suppression_and_the_bias_it_implies() -> None:
    """The ratio of two slopes, checked across seeds because one draw is not a coverage claim.

    `u_factor` combines two slope standard errors, each estimated on four residual degrees of
    freedom, so a single draw can sit several nominal sigma out for the same small-sample reason the
    extrapolation interval needs a `t` factor. The point estimate is what is asserted per draw; the
    interval is asserted as a rate over twenty seeds.
    """
    inside = 0
    for seed in range(20):
        target, clean, _, planted_factor = _planted_addition_arms(seed=seed)
        got = matrix_factor(standard_addition(target), standard_addition(clean))
        assert not isinstance(got, Refusal), got.render()
        assert abs(got.factor - planted_factor) < 0.05, got.render()
        assert not got.is_consistent_with_no_effect, "a threefold suppression is not no effect"
        assert got.bias_of_external_calibration == pytest.approx(1.0 / got.factor)
        inside += abs(got.factor - planted_factor) < 3.0 * got.u_factor
    assert inside >= 15, f"{inside}/20 draws within three nominal sigma of the planted factor"


def test_spike_recovery_is_near_one_when_the_calibration_is_in_the_right_matrix() -> None:
    target, _, _, _ = _planted_addition_arms()
    fit = standard_addition(target)
    recovery = spike_recovery(
        unspiked=fit.responses[0], spiked=fit.responses[-1], added=fit.added[-1], slope=fit.slope
    )
    assert abs(recovery - 1.0) < 0.1, recovery


@pytest.mark.parametrize(
    "additions,reason",
    [
        (
            [Addition(added=0.0, response=0.1), Addition(added=1.0, response=0.6)],
            RefusalReason.RECORD_INCOMPLETE,
        ),
        (
            [Addition(added=a, response=0.42) for a in (0.0, 0.5, 1.0)],
            RefusalReason.BELOW_LOD,
        ),
        (
            [Addition(added=a, response=1.0 - a) for a in (0.0, 0.5, 1.0)],
            RefusalReason.RECORD_INCOMPLETE,
        ),
    ],
    ids=["two-levels", "flat-response", "inverted-response"],
)
def test_standard_addition_refuses_rather_than_extrapolating_nonsense(additions, reason) -> None:
    got = standard_addition(additions)
    assert isinstance(got, Refusal), got
    assert got.reason is reason, got.render()
    assert got.remedy and len(got.remedy) > 40, "a remedy is a user interface, not a label"


def test_linearity_check_finds_a_curved_addition_line_and_admits_when_it_cannot() -> None:
    """The runs-test version of this passed a plainly curved sweep, which is why it is a lack-of-fit
    test now: an arch across five points makes three sign runs, not one."""
    levels = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    curved = [Addition(added=a, response=0.2 + 0.5 * a - 0.35 * a * a) for a in levels]
    ok, note = linearity_check(standard_addition(curved))
    assert not ok, note
    assert "quadratic term" in note

    straight = [Addition(added=a, response=0.2 + 0.5 * a) for a in levels]
    ok_straight, note_straight = linearity_check(standard_addition(straight))
    assert ok_straight, note_straight
    assert "little power" in note_straight, "a non-detection has to say how weak it is"

    short, note_short = linearity_check(standard_addition(_planted_addition_arms()[0][:3]))
    assert short and "too few" in note_short, note_short


# ---------------------------------------------------------------------------
# 2b. K2's transport gate and its ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edges,given,expected",
    [
        ((("X", "Z"), ("Z", "Y")), (), False),
        ((("X", "Z"), ("Z", "Y")), ("Z",), True),
        ((("X", "Z"), ("Y", "Z")), (), True),
        ((("X", "Z"), ("Y", "Z")), ("Z",), False),
    ],
    ids=["chain-open", "chain-blocked", "collider-blocked", "collider-opened"],
)
def test_d_separation_gets_the_textbook_cases_right(edges, given, expected) -> None:
    g = SelectionDiagram(nodes=frozenset({"X", "Y", "Z"}), edges=edges)
    assert d_separated(g, ["X"], ["Y"], given) is expected


def test_conditioning_on_a_colliders_descendant_opens_the_path() -> None:
    g = SelectionDiagram(
        nodes=frozenset({"X", "Y", "Z", "W"}), edges=(("X", "Z"), ("Y", "Z"), ("Z", "W"))
    )
    assert d_separated(g, ["X"], ["Y"], []) is True
    assert d_separated(g, ["X"], ["Y"], ["W"]) is False


def test_the_k2_diagram_licenses_the_score_only_by_reweighting_on_length() -> None:
    """X3's own finding, recovered as a graphical statement rather than as a number.

    Under `append` the hack lengthens a working solution and under `substitute` it replaces one, so
    `length` is generated differently in the two domains and the instrument's score depends on it.
    The diagram says the score transports only after reweighting on length, which is exactly the
    0.4528 design spread X3 measured, and it says `hack` itself transports directly.
    """
    diagram = planted_to_real_diagram()
    verdict = transportable(diagram, outcome="score", treatment="hack")
    assert verdict.verdict == "reweighted", verdict.render()
    assert verdict.licence == ("length",), verdict.render()

    blind = planted_to_real_diagram(measurable=())
    refused = transportable(blind, outcome="score", treatment="hack")
    assert refused.verdict == "not_transportable"
    assert not refused.may_cross
    assert "S:length" in refused.blocking


def test_a_degenerate_transport_query_refuses_instead_of_answering_direct() -> None:
    got = transportable(planted_to_real_diagram(), outcome="score", treatment="score")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.QUANTITY_UNDEFINED


def test_the_ladder_refuses_to_subtract_two_different_quantities() -> None:
    """E15 at the ladder: an undecided or mismatched unit makes two rungs incomparable."""
    good = build_ladder(
        [
            LadderRung(
                rung=0, value=0.47, n=1096, quantity="calibration.transfer_t32", estimator="a"
            ),
            LadderRung(
                rung=1, value=0.05, n=1096, quantity="calibration.transfer_t32", estimator="b"
            ),
        ]
    )
    assert not isinstance(good, Refusal), good
    assert good.improvement == pytest.approx(-0.42)
    assert good.transfers[0].name == "t32"

    mixed = build_ladder(
        [
            LadderRung(
                rung=0, value=0.47, n=1096, quantity="calibration.transfer_t32", estimator="a"
            ),
            LadderRung(
                rung=1, value=140.0, n=1096, quantity="instrument.shelf_life", estimator="b"
            ),
        ]
    )
    assert isinstance(mixed, Refusal)
    assert mixed.reason is RefusalReason.UNIT_MISMATCH
    assert "auc_difference" in mixed.detail and "count/step" in mixed.detail

    across_corpora = build_ladder(
        [
            LadderRung(
                rung=0, value=0.47, n=1096, quantity="calibration.transfer_t32", estimator="a"
            ),
            LadderRung(
                rung=1, value=0.05, n=4000, quantity="calibration.transfer_t32", estimator="b"
            ),
        ]
    )
    assert isinstance(across_corpora, Refusal)
    assert across_corpora.reason is RefusalReason.RECORD_INCOMPLETE


def test_the_chain_refuses_to_publish_a_total_against_an_uncertified_reference() -> None:
    """The kernel's rule, reached through this package's own composition path.

    L1's own finding is that homogeneity is the largest of the three terms at 2.4x
    characterisation, so an uncertified reference is not a rounding error and the chain must not
    quietly drop it.
    """
    ladder = build_ladder(
        [
            LadderRung(
                rung=0,
                value=T32_EXTERNAL_MAX,
                n=1096,
                quantity="calibration.transfer_t32",
                estimator="external",
            ),
            LadderRung(
                rung=1,
                value=0.05,
                n=1096,
                quantity="calibration.transfer_t32",
                estimator="standard addition",
            ),
        ]
    )
    uncertified = ReferenceMaterial(
        id="k2.target-dosed",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.04266,
        matrix=MatrixDescription(system="clean organism"),
    )
    chain = compose(ladder, uncertified, working_matrix=MatrixDescription(system="the target"))
    assert chain.u_total is None
    assert chain.u_total_lower_bound > 0
    assert "not characterised" in chain.render()
    assert chain.matrix_mismatch()


def test_k2_computes_the_two_coefficients_and_carries_the_transport_verdict() -> None:
    target, clean, _, _ = _planted_addition_arms()
    names = ("baseline.string_match", "baseline.length", "baseline.tfidf_logreg")
    inst = StandardAdditionTransfer(
        target_additions=target,
        clean_additions=clean,
        arm_external={
            "baseline.string_match": 0.88,
            "baseline.length": 0.43,
            "baseline.tfidf_logreg": 0.85,
        },
        arm_standard_addition={n: 0.88 for n in names},
        arm_refit={n: 0.90 for n in names},
        design_spread_standard_addition=0.012,
    )
    got = inst.compute()
    assert not isinstance(got, Refusal), got
    assert got.t32_external == pytest.approx(0.47, abs=1e-9)
    assert got.t32_standard_addition == pytest.approx(0.02, abs=1e-9)
    assert got.improvement < 0
    assert got.spread_collapsed
    assert got.transport == "reweighted"
    assert got.n_instruments == 3
    ladder = inst.ladder()
    assert ladder.by_rung[0].value == pytest.approx(0.47, abs=1e-9)


def test_k2_refuses_when_the_arms_do_not_carry_the_same_instruments() -> None:
    target, clean, _, _ = _planted_addition_arms()
    inst = StandardAdditionTransfer(
        target_additions=target,
        clean_additions=clean,
        arm_external={"a": 0.8, "b": 0.7},
        arm_standard_addition={"a": 0.85},
        arm_refit={"a": 0.9, "b": 0.9},
    )
    got = inst.compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "some arms and not others" in got.detail


def test_k2_refuses_with_the_number_attached_when_the_diagram_licenses_nothing() -> None:
    """The gate, both ways round, and the refusal has to survive being printed.

    `Refusal.partial` is typed `Evidence | None` and `Refusal.render` reads `.value` off it, so
    handing it the bare payload makes the refusal raise the moment anybody prints it. That is what
    the first version of this instrument did and what this test now pins: `compute` has no
    `Context`, so its numbers travel in `statistics`, and `measure` attaches the recorded Evidence.
    """
    from reward_lens.measure.base import Context

    target, clean, _, _ = _planted_addition_arms()
    names = ("baseline.length",)
    inst = StandardAdditionTransfer(
        target_additions=target,
        clean_additions=clean,
        arm_external={n: 0.43 for n in names},
        arm_standard_addition={n: 0.88 for n in names},
        arm_refit={n: 0.90 for n in names},
        diagram=planted_to_real_diagram(measurable=()),
    )
    got = inst.compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.QUANTITY_UNDEFINED
    assert got.partial is None, "no Context, so there is no Evidence to attach"
    assert got.statistics["t32_standard_addition"] == pytest.approx(0.02, abs=1e-9)
    assert got.render(), "a refusal that cannot be printed is not a reading"

    measured = inst.measure(Context(readout="score"))
    assert isinstance(measured, Refusal)
    assert measured.partial is not None
    assert measured.partial.value.t32_standard_addition == pytest.approx(0.02, abs=1e-9)
    assert measured.render()

    # And the ladder does not route around the gate.
    assert isinstance(inst.ladder(), Refusal)


# ---------------------------------------------------------------------------
# 2c. K4's arithmetic: the BF16 cast, checked against torch
# ---------------------------------------------------------------------------


def test_bf16_rounding_matches_torchs_own_cast_bit_for_bit() -> None:
    """The whole format argument rests on this cast, so it is checked against the reference one."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 200_000
    magnitudes = np.exp(rng.uniform(-6.0, 2.0, n)).astype(np.float32)
    x = (rng.standard_normal(n).astype(np.float32) * magnitudes).astype(np.float32)
    mine = bf16_round(x)
    theirs = torch.from_numpy(x).to(torch.bfloat16).float().numpy().astype(np.float64)
    assert np.array_equal(mine, theirs), (
        f"{int((mine != theirs).sum())} of {n} values disagree with torch's bfloat16 cast"
    )


def test_a_run_where_nothing_is_sparse_reads_as_96_percent_sparse_through_a_bf16_checkpoint() -> (
    None
):
    """The planted subject for K4, and the whole reason the row exists.

    Every parameter moves in FP32. A checkpoint-differencing study reading the stored BF16 copies
    sees almost none of it, and the format floor accounts for all of what it does see.
    """
    rng = np.random.default_rng(1)
    n = 100_000
    master_before = rng.standard_normal(n) * 0.02
    master_after = master_before + rng.standard_normal(n) * 1e-6

    assert update_sparsity(master_before, master_after) == 0.0, "planted: everything moves in FP32"

    stored = update_sparsity(bf16_round(master_before), bf16_round(master_after))
    floor = format_floor(master_before, master_after)
    assert stored == pytest.approx(0.96635, abs=1e-4), stored
    assert floor == pytest.approx(stored, abs=1e-12), (
        "with nothing sparse in FP32, the format floor is the whole of the stored figure"
    )
    assert stored > PUBLISHED_SPARSITY_RANGE[1], (
        "the format alone clears the top of the published cross-algorithm range, which is the "
        "claim this row is registered to test on a real run"
    )


def test_the_representable_step_is_the_one_number_the_format_argument_turns_on() -> None:
    assert representable_step(0.01) == pytest.approx(3.0517578125e-05)
    assert representable_step(0.02) == pytest.approx(6.103515625e-05)
    assert representable_step(0.0) == 0.0
    # Thirty times a per-step update at a learning rate of 1e-6, which is the whole argument.
    assert representable_step(0.01) / 1e-6 > 30.0


def test_update_sparsity_raises_on_a_mispaired_site_rather_than_returning_a_fraction() -> None:
    with pytest.raises(ValueError, match="parameters at this site"):
        update_sparsity(np.zeros(10), np.zeros(11))


def test_the_staleness_curve_fits_a_planted_slope_and_refuses_below_three_levels() -> None:
    readings = [
        SparsityReading(
            staleness=s,
            stored=0.90 - 0.001 * s,
            master=0.02 - 0.0005 * s,
            floor=0.88,
            n_parameters=8_000_000_000,
            seed=seed,
        )
        for s in (0, 1, 2, 4, 8)
        for seed in (0, 1, 2)
    ]
    curve = fit_staleness_curve(readings, which="master")
    assert not isinstance(curve, Refusal), curve
    assert curve.slope == pytest.approx(-0.0005, abs=1e-9)
    assert curve.at_zero == pytest.approx(0.02, abs=1e-9)
    assert curve.n_levels == 5 and curve.n_seeds == 3

    short = fit_staleness_curve(readings[:6], which="master")
    assert isinstance(short, Refusal)
    assert short.reason is RefusalReason.RECORD_INCOMPLETE


def test_k4_refuses_when_the_master_weight_column_is_missing() -> None:
    """The failure to catch on the first arm rather than the fifteenth."""
    readings = [
        SparsityReading(staleness=s, stored=0.9, master=float("nan"), floor=0.88, n_parameters=1000)
        for s in (0, 1, 2)
    ]
    got = UpdateSparsityUnderStaleness(readings).curve()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "master" in got.detail
    assert "optimiser" in got.remedy

    empty = UpdateSparsityUnderStaleness().curve()
    assert isinstance(empty, Refusal)
    assert empty.reason is RefusalReason.RECORD_INCOMPLETE


# ---------------------------------------------------------------------------
# 2d. K3's arithmetic on a planted decay
# ---------------------------------------------------------------------------


def _planted_decay(a0: float = 0.61, crossing: float = 140.0, theta: float = DEFAULT_THRESHOLD):
    """A readout whose shelf life at `theta` is `crossing` by construction."""
    tau = crossing / math.log((a0 - 0.5) / (theta - 0.5))
    steps = (0, 25, 50, 75, 100, 150, 200, 250)
    return tau, [
        CheckpointAUROC(step=s, auroc=0.5 + (a0 - 0.5) * math.exp(-s / tau), u=0.008, n=2000)
        for s in steps
    ]


def test_the_shelf_life_fit_recovers_a_planted_crossing_exactly() -> None:
    tau, series = _planted_decay()
    got = fit_shelf_life(series)
    assert not isinstance(got, Refusal), got
    assert got.steps == pytest.approx(140.0, abs=1e-6), got.render()
    assert got.tau == pytest.approx(tau, rel=1e-9)
    assert got.auroc_at_zero == pytest.approx(0.61, abs=1e-9)
    assert got.half_life == pytest.approx(tau * math.log(2.0), rel=1e-9)
    assert not got.is_bound
    assert got.r_squared == pytest.approx(1.0, abs=1e-9)


def test_the_shelf_life_fit_survives_noise_and_reports_a_standard_error() -> None:
    tau, _ = _planted_decay()
    rng = np.random.default_rng(11)
    steps = (0, 25, 50, 75, 100, 150, 200, 250)
    noisy = [
        CheckpointAUROC(
            step=s,
            auroc=float(0.5 + 0.11 * math.exp(-s / tau) + rng.normal(0, 0.008)),
            u=0.008,
            n=2000,
        )
        for s in steps
    ]
    got = fit_shelf_life(noisy)
    assert not isinstance(got, Refusal), got
    assert abs(got.steps - 140.0) < 20.0, got.render()
    assert got.tau_se > 0 and got.decay_is_resolved, got.render()


def test_a_crossing_beyond_the_last_checkpoint_is_reported_as_a_bound_and_not_a_number() -> None:
    steps = (0, 25, 50, 75, 100, 150, 200, 250)
    slow = [
        CheckpointAUROC(step=s, auroc=0.5 + 0.11 * math.exp(-s / 2000.0), u=0.008) for s in steps
    ]
    got = fit_shelf_life(slow)
    assert not isinstance(got, Refusal), got
    assert got.is_bound
    assert got.steps == 250
    assert "at least" in got.render()


@pytest.mark.parametrize(
    "series,reason,fragment",
    [
        (
            [CheckpointAUROC(step=s, auroc=0.61, u=0.01) for s in (0, 100)],
            RefusalReason.RECORD_INCOMPLETE,
            "checkpoint",
        ),
        (
            [CheckpointAUROC(step=s, auroc=0.61, u=0.01) for s in (0, 100, 200)],
            RefusalReason.BELOW_LOD,
            "does not decay",
        ),
        (
            [CheckpointAUROC(step=s, auroc=0.52, u=0.01) for s in (0, 100, 200)],
            RefusalReason.BELOW_LOD,
            "never above it",
        ),
    ],
    ids=["too-few", "no-decay", "below-threshold-at-the-start"],
)
def test_shelf_life_refuses_rather_than_inventing_an_expiry_date(series, reason, fragment) -> None:
    got = fit_shelf_life(series)
    assert isinstance(got, Refusal), got
    assert got.reason is reason, got.render()
    assert fragment in got.detail, got.render()


def test_a_threshold_at_or_below_chance_refuses_because_the_answer_is_infinite() -> None:
    _, series = _planted_decay()
    got = fit_shelf_life(series, threshold=0.5)
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.QUANTITY_UNDEFINED


def test_m8_says_whether_there_is_a_curve_before_anything_is_fitted() -> None:
    """The precondition, composed from the shipped interlaboratory comparison rather than rebuilt."""
    tau, _ = _planted_decay()
    rng = np.random.default_rng(11)
    steps = (0, 25, 50, 75, 100, 150, 200, 250)
    decaying = [
        CheckpointAUROC(
            step=s,
            auroc=float(0.5 + 0.11 * math.exp(-s / tau) + rng.normal(0, 0.008)),
            u=0.008,
            n=2000,
        )
        for s in steps
    ]
    outcomes = rng.binomial(1, 0.61, 2000).astype(float)
    got = decay_is_real(decaying, per_item_outcomes=outcomes, seed=1)
    panel = getattr(got, "value", got)
    assert panel.excess_dispersion > 2.0, panel.says()
    assert not panel.labs_understand_their_errors

    flat = [CheckpointAUROC(step=s, auroc=0.61, u=0.05, n=2000) for s in steps]
    still = decay_is_real(flat, per_item_outcomes=outcomes, seed=1)
    flat_panel = getattr(still, "value", still)
    assert flat_panel.excess_dispersion < 2.0, flat_panel.says()


def test_the_catalogue_illustration_does_not_sit_on_a_single_decay_law() -> None:
    """Reproduced before it was written down, which is what the module docstring claims.

    K3's `says` line prints 0.61 falling to 0.51 over 250 steps with a shelf life of 140 at 0.55.
    An exponential through the two endpoints crosses at 82.2 and a straight line crosses at 150.0.
    The sentence is an illustration rather than a measurement, so this is not a wrong result; it is
    a worked example that does not work, and the module says so where a reader will meet it.
    """
    a0, a250, theta, window = 0.61, 0.51, 0.55, 250
    tau = window / math.log((a0 - 0.5) / (a250 - 0.5))
    exponential = tau * math.log((a0 - 0.5) / (theta - 0.5))
    linear = (a0 - theta) / ((a0 - a250) / window)
    assert exponential == pytest.approx(82.2, abs=0.1)
    assert linear == pytest.approx(150.0, abs=0.1)
    assert not (81 < 140 < 84 or 149 < 140 < 151)


# ---------------------------------------------------------------------------
# 3. The studies freeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("study", STUDIES, ids=lambda s: s.id)
def test_every_study_carries_a_prediction_a_kill_and_a_disclosure(study) -> None:
    assert study.hypotheses, f"{study.id} registers no hypothesis"
    assert study.kill_criteria, f"{study.id} registers no kill criterion"
    assert len(study.notes) > 200, f"{study.id}'s disclosure is too short to disclose anything"
    for h in study.hypotheses:
        assert h.prediction.rationale, f"{study.id}/{h.id} predicts without saying why"
        assert h.scoreboard_row, f"{study.id}/{h.id} names no catalogue row"


@pytest.mark.parametrize("freezer", [k2_freeze, k3_freeze, k4_freeze], ids=["K2", "K3", "K4"])
def test_freezing_produces_a_content_addressed_study_id(freezer) -> None:
    frozen = freezer()
    assert frozen.study_id.startswith("study:")
    assert "#" in frozen.study_id
    assert frozen.spec_hash
    assert frozen.frozen_at
    # Freezing twice gives the same id: the hash is over the spec, not over the clock.
    again = freezer()
    assert again.study_id == frozen.study_id


def test_editing_a_prediction_after_the_fact_changes_the_study_id() -> None:
    """What the freeze is for: a rewritten threshold is a visibly different study."""
    from dataclasses import replace

    from reward_lens.studies.freeze import freeze

    original = freeze(K2_STUDY)
    edited_prediction = replace(K2_STUDY.hypotheses[0].prediction, threshold=0.5)
    edited_hypothesis = replace(K2_STUDY.hypotheses[0], prediction=edited_prediction)
    edited = freeze(replace(K2_STUDY, hypotheses=(edited_hypothesis,) + K2_STUDY.hypotheses[1:]))
    assert edited.study_id != original.study_id


def test_the_new_k2_row_names_its_designs_and_says_it_does_not_replace_p6() -> None:
    """The brief's constraint, checked in the artifact rather than left to the report.

    P6 is frozen and not well posed, and rewriting it after seeing X3's answer is the failure the
    freeze exists to prevent. This spec is registered beside it and its comparators name the design
    they were measured under.
    """
    assert "does not replace P6" in K2_STUDY.notes
    assert "substitute" in K2_STUDY.notes and "append" in K2_STUDY.notes
    spread_row = next(h for h in K2_STUDY.hypotheses if h.id == "H-spread-collapses")
    assert spread_row.prediction.effect == pytest.approx(T32_DESIGN_SPREAD)
    below_row = next(h for h in K2_STUDY.hypotheses if h.id == "H-below-external")
    assert below_row.prediction.threshold == pytest.approx(T32_EXTERNAL_MAX)


# ---------------------------------------------------------------------------
# 4. The generated invariance tests
# ---------------------------------------------------------------------------


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from reward_lens.stats.baselines.base import auroc

    return float(auroc(np.asarray(scores, dtype=np.float64), np.asarray(labels)))


def test_k2s_coefficient_is_invariant_under_both_groups_it_declares() -> None:
    """Standing rule 4, run from `resolve_relation` so the test cannot be handed the answer.

    The coefficient is a maximum of absolute AUROC differences, and an AUROC is a rank statistic. An
    affine rescaling of the reward and an orthogonal change of representation basis both leave every
    pairwise ordering alone, so both relations are invariant and both are checkable.
    """
    rng = np.random.default_rng(5)
    n, d = 400, 16
    labels = rng.integers(0, 2, n)
    activations = rng.standard_normal((n, d)) + 0.6 * labels[:, None]
    readouts = rng.standard_normal((2, d))
    payload = InvariancePayload(
        scores=activations @ readouts[0],
        activations=activations,
        readouts=readouts,
        group_ids=np.arange(n) // 8,
    )
    reference = {"a": _auc(payload.scores, labels), "b": _auc(activations @ readouts[1], labels)}

    def run_affine(_inst, p: InvariancePayload) -> float:
        from studies.w6_transfer.k2_standard_addition import transfer_coefficient

        got, _ = transfer_coefficient({"a": _auc(p.scores, labels)}, {"a": reference["a"]})
        return got

    def run_basis(_inst, p: InvariancePayload) -> float:
        from studies.w6_transfer.k2_standard_addition import transfer_coefficient

        scores = {
            "a": _auc(np.asarray(p.activations) @ np.asarray(p.readouts)[0], labels),
            "b": _auc(np.asarray(p.activations) @ np.asarray(p.readouts)[1], labels),
        }
        got, _ = transfer_coefficient(scores, reference)
        return got

    instrument = StandardAdditionTransfer()
    for gid, runner in (("reward.affine", run_affine), ("repr.basis", run_basis)):
        relation = resolve_relation(instrument, gid)
        report = check_invariance(instrument, gid, payload, n=32, relation=relation, run=runner)
        assert report.passed, report.render()
        assert report.relation == relation


def test_k3s_unit_group_is_a_refusal_check_and_the_refusal_fires() -> None:
    """`units` is the one refusal-only group: its assertion is that a comparison raises.

    A shelf life in steps compared against a transfer coefficient in ΔAUC has to refuse rather than
    convert, and the ladder is the comparison surface where that is enforced. E15's rule, exercised.
    """
    from reward_lens.core.invariance import get_group

    assert get_group("units").refusal_only
    assert resolve_relation(ReadoutShelfLife(), "units").status == "invariant"

    refused = build_ladder(
        [
            LadderRung(rung=0, value=140.0, n=10, quantity="instrument.shelf_life", estimator="a"),
            LadderRung(
                rung=1, value=0.47, n=10, quantity="calibration.transfer_t32", estimator="b"
            ),
        ]
    )
    assert isinstance(refused, Refusal)
    assert refused.reason is RefusalReason.UNIT_MISMATCH


# ---------------------------------------------------------------------------
# 5. The prices
# ---------------------------------------------------------------------------


def test_the_rate_table_reproduces_the_dossiers_own_published_workload_costs() -> None:
    """A rate table that does not reproduce the costs it was quoted beside has been edited."""
    got = check_dossier_arithmetic()
    assert got["one_arm_low"] == pytest.approx(576.0, abs=1.0)
    assert got["multi_seed_hours"] == pytest.approx(11_520.0)
    assert got["multi_seed_low"] == pytest.approx(17_280.0, abs=1.0)
    assert got["multi_seed_high"] == pytest.approx(23_155.2, abs=1.0)
    assert got["multi_seed_modal"] == pytest.approx(45_504.0, abs=1.0)
    # The campaign's own $17.73 over 4.465 GPU-hours implies $3.97, which is Modal's published
    # $3.95 to within half a percent. Two independent numbers describing one purchase.
    assert got["campaign_implied_rate"] == pytest.approx(RATES["H100-modal"][0], rel=0.006)


def test_the_build_specs_scale_sentence_does_not_reconcile_with_its_own_band() -> None:
    """Reproduced rather than repeated, and reported as a finding.

    The specification's scale sentence quotes 11,520 GPU-hours, $17,000 to $23,000, and a floor of about $2.15
    per GPU-hour preemptible. 11,520 x 2.15 = $24,768, above the top of the band in the same
    sentence. The dossier's H100 floor band, $1.50 to $2.01, reproduces $17,280 to $23,155, which
    is the band exactly. So the band is right and $2.15 is not the floor it was computed from, and
    every quote in this package is struck at the dossier rates.
    """
    got = check_dossier_arithmetic()
    assert got["spec_floor_on_multi_seed"] == pytest.approx(24_768.0, abs=1.0)
    assert got["spec_floor_on_multi_seed"] > got["multi_seed_high"]
    assert RATES["H100-80GB"][0] < 2.15


@pytest.mark.parametrize("quoter", [k2_quote, k3_quote0, k4_quote0], ids=["K2", "K3", "K4"])
def test_every_quote_carries_its_assumptions_its_subject_and_a_finite_price(quoter) -> None:
    q = quoter()
    lo, hi = q.dollars
    assert math.isfinite(lo) and math.isfinite(hi) and hi >= lo >= 0
    assert q.items, f"{q.row} has no line items"
    assert all(i.why for i in q.items), f"{q.row} has a line item nobody can argue with"
    assert len(q.assumptions) >= 3, f"{q.row} states too few assumptions"
    assert q.subject_needed, f"{q.row} does not name the real subject it needs"
    cost = q.cost_model()
    assert cost.dollars == pytest.approx(q.dollars_mid)


def test_the_three_rows_rank_cheapest_decisive_first_and_k3_needs_no_purchase() -> None:
    """The ranking the maintainer buys from, asserted as an ordering rather than described."""
    quotes = all_quotes()
    text = ranked()
    order = [q.row for q in sorted(quotes, key=lambda q: (-q.decisiveness_per_1k, q.dollars_mid))]
    assert order[0].startswith("W6.8"), order
    assert order[1].startswith("W6.6"), order
    assert order[2].startswith("W6.7"), order
    assert "nothing to buy" in text
    k3 = next(q for q in quotes if q.row.startswith("W6.8"))
    assert k3.dollars == (0.0, 0.0)
    assert math.isinf(k3.decisiveness_per_1k)
    total_high = sum(q.dollars[1] for q in quotes)
    assert total_high < 2000.0, (
        f"all three headline rungs together come to ${total_high:,.0f}, which is the number the "
        f"maintainer is deciding about"
    )


def test_the_power_plans_say_plainly_which_rows_they_cannot_settle() -> None:
    """An underpowered arm registered as adequate is the failure M10 exists to stop."""
    k2 = k2_power(replicates=400)
    assert k2.resolution.resolved, k2.render()
    assert k2.mde < 0.06, k2.render()

    from studies.w6_transfer.k3_shelf_life import power_plan as k3_power
    from studies.w6_transfer.k4_sparsity import power_plan as k4_power

    assert not k3_power(replicates=400).resolution.resolved
    assert not k4_power(replicates=400).resolution.resolved

    from studies.w6_transfer.k3_shelf_life import resolvable_rows as k3_rows
    from studies.w6_transfer.k4_sparsity import resolvable_rows as k4_rows

    assert k3_rows(replicates=400) < len(K3_STUDY.hypotheses) + len(K3_STUDY.kill_criteria)
    assert k4_rows(replicates=400) < len(K4_STUDY.hypotheses) + len(K4_STUDY.kill_criteria)


@pytest.mark.parametrize("book", [k2_runbook, k3_runbook, k4_runbook], ids=["K2", "K3", "K4"])
def test_every_runbook_names_what_to_fetch_and_what_a_failed_arm_looks_like(book) -> None:
    text = book()
    assert "What a failed arm looks like" in text
    assert "Freeze" in text or "freeze" in text
    assert "$" in text, "a runbook without a price is a wish list"
    assert text.count("\n") > 30, "a runbook shorter than thirty lines is a docstring"
