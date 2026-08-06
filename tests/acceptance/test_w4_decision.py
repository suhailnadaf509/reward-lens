"""Acceptance: the contract layer, N5 to N8, on a real composite reward.

The clause: *on one real composite reward with at least three components and A2 variance components
available, N5 through N8 render a weight recommendation, an equal-compensation table naming the
starved component, a sorting cutoff, and a noise-and-angle pair per component, with the five
assumptions printed. A test asserts the matrix ordering* `C'' Sigma` *against a worked example
computed by hand.*

**What is real here and what is not, said plainly.** The component values are somebody else's:
1,327 RM-Bench prompts, each answered in three response variants, scored by ten open reward models,
recorded in the campaign store. **The composition is mine.** Which three of the ten models are the
components, what each is worth to the principal, how hard effort is and how risk averse the policy
is: none of that is in the store and all of it is stated here. What the store supplies is the noise,
which is the one parameter of the five that is measurable at all, and the sensitivity, which is
measurable once an identification is stated.

The three components are `armorm`, `tulu-rm` and `skywork-v2-qwen3-8b`. They are chosen to span the
panel's noise range rather than at random, and saying so is part of the reading: a composite of three
components with similar noise would produce a recommendation close to the value-weighted baseline and
would show nothing.

**Where Sigma comes from, and the boundary A2 draws that is worth recording.** For each component the
design is 1,327 prompts by 3 response variants with that one grader held fixed, and the noise is
everything in its score variance that is not the prompt: `sigma_GRR^2 = total - sigma2(p)`. The `A2`
*instrument* refuses on that design, correctly, because with one rater it cannot separate
repeatability from reproducibility and a `%GRR` reported there would be one wearing the other's name.
What is used is not `%GRR` but the gauge variance, which is well defined with one rater, so the
arithmetic is taken from A2's own `gauge_rr` on A2's own `GStudy`. The refusal is exercised below
rather than routed around, and the panel-level A2 reading that the instrument does accept is computed
beside it. The off-diagonal of Sigma comes from the correlation of the same residuals across
components, which is estimable because every model scored every cell.

**Where M comes from, and the identification that makes it a measurement.** Under the identification
that one unit of effort raises the latent quality component `i` measures by one standard deviation of
that component's prompt-level score, `mu'_i` is the standard deviation of that component's prompt
means in raw units. The identification is mine and the numbers under it are the store's, so the
parameter is recorded as supplied with the identification on the reading rather than as measured.

**The consequence is N8's kill condition, and it fires.** With the tasks identified one-to-one with
the components, the sensitivity matrix is diagonal by construction, so every component's distortion
is exactly zero and the angle half of the pair measures nothing on this subject. The pair is still
rendered per component, the noise half is a measurement, and the reading says which half is which.
The non-degenerate path is exercised on a constructed off-diagonal `M` at the end of this file and
labelled as constructed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeReading
from reward_lens.core.evidence import Evidence
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Phase
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.decision import (
    ASSUMPTION_KEYS,
    DECISION,
    ContractParameters,
    EqualCompensation,
    NoiseAndAngle,
    OptimalWeights,
    ParameterSource,
    SortingCutoff,
    Sweep,
    equal_compensation,
    noise_and_angle,
    noise_correlation_from_residuals,
    noise_from_gauge_studies,
    noise_to_signal,
    optimal_weights,
    recommend_weights,
    register_proposed,
    sorting_cutoff,
    two_task_surplus,
    unmeasurable_correction,
)
from reward_lens.measure.metrology.grr import VarianceComponents
from reward_lens.measure.metrology.gstudy import ReplicationDesign
from reward_lens.record.convert.store import CampaignStore
from reward_lens.stats.variance import gauge_rr

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

pytestmark = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. It holds the recorded scores of ten open reward models "
        "over a shared bank, which is what makes the noise in this composite a measurement "
        "rather than a number somebody picked. Set REWARD_LENS_CAMPAIGN_STORE."
    ),
)

#: `hackfore-flagged` is a derived marker rather than a reward model. Series A established this.
NOT_A_GRADER = {"hackfore-flagged"}

#: The three components of the composite. Chosen to span the panel's noise range, which is a choice
#: and is stated as one.
COMPONENTS = ("armorm", "tulu-rm", "skywork-v2-qwen3-8b")


# ---------------------------------------------------------------------------
# Loading the real component values
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _registry():
    """Register this package's six proposals, then take them back out.

    The quantity registry is process-global and `tests/acceptance/test_w1_kernel.py` asserts an
    exact count, so a module that registers and does not clean up breaks a test in a package it has
    nothing to do with. E40 settled the shape: snapshot what was there and remove
    whatever appeared, rather than listing the ids.
    """
    load_quantities()
    before = set(QUANTITIES)
    yield
    for added in set(QUANTITIES) - before:
        QUANTITIES._items.pop(added, None)


@pytest.fixture(scope="module")
def store() -> CampaignStore:
    return CampaignStore(CAMPAIGN_STORE)


@pytest.fixture(scope="module")
def panel(store: CampaignStore):
    """10 graders x 1,327 prompts x 3 response variants, in raw units. Returns (graders, cube).

    Raw and not gauge fixed, deliberately. The whole point of dividing a variance by a squared
    sensitivity is that it removes the scale, and standardising first would remove the scale before
    the instrument got to demonstrate that it can.
    """
    rows = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rmbench-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "pairs":
            continue
        rows[row.roster_key] = (
            list(value["item_ids"]),
            np.asarray(value["scores"], dtype=np.float64),
        )
    graders = sorted(rows)
    ids = rows[graders[0]][0]
    assert all(rows[g][0] == ids for g in graders), "the banks are not the same items"
    prompts = sorted({i.split("::")[0] for i in ids})
    index = {iid: j for j, iid in enumerate(ids)}
    cube = np.empty((len(prompts), len(graders), 3), dtype=np.float64)
    for pi, prompt in enumerate(prompts):
        for ri, grader in enumerate(graders):
            scores = rows[grader][1]
            for si in range(3):
                cube[pi, ri, si] = scores[index[f"{prompt}::c{si}r0"], 0]
    return graders, cube


@pytest.fixture(scope="module")
def component_cube(panel):
    """The three chosen components' raw scores: (1327 prompts, 3 components, 3 response variants)."""
    graders, cube = panel
    return cube[:, [graders.index(c) for c in COMPONENTS], :]


@pytest.fixture(scope="module")
def component_designs(component_cube):
    """One single-grader crossed design per component: prompts by response variants."""
    return [
        ReplicationDesign(
            scores=component_cube[:, k, :],
            single_rater=True,
            object_label="prompt",
            facet_labels=("response variant", "rubric"),
        )
        for k in range(len(COMPONENTS))
    ]


@pytest.fixture(scope="module")
def gauge_studies(component_designs):
    """A2's gauge reading per component: everything in the score variance that is not the prompt."""

    class PerComponent:
        """What `noise_from_gauge_studies` reads: a gauge and a rung, from A2's own arithmetic."""

        def __init__(self, design):
            self.gstudy = design.fit()
            self.gauge = gauge_rr(self.gstudy.components, part="p")
            self.rung = 1

    return [PerComponent(d) for d in component_designs]


@pytest.fixture(scope="module")
def real_parameters(component_cube, gauge_studies) -> ContractParameters:
    """The contract parameters: one measured, one identified, three stated."""
    residuals = np.stack(
        [
            (component_cube[:, k, :] - component_cube[:, k, :].mean(axis=1, keepdims=True)).ravel()
            for k in range(len(COMPONENTS))
        ],
        axis=1,
    )
    sigma, note = noise_from_gauge_studies(
        list(zip(COMPONENTS, gauge_studies)),
        noise_correlation_from_residuals(residuals),
    )
    sensitivity = np.diag(
        [float(np.std(component_cube[:, k, :].mean(axis=1), ddof=1)) for k in range(3)]
    )
    return ContractParameters.supplied(
        COMPONENTS,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=sigma,
        risk_aversion=1.0,
        sensitivity=sensitivity,
        source={"noise": ParameterSource.MEASURED},
        notes={
            "noise": note,
            "benefit": (
                "equal across components, stated. Nothing in the store says what any of the three "
                "is worth to a principal, and equal is the neutral statement rather than a "
                "measurement. The recommendation is homogeneous of degree one in B', so only the "
                "ratios matter."
            ),
            "sensitivity": (
                "the standard deviation of each component's prompt-level mean in raw units, under "
                "the identification that one unit of effort raises the latent quality that "
                "component measures by one standard deviation of its own prompt-level score. The "
                "numbers are the store's and the identification is not, so this is recorded as "
                "supplied. It makes M diagonal by construction, which is what fires N8's kill "
                "condition."
            ),
            "cost_curvature": (
                "the identity, stated. Nothing in a reward record identifies the curvature of a "
                "policy's effort cost, and the sweep over r is simultaneously a sweep over a "
                "common scaling of it."
            ),
            "risk_aversion": "r = 1, stated, and swept over six decades on every reading below.",
        },
    ).assume_equal_effort()


@pytest.fixture(scope="module")
def composite_weights() -> np.ndarray:
    """The composite as it stands: three components, equal weight. Mine, and the thing N6 argues with."""
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ---------------------------------------------------------------------------
# The subject
# ---------------------------------------------------------------------------


def test_the_subject_is_a_real_three_component_composite_with_a2_variance_components(
    panel, component_cube, gauge_studies, real_parameters
):
    graders, cube = panel
    assert len(graders) == 10
    assert cube.shape == (1327, 10, 3)
    assert all(c in graders for c in COMPONENTS)
    assert component_cube.shape == (1327, 3, 3)
    assert real_parameters.m == 3
    assert real_parameters.source["noise"] is ParameterSource.MEASURED
    assert "grader.variance_components" in real_parameters.note["noise"]
    # Every component's gauge variance is a real positive number from a real crossed design.
    for name, study in zip(COMPONENTS, gauge_studies):
        assert study.gauge.sigma_grr > 0.0, name
        assert 0.0 < study.gauge.part_share < 1.0, name
    sigma = real_parameters.noise
    assert np.all(np.linalg.eigvalsh(sigma) > 0.0), "Sigma is not positive definite"
    assert not np.allclose(sigma, np.diag(np.diag(sigma))), "the noise is correlated across models"


def test_a2_refuses_on_the_per_component_design_and_the_refusal_is_the_reason_for_the_route_taken(
    component_designs, panel
):
    """The boundary, exercised rather than routed around.

    A2 declines to report `%GRR` from a single-rater design, and it is right to: with one rater
    there is no reproducibility to report and the number would be repeatability wearing both names.
    The gauge variance it is built from is well defined there, which is what this package consumes.
    The panel design that A2 does accept is fitted beside it, at rung 2.
    """
    refusal = VarianceComponents(design=component_designs[0]).compute()
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "1 rater level" in refusal.detail
    assert "second grader draw" in refusal.remedy

    graders, cube = panel
    accepted = VarianceComponents(
        design=ReplicationDesign(
            scores=cube[:, [graders.index(c) for c in COMPONENTS], :],
            raters=COMPONENTS,
            object_label="prompt",
            facet_labels=("reward model", "response variant"),
        )
    ).compute()
    assert not isinstance(accepted, Refusal), accepted
    assert accepted.rung == 2
    assert set(accepted.components.names) == {"p", "r", "o", "pr", "po", "ro", "pro,e"}


def test_the_raw_variances_span_four_orders_of_magnitude_and_the_effort_equivalent_noise_does_not(
    real_parameters,
):
    """Why the sort is on `n` and not on `sigma^2`, measured on three real reward models.

    The three components' raw noise variances differ by a factor of about 75,000, which is entirely
    the scale each model happens to emit on. Dividing by the squared dose-response slope removes
    exactly that, and what is left is a factor of 80, which is a real difference in how noisy the
    graders are.
    """
    sigma2 = np.diag(real_parameters.noise)
    mu = np.diag(real_parameters.sensitivity)
    n = noise_to_signal(sigma2, mu)
    assert sigma2.max() / sigma2.min() > 1e4
    assert 10.0 < n.max() / n.min() < 1e3
    assert n.max() / n.min() < sigma2.max() / sigma2.min() / 100.0


# ---------------------------------------------------------------------------
# Clause: N5 renders a weight recommendation
# ---------------------------------------------------------------------------


def test_n5_renders_a_weight_recommendation_on_the_real_composite(real_parameters):
    reading = recommend_weights(real_parameters, sweep=Sweep.for_risk_aversion(n=41))
    assert reading.weights.shape == (3,)
    assert np.all(np.isfinite(reading.weights))
    # The noisiest component gets the lowest weight and the crispest the highest, from measured
    # noise alone: every component has the same stated value.
    order = [reading.components[i] for i in np.argsort(-reading.weights)]
    assert order == ["armorm", "tulu-rm", "skywork-v2-qwen3-8b"]
    assert np.all(np.diff(np.sort(reading.shrinkage)) >= 0)
    assert reading.most_shrunk[0] == "skywork-v2-qwen3-8b"
    assert reading.most_shrunk[1] < 0.2
    # And it is worth something: the recommendation beats both mandatory baselines on the
    # principal's own objective.
    assert reading.surplus > reading.surplus_value
    assert reading.surplus > reading.surplus_equal
    assert set(reading.baselines) == {"baseline.equal_weights", "baseline.value_weights"}
    assert "Weight armorm" in reading.says()


def test_n5_reports_the_recommendation_as_a_function_of_the_two_parameters_nobody_measured(
    real_parameters,
):
    """Sensitivity to `r` is part of the reading. Here the ordering is not stable and it says where."""
    curve = recommend_weights(real_parameters, sweep=Sweep.for_risk_aversion(n=41)).curve
    assert curve is not None
    assert curve.weights.shape == (41, 3)
    assert not curve.ordering_is_stable
    assert curve.named_ordering(curve.dominant_ordering) == (
        "armorm > tulu-rm > skywork-v2-qwen3-8b"
    )
    assert 0.5 < curve.dominant_span < 1.0
    assert len(curve.crossings) == 1
    crossing = curve.crossings[0]
    assert {crossing.first, crossing.second} == {"tulu-rm", "skywork-v2-qwen3-8b"}
    assert 1.0 < crossing.at < 100.0
    assert "not stable" in curve.says()


def test_n5_refuses_when_a_parameter_it_needs_is_neither_stated_nor_swept(real_parameters):
    """The rule, on the real subject: supplied and recorded, swept, or refused."""
    unstated = ContractParameters(
        components=real_parameters.components,
        benefit=real_parameters.benefit,
        cost_curvature=real_parameters.cost_curvature,
        noise=real_parameters.noise,
        risk_aversion=real_parameters.risk_aversion,
        sensitivity=real_parameters.sensitivity,
        effort=real_parameters.effort,
        source={**real_parameters.source, "risk_aversion": ParameterSource.UNKNOWN},
    )
    ctx = Context(phase=Phase.PRE_RUN)
    refusal = OptimalWeights(unstated).estimate(ctx)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "risk_aversion is unstated" in refusal.detail
    assert "Sweep.for_risk_aversion" in refusal.remedy
    swept = OptimalWeights(unstated, sweep=Sweep.for_risk_aversion(n=21)).estimate(ctx)
    assert isinstance(swept, Evidence)


# ---------------------------------------------------------------------------
# Clause: the matrix ordering, against a worked example computed by hand
# ---------------------------------------------------------------------------


def test_the_matrix_ordering_is_c_before_sigma_and_transposing_it_triples_one_weight():
    r"""The worked example, computed by hand. `C'' Sigma` is not `Sigma C''`.

    Take::

        C'' = [[2, 1],      Sigma = [[1, 0],       r = 0.5,   B' = (1, 1)
               [1, 4]]              [0, 3]]

    The two products, multiplied out entry by entry::

        C'' Sigma = [[2*1 + 1*0,  2*0 + 1*3]  = [[2,  3],
                     [1*1 + 4*0,  1*0 + 4*3]]    [1, 12]]

        Sigma C'' = [[1*2 + 0*1,  1*1 + 0*4]  = [[2,  1],
                     [0*2 + 3*1,  0*1 + 3*4]]    [3, 12]]

    They differ in the off-diagonal: 3 against 1 above the diagonal and 1 against 3 below it.

    **The correct ordering.**
    `I + r C'' Sigma = [[1 + 0.5*2, 0.5*3], [0.5*1, 1 + 0.5*12]] = [[2, 1.5], [0.5, 7]]`, whose
    determinant is `2*7 - 1.5*0.5 = 14 - 0.75 = 13.25`. Inverting a two by two by the adjugate rule
    gives `(1/13.25) [[7, -1.5], [-0.5, 2]]`, so::

        alpha* = (1/13.25) (7 - 1.5,  -0.5 + 2) = (1/13.25)(5.5, 1.5)

    and `13.25 = 53/4`, so `5.5/13.25 = (11/2)(4/53) = 22/53` and `1.5/13.25 = (3/2)(4/53) = 6/53`::

        alpha* = (22/53, 6/53) = (0.4150943..., 0.1132075...)

    Substituting back into the original equation, which is the arithmetic worth doing twice::

        row 1:  2*(22/53) + 1.5*(6/53) = 44/53 + 9/53 = 53/53 = 1  = B_1
        row 2:  0.5*(22/53) + 7*(6/53) = 11/53 + 42/53 = 53/53 = 1 = B_2

    **The transposed ordering.** `I + r Sigma C'' = [[2, 0.5], [1.5, 7]]`, determinant
    `14 - 0.75 = 13.25` again, because `AB` and `BA` have the same eigenvalues, so the whole
    difference lands in the direction rather than in the scale. Its inverse is
    `(1/13.25)[[7, -0.5], [-1.5, 2]]` and::

        alpha_wrong = (1/13.25)(7 - 0.5, -1.5 + 2) = (1/13.25)(6.5, 0.5) = (26/53, 2/53)

    So `6/53` against `2/53`: **transposing the product divides the recommended weight on the second
    component by exactly three.** It is not a rounding difference, and it is why the specification
    writes the ordering out and why this test exists.
    """
    c = np.array([[2.0, 1.0], [1.0, 4.0]])
    sigma = np.diag([1.0, 3.0])
    r, b = 0.5, np.array([1.0, 1.0])

    assert not np.allclose(c @ sigma, sigma @ c)
    assert np.array_equal(c @ sigma, np.array([[2.0, 3.0], [1.0, 12.0]]))
    assert np.array_equal(sigma @ c, np.array([[2.0, 1.0], [3.0, 12.0]]))
    assert np.linalg.det(np.eye(2) + r * c @ sigma) == pytest.approx(13.25)
    assert np.linalg.det(np.eye(2) + r * sigma @ c) == pytest.approx(13.25)

    p = ContractParameters.supplied(
        ("a", "b"), benefit=b, cost_curvature=c, noise=sigma, risk_aversion=r
    ).assume_unit_sensitivity()
    alpha = optimal_weights(p)
    assert alpha[0] == pytest.approx(22.0 / 53.0, rel=1e-12)
    assert alpha[1] == pytest.approx(6.0 / 53.0, rel=1e-12)
    assert np.allclose((np.eye(2) + r * c @ sigma) @ alpha, b, atol=1e-14)

    wrong = np.linalg.solve(np.eye(2) + r * sigma @ c, b)
    assert wrong[0] == pytest.approx(26.0 / 53.0, rel=1e-12)
    assert wrong[1] == pytest.approx(2.0 / 53.0, rel=1e-12)
    assert alpha[1] / wrong[1] == pytest.approx(3.0, rel=1e-12)


def test_the_ordering_matters_on_the_real_noise_matrix_too(real_parameters):
    """Not only on a constructed example: the real Sigma does not commute with a non-diagonal C''."""
    c = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, 0.3], [0.1, 0.3, 1.0]])
    sigma = real_parameters.noise
    assert not np.allclose(c @ sigma, sigma @ c)
    r = real_parameters.risk_aversion
    b = real_parameters.benefit
    right = np.linalg.solve(np.eye(3) + r * c @ sigma, b)
    wrong = np.linalg.solve(np.eye(3) + r * sigma @ c, b)
    assert not np.allclose(right, wrong)
    assert np.max(np.abs(right - wrong) / np.abs(right)) > 0.05


# ---------------------------------------------------------------------------
# Clause: N6 renders an equal-compensation table naming the starved component
# ---------------------------------------------------------------------------


def test_n6_names_the_starved_component_of_the_real_composite(real_parameters, composite_weights):
    """Equal weights across three real reward models does not mean equal incentives.

    `armorm`'s raw score barely moves across prompts, so a third of the weight on it delivers a
    small fraction of the marginal return the other two deliver, and the policy's capacity goes
    where the return is. That the starved component is the *least noisy* one is the reading being
    about incentives rather than about noise.
    """
    table = equal_compensation(composite_weights, real_parameters)
    assert len(table.rows) == 3
    assert table.starved.component == "armorm"
    assert not table.holds
    assert table.spread < 0.1
    assert table.rung == 0
    assert table.sensitivity_source is ParameterSource.SUPPLIED
    assert "identification" in table.sensitivity_note
    # The reweighting it recommends does equalise, checked by recomputing on it.
    equalised = equal_compensation(
        np.array([r.equalising_weight for r in table.rows]), real_parameters
    )
    assert equalised.holds
    assert equalised.spread == pytest.approx(1.0)
    assert np.sum([r.equalising_weight for r in table.rows]) == pytest.approx(1.0)
    assert "starved component" in table.says()


def test_the_specifications_printed_starvation_rule_names_the_opposite_component_here(
    real_parameters, composite_weights
):
    """The specification prints the reciprocal of what its own preceding clause implies.

    It says the commissions must be equal after dividing each signal by its own sensitivity, which
    is right and gives `kappa_i = alpha_i mu'_i`, and then concludes that the raw weights satisfy
    `alpha_i` proportional to `mu'_i` and that the starved component is the one with the lowest
    `alpha_i / mu'_i`. Both halves of that conclusion are inverted.

    On this composite the two rules do not merely disagree, they name opposite ends of the panel.
    The commissions are `(0.037, 0.421, 1.117)`, so the derivation starves `armorm`, the component
    whose score barely moves. The printed rule computes `alpha_i / mu'_i = (3.04, 0.264, 0.100)` and
    starves `skywork-v2-qwen3-8b`, the component with the largest response of the three. A table
    built on the printed rule would tell a reader to raise the weight on the component that is
    already paying the policy thirty times what the starved one pays.
    """
    table = equal_compensation(composite_weights, real_parameters)
    mu = np.diag(real_parameters.sensitivity)
    as_derived = np.array([r.commission for r in table.rows])
    as_printed = composite_weights / mu

    assert table.components[int(np.argmin(as_derived))] == "armorm"
    assert table.components[int(np.argmin(as_printed))] == "skywork-v2-qwen3-8b"
    assert table.starved.component == "armorm"
    assert as_derived[0] / as_derived[2] < 0.05
    assert any("reciprocal" in d for d in EqualCompensation.deviations)


def test_n6_refuses_with_record_incomplete_when_the_run_carries_no_weight_sweep(real_parameters):
    """E30's test, on the real subject: the remedy is upstream, not here.

    The campaign measured reward models on fixed banks and never optimised a policy against a
    reward, so no weight was ever perturbed and no dose-response slope was ever recorded. Nothing a
    reader does to this store recovers one.
    """
    without = ContractParameters(
        components=real_parameters.components,
        benefit=real_parameters.benefit,
        cost_curvature=real_parameters.cost_curvature,
        noise=real_parameters.noise,
        risk_aversion=real_parameters.risk_aversion,
        source={**real_parameters.source, "sensitivity": ParameterSource.UNKNOWN},
    )
    out = EqualCompensation(
        np.array([1 / 3, 1 / 3, 1 / 3]),
        without,
        record="the campaign store, 1,327 prompts scored on fixed banks",
    ).estimate(Context(phase=Phase.PRE_RUN))
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "the campaign store" in out.detail
    assert "re-run with a weight sweep" in out.remedy


# ---------------------------------------------------------------------------
# Clause: N7 renders a sorting cutoff
# ---------------------------------------------------------------------------


def test_n7_renders_a_sorting_cutoff_on_the_real_composite(real_parameters):
    """The crispest of three real reward models belongs in its own stage, and the split is priced."""
    reading = sorting_cutoff(real_parameters, n_contracts=2)
    groups = [set(c.members) for c in reading.contracts]
    assert {"armorm"} in groups
    assert {"tulu-rm", "skywork-v2-qwen3-8b"} in groups
    assert reading.cutoff is not None
    assert reading.cutoff_bracket is not None
    lo, hi = reading.cutoff_bracket
    assert lo < reading.cutoff < hi
    assert reading.split_gain > 0.5
    assert reading.best_contract_count == 2
    assert len(reading.value_by_contract_count) == 3
    assert reading.value_by_contract_count[1] == max(reading.value_by_contract_count)
    # The sorting theorem's own claim, checked on this subject rather than assumed.
    assert reading.exhaustive_ran
    assert reading.interval_optimal is True
    assert "Split at rho" in reading.says()


def test_n7_prices_what_keeping_the_noisy_component_in_the_sum_costs_the_others(real_parameters):
    """The dilution factor: one component's noise is a tax on every other component in the contract."""
    reading = sorting_cutoff(real_parameters, n_contracts=2)
    name, factor = reading.worst_dilution
    assert name == "skywork-v2-qwen3-8b"
    assert 0.0 < factor < 0.6
    n = reading.noise
    total = float(np.sum(n))
    r = real_parameters.risk_aversion
    cost = float(np.mean(np.diag(real_parameters.cost_curvature)))
    assert reading.dilution == pytest.approx(
        [(1 + r * cost * (total - x)) / (1 + r * cost * total) for x in n]
    )


# ---------------------------------------------------------------------------
# Clause: N8 renders a noise-and-angle pair per component
# ---------------------------------------------------------------------------


def test_n8_renders_a_noise_and_angle_pair_for_every_component(real_parameters):
    """Both halves per component, and the reading says which half is a measurement here.

    N8's kill condition fires on this subject. The tasks are identified one-to-one with the
    components, so the sensitivity matrix is diagonal, so every distortion is exactly zero by
    construction. The noise half is a real measurement and is what the pair is carrying.
    """
    reading = noise_and_angle(real_parameters)
    assert len(reading.rows) == 3
    for row in reading.rows:
        assert math.isfinite(row.noise) and row.noise > 0.0
        assert math.isfinite(row.congruity)
        assert math.isfinite(row.shrinkage)
        assert row.verdict
    assert reading.diagonal_sensitivity
    assert all(r.distortion == pytest.approx(0.0, abs=1e-12) for r in reading.rows)
    assert reading.wants_a_different_measure == ()
    assert set(reading.wants_lower_weight) == {"tulu-rm", "skywork-v2-qwen3-8b"}
    assert "exactly zero by construction" in reading.says()
    assert math.isfinite(reading.contract_congruity)


def test_n8_kill_condition_fires_on_this_subject_and_the_non_degenerate_path_still_works():
    """*If every real composite tested has a diagonal M, only the noise half of the pair ships.*

    It fires here, and the reason is structural rather than an accident of this store: a task space
    identified one-to-one with the components makes `M` diagonal by construction, and a task space
    that is not so identified is not recoverable from recorded scores. The instrument's other half is
    exercised below on a **constructed** off-diagonal `M`, which is synthetic and said so, to show
    that the reading is capable of the measurement when a subject supports it.
    """
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.5, 0.0, 1.0]])
    constructed = ContractParameters.supplied(
        ("unit_tests", "format_ok", "judge"),
        benefit=[0.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.05, 0.05]),
        risk_aversion=1.0,
        sensitivity=m,
    )
    reading = noise_and_angle(constructed)
    rows = {r.component: r for r in reading.rows}
    assert not reading.diagonal_sensitivity
    assert rows["judge"].distortion > 0.3
    assert reading.wants_a_different_measure == ("judge",)
    assert "replace the measure" in rows["judge"].verdict


# ---------------------------------------------------------------------------
# Clause: the five assumptions are printed
# ---------------------------------------------------------------------------


def test_all_four_readings_print_all_five_assumptions_on_the_real_subject(
    real_parameters, composite_weights
):
    renders = [
        recommend_weights(real_parameters).render(),
        equal_compensation(composite_weights, real_parameters).render(),
        sorting_cutoff(real_parameters, n_contracts=2).render(),
        noise_and_angle(real_parameters).render(),
    ]
    for text in renders:
        for key in ASSUMPTION_KEYS:
            assert key in text, key
        assert "Holmstrom and Milgrom 1991" in text


def test_the_evidence_the_four_instruments_emit_carries_the_assumptions_and_the_provenance(
    real_parameters, composite_weights
):
    load_quantities()
    register_proposed()
    ctx = Context(phase=Phase.PRE_RUN)
    readings = {
        "N5": OptimalWeights(real_parameters).estimate(ctx),
        "N6": EqualCompensation(composite_weights, real_parameters).estimate(ctx),
        "N7": SortingCutoff(real_parameters).estimate(ctx),
        "N8": NoiseAndAngle(real_parameters).estimate(ctx),
    }
    for label, reading in readings.items():
        assert isinstance(reading, Evidence), (label, reading)
        assert [a["key"] for a in reading.value["assumptions"]] == list(ASSUMPTION_KEYS)
        assert reading.value["says"]
        assert reading.value["baselines"]
        assert reading.quantity.startswith("reward."), label
    # The one measured parameter is recorded as measured and the rest as what they are.
    provenance = {
        row["parameter"]: row["source"] for row in readings["N5"].value["parameter_provenance"]
    }
    assert provenance["noise"] == "MEASURED"
    assert provenance["benefit"] == "SUPPLIED"
    assert provenance["risk_aversion"] == "SUPPLIED"
    assert provenance["effort"] == "ASSUMED"
    assert lint_instrument(OptimalWeights()) == []
    for cls in DECISION:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_layer_refuses_on_the_real_subject_when_the_grader_is_not_stationary(real_parameters):
    """`COMMITMENT_ONE_PERIOD` is the one assumption with a measurable half, and it is enforced."""
    drifting = Context(
        phase=Phase.PRE_RUN, regime_reading=RegimeReading.of(STATIONARY_GRADER=False)
    )
    for cls in (OptimalWeights, SortingCutoff, NoiseAndAngle):
        out = cls(real_parameters).estimate(drifting)
        assert isinstance(out, Refusal), cls.__name__
        assert out.reason is RefusalReason.ENVELOPE_VIOLATED
        assert "STATIONARY_GRADER" in out.detail


# ---------------------------------------------------------------------------
# The two theorems, on parameters the real subject supplies where it can
# ---------------------------------------------------------------------------


def test_the_unmeasurable_task_correction_on_the_real_noisiest_component(real_parameters):
    """The measured `sigma^2` of a real reward model, against a task nobody can measure.

    The unmeasured half of quality is stated rather than measured, because by construction there is
    no signal for it. What is real is `sigma_1^2`: the gauge variance of `skywork-v2-qwen3-8b` on
    1,327 prompts.

    Both halves of the correction fire and they pull opposite ways, which is worth seeing on real
    numbers. Effort substitutes between the two tasks at `C_12/C_22 = 0.5`, so the numerator halves,
    from 1 to 0.5. And the Schur complement `C_11 - C_12^2/C_22 = 0.75` is below `C_11 = 1`, so the
    denominator falls too, from `1 + r sigma^2` to `1 + 0.75 r sigma^2`: the agent responds harder
    to incentives on the measurable task precisely because it can shed effort onto the other one.
    The numerator wins, and the net is a 34% discount rather than the 50% the numerator alone would
    suggest.
    """
    sigma1 = float(real_parameters.noise[2, 2])
    corr = unmeasurable_correction(
        benefit=[1.0, 1.0],
        cost_curvature=[[1.0, 0.5], [0.5, 1.0]],
        noise_variance=sigma1,
        risk_aversion=real_parameters.risk_aversion,
        names=("skywork-v2-qwen3-8b", "the unmeasured half of quality"),
    )
    assert corr.substitution == pytest.approx(0.5)
    assert corr.numerator == pytest.approx(0.5)
    assert corr.schur == pytest.approx(0.75)
    assert corr.weight < corr.weight_ignoring
    assert corr.discount == pytest.approx(0.3415, abs=1e-3)
    assert corr.denominator < 1.0 + real_parameters.risk_aversion * sigma1
    assert "cannot be measured" in corr.says()


def test_the_zero_weight_theorem_is_reachable_and_is_exactly_zero(real_parameters):
    """Perfect substitutes plus one unmeasurable task of equal value, at the real measured noise.

    `C_11 = C_12 = C_22 = 1` makes the substitution rate exactly 1 and the Schur complement exactly
    0, so the numerator is `1 - 1*1 = 0` and the denominator is `1 + r*sigma^2*0 = 1`. The optimum is
    `0.0/1.0`, which is 0.0 and not 1e-17: the assertion is `== 0.0` because a tolerance here would
    hide the content of the theorem, which is that the answer is zero rather than small.

    And any positive weight is worse than none, unboundedly so: along a family approaching perfect
    substitutability the surplus at a fixed weight of 0.1 falls without bound while the surplus at
    zero stays exactly zero.
    """
    sigma1 = float(real_parameters.noise[2, 2])
    corr = unmeasurable_correction(
        benefit=[1.0, 1.0],
        cost_curvature=[[1.0, 1.0], [1.0, 1.0]],
        noise_variance=sigma1,
        risk_aversion=real_parameters.risk_aversion,
        names=("skywork-v2-qwen3-8b", "the unmeasured half of quality"),
    )
    assert corr.weight == 0.0
    assert corr.zero_weight is True
    assert "exactly zero, not small" in corr.says()

    at = []
    for rho in (0.9, 0.99, 0.999):
        near = unmeasurable_correction(
            benefit=[1.0, 1.0],
            cost_curvature=[[1.0, rho], [rho, 1.0]],
            noise_variance=sigma1,
            risk_aversion=real_parameters.risk_aversion,
        )
        assert two_task_surplus(0.0, near) == 0.0
        assert two_task_surplus(0.1, near) < 0.0
        at.append(two_task_surplus(0.1, near))
    assert at == sorted(at, reverse=True)


# ---------------------------------------------------------------------------
# The kill conditions, answered
# ---------------------------------------------------------------------------


def test_n5_kill_condition_does_not_fire(real_parameters):
    """*If the recommended weights match the value-proportional baseline, the noise term is inert.*

    They do not match, and the gap is not a refinement. The noisiest of three real reward models is
    recommended at 14% of what its value alone would buy while the crispest keeps 93% of its, and
    the value-weighted contract scores **below zero** on the principal's own objective: the risk it
    loads onto the agent exceeds the effort it buys. So the noise term is the difference between a
    contract that pays and one that costs, rather than between a good contract and a better one.
    """
    reading = recommend_weights(real_parameters)
    ratio = reading.weights / reading.baseline_value
    assert np.min(ratio) < 0.2
    assert np.max(ratio) > 0.9
    assert np.max(ratio) / np.min(ratio) > 5.0
    assert reading.surplus > 0.0 > reading.surplus_value
    assert reading.surplus_equal < reading.surplus_value
    assert "below doing nothing at all" in reading.says()


def test_n6_kill_condition_does_not_fire(real_parameters, composite_weights):
    """*If every real composite is already equal-compensating, the table is a formality.*

    This one is not, and not by a little: the starved component pays under four percent of what the
    best-paying component pays for the same unit of capacity.
    """
    table = equal_compensation(composite_weights, real_parameters)
    assert table.spread < 0.05
    assert not table.holds


def test_n7_kill_condition_does_not_fire(real_parameters):
    """*If the single-contract baseline is optimal, splitting never pays.*

    It is not optimal at any of three risk aversions spanning two decades, and the best number of
    contracts is two rather than one every time.
    """
    moved = 0
    for r in (0.1, 1.0, 10.0):
        p = ContractParameters(
            components=real_parameters.components,
            benefit=real_parameters.benefit,
            cost_curvature=real_parameters.cost_curvature,
            noise=real_parameters.noise,
            risk_aversion=r,
            sensitivity=real_parameters.sensitivity,
            effort=real_parameters.effort,
            source=real_parameters.source,
            note=real_parameters.note,
        )
        reading = sorting_cutoff(p, n_contracts=2)
        moved += int(reading.best_contract_count > 1 and reading.split_gain > 0.0)
    assert moved == 3
