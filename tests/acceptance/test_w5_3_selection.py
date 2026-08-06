"""Acceptance: C1's susceptibility triple and C2's heritability.

**The clause.** *The campaign's stored forecasts are recomputed with `β` and again filtered by
`h²`, and the ordering change is reported either way.*

It is discharged in two halves and the two halves resolve against different subjects, because the
campaign store contains no reinforcement-learning training run at all (E18: 27 stores
and 3,023 rows scanned, no per-step training series anywhere). So:

* the **`β` half** runs on the campaign's own `CHI-DRIFT` bank, which is what the stored forecast
  was made on, and
* the **`h²` half is refused on that bank**, by name and with the numbers, rather than quietly
  dropped. `h² = G_ii/C_ii` needs `G = J F⁻¹ Jᵀ`, which needs a policy to differentiate; the bank is
  a fixed set of responses scored by two reward models and there is no policy in it at any operator.
  It resolves against the 200-step GRPO record instead, and there it refuses again, for a second and
  entirely different reason that is worth more than a number would have been.

---

## The `β` half, on the campaign bank

**The subject.** `campaign.features.banked`, 2,000 UltraFeedback prompts x 4 responses x 7 surface
features, scored by `skywork-v2-qwen3-8b` and `skywork-v2-llama31-8b`. The card splits the bank by
prompt identity before anything is estimated: `χ` reads half A and the realised best-of-16 drift
reads half B, so the forecast is genuinely predictive. Half A is 1,000 prompts and 4,000 responses.

**Everything below was reproduced from the store before anything was recomputed.** The stored `χ`
vectors come back to a maximum absolute difference of 4.43e-05 on `len_chars` at 4573.74, which is
9.7e-09 relative; the residue is the store's own float32 score sidecars against the float64 the
original ran in, and it is at the precision floor of the stored data rather than a disagreement.
E17 describes this reproduction as bit-exact and it is not quite: the scores are stored
as float32 and the chi vectors agree to nine significant figures, not to the bit. The stored
forecast itself reproduces exactly: `chi_bon_spearman` = 1.0000 (llama) and 0.9643 (qwen), and the
variance baseline 0.4286 and 0.3929, all four matching the stored numbers to the last digit.

**Which covariance operator, which is the question E17 settles.** The ledger's sampling
distribution is given as "the policy's own group at step `t`", so `C` is the **within-group**
covariance, and that is the primary reading here. The pooled operator is reported as the contrast,
because the size of the disagreement is the point:

| operator | n_D of 7 | ρ(S, β) | sign flips | max rank change |
|---|---|---|---|---|
| **within-group (primary)** | 3.474 | **+0.9643** | 1 (llama), 0 (qwen) | 1 |
| pooled (contrast) | 3.332 | +0.4286 | 1 (both) | 5 |

`ρ = +0.9643` on seven items is exactly one adjacent transposition, which is the discreteness of
Spearman at `k = 7`. So under the operator the specification means, `β` and `S` differ by a single
swap on this data; under the shipped index's operator they reorder decisively.

**A defect in E17's own table, found reproducing it.** Those numbers only come out that way after
the features are variance-standardised, and E17 does not say so. On the raw covariance the same
arithmetic gives ρ(S, β) = +0.4286 (llama) and +0.6071 (qwen) within-group with a maximum rank
change of 6, which is the opposite verdict. The standardisation is right and it is load-bearing:
`len_chars` is counted in characters and `hedging_rate` is a proportion, so an ordering of raw `|β|`
is an ordering of units. It has to be stated. Both forms ship on every reading here.

**The suppressor, which `χ` cannot see.** Under the pooled operator `len_tokens` has the second
largest positive `S` and a **negative** `β`, holding in 2,000 of 2,000 prompt-level bootstrap
resamples on both models. It correlates 0.974 with `len_chars` and the joint fit gives the whole
positive direct effect to `len_chars`, whose standardised `β` rises from +4.19 to +10.34.

**And the answer to the clause: recomputing the forecast with `β` makes it worse, on every
operator.**

| model | operator | ρ(S, drift) | ρ(β, drift) | Δ |
|---|---|---|---|---|
| llama31-8b | **within (primary)** | +0.9643 | +0.9286 | **−0.0357** |
| qwen3-8b | **within (primary)** | +1.0000 | +0.9643 | **−0.0357** |
| llama31-8b | pooled | +1.0000 | +0.4286 | −0.5714 |
| qwen3-8b | pooled | +0.9643 | +0.3571 | −0.6071 |

Over 2,000 prompt-level bootstrap draws resampling both halves in parallel, the pooled loss is
−0.5997 [−0.7857, −0.4643] on llama and −0.5836 [−0.7857, −0.4286] on qwen, with `β` beating `S` in
**0 of 2,000** draws on either model. The within-group loss is −0.2588 [−0.7500, +0.0366] and
−0.2404 [−0.7143, +0.0714], spanning zero, with `β` ahead in 7.2% and 8.5% of draws.

**That is not a failure of `β`, and the reason is the third member of the triple.** The realised
quantity being forecast is a *response*, and a response is `Gβ`, not `β`. Best-of-n
selection on a fixed bank moves the feature mean by reweighting responses that already exist, so
every direction of feature variance is reachable and the response operator is `C` itself. Then
`Gβ = Cβ = S` identically, which is asserted below and holds to 1e-12. **On a fixed bank the
correct forecaster of drift is exactly `S`, and `β` answers a different question.** `β` can only
beat `S` where `G ≠ C`, which requires a policy parameterisation, which this bank does not have.
So C1's kill condition ("if `β` never reorders features against `S` on stored data, the transplant
is elegant and inert") does **not** fire: `β` reorders, decisively under one operator and by one
swap under the other. What this subject cannot test is whether the reordering is an improvement.

## The `h²` half

Refused on the bank, for the reason above, and refused again on the 200-step GRPO record, where the
rung-0 realised estimator returns `ĥ²` of 1.21e+04, 4.27e+03 and 3.08e+04 on the three varying
surface features against a theoretical upper bound of 1. That is not a large heritability. The
cumulative selection term is 9.6e-07 of the cumulative motion on `response_chars` and every
consecutive step pair draws a disjoint prompt set, so `Δz` is a difference between two task samples
rather than between two policies and the breeder's equation relates nothing to nothing. The price
ledger measured `Λ` = 0.0012 [0.0000, 0.0195] on the same record independently. **The subject this half
needs is a training run whose consecutive steps share their prompts and whose step size makes `η·S`
a material part of the motion**, or a Fisher solve for `G` at rung 2, which is `measure/efficiency/`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.indices.chi import (
    differential,
    feature_covariance,
    selection_gradient,
    susceptibility,
    susceptibility_triple,
)
from reward_lens.measure.indices.heritability import (
    RealisedHeritability,
    heritability,
    refuse_out_of_bounds,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run"
LONG_RUN = FIXTURES / "long"
LONG_ID = "run:f77bf75940ab982bbc35407af99cc094"

#: The two reward models CHI-DRIFT was confirmed on.
MODELS = ("skywork-v2-llama31-8b", "skywork-v2-qwen3-8b")

#: The campaign's own prompt split, `campaign/chidrift.py::_split_halves`: chi reads half A and the
#: realised drift reads half B, so the forecast is predictive rather than circular. Transcribed
#: rather than imported because the campaign package is not on this branch, and asserted against the
#: stored chi vectors below, which is what makes the transcription checkable.
CAMPAIGN_SEED = 20260716
BON_DRIFT_N = 16


def task_seed(*parts: str) -> int:
    import hashlib

    h = hashlib.blake2b("/".join(parts).encode(), digest_size=4).hexdigest()
    return (CAMPAIGN_SEED + int(h, 16)) % (2**31 - 1)


def split_halves(n_prompts: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(task_seed("chidrift", "split")).permutation(n_prompts)
    half = n_prompts // 2
    return perm[:half], perm[half : 2 * half]


def midranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(x, dtype=np.float64), method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = midranks(x), midranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bon_feature_drift(rewards: np.ndarray, features: np.ndarray, n: int) -> np.ndarray:
    """Realised best-of-n drift by concomitants of order statistics, as the campaign computed it.

    For each prompt the `m` responses are sorted ascending by reward and the rank-`r` response
    carries the probability that it is the best of `n` draws. Drift is the weighted feature mean
    minus the base feature mean, averaged over prompts.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    n_prompts, m = rewards.shape
    k = features.shape[2]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    weights = (ranks / m) ** n - ((ranks - 1) / m) ** n
    acc = np.zeros(k)
    for p in range(n_prompts):
        acc += weights @ features[p][np.argsort(rewards[p], kind="stable")]
    return acc / n_prompts - features.reshape(-1, k).mean(axis=0)


needs_store = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. Set REWARD_LENS_CAMPAIGN_STORE to the directory holding "
        "evidence.jsonl and payloads/."
    ),
)


@pytest.fixture(scope="module")
def bank() -> dict:
    """The CHI-DRIFT bank, the two models' scores, and the stored chi vectors and result."""
    from reward_lens.record.convert.store import CampaignStore

    store = CampaignStore(CAMPAIGN_STORE)
    banked = store.value(store.by_observable("campaign.features.banked")[0])
    scores = {}
    for row in store.by_observable("campaign.scores"):
        if row.extra.get("slice") == "ultrafeedback-bank":
            value = store.value(row)
            if value["layout"] == "bank":
                scores[row.roster_key] = np.asarray(value["scores"], dtype=np.float64)
    stored_chi = {}
    for row in store.by_observable("loops.tilt.susceptibility"):
        # The tilt payload decodes to the real `SusceptibilitySpectrum` when the class is
        # importable and to a field mapping when it is not, and both forms appear depending on
        # which branch the store is read from.
        spectrum = store.value(row)
        chi = spectrum["chi"] if isinstance(spectrum, dict) else spectrum.chi
        stored_chi[row.roster_key] = np.asarray(chi, dtype=np.float64)
    result = store.value(store.by_observable("campaign.result.CHI-DRIFT")[0])
    tensor = np.asarray(banked["tensor"], dtype=np.float64)
    idx_a, idx_b = split_halves(tensor.shape[0])
    return {
        "names": tuple(banked["names"]),
        "tensor": tensor,
        "scores": scores,
        "stored_chi": stored_chi,
        "result": result,
        "idx_a": idx_a,
        "idx_b": idx_b,
    }


# ---------------------------------------------------------------------------
# Reproduce the store before recomputing anything on it
# ---------------------------------------------------------------------------


@needs_store
def test_the_bank_is_the_one_the_card_measured(bank: dict) -> None:
    assert bank["tensor"].shape == (2000, 4, 7)
    assert bank["names"] == (
        "len_chars",
        "len_tokens",
        "markdown_density",
        "hedging_rate",
        "sentiment",
        "keyword_stuffing",
        "sycophancy_rate",
    )
    assert bank["idx_a"].size == 1000 and bank["idx_b"].size == 1000
    assert not set(bank["idx_a"]) & set(bank["idx_b"])
    assert set(bank["scores"]) >= set(MODELS)


@needs_store
def test_the_stored_chi_vectors_reproduce_from_the_store(bank: dict) -> None:
    """The anchor. Nothing below means anything if the stored numbers do not come back first.

    They come back to 9.7e-09 relative rather than to the bit, because the scores are stored as
    float32 sidecars and the original ran in float64. That is the precision floor of the stored
    data, and averaging 4,000 rows pulls the per-score float32 error of ~1e-07 down by sqrt(4000).
    """
    tensor, idx_a = bank["tensor"], bank["idx_a"]
    for model in MODELS:
        got = susceptibility(tensor[idx_a].reshape(-1, 7), bank["scores"][model][idx_a].reshape(-1))
        stored = bank["stored_chi"][model]
        rel = np.max(np.abs(got - stored) / np.maximum(np.abs(stored), 1e-300))
        assert rel < 1e-7, f"{model}: {rel:.3e}"
        assert np.sign(got).tolist() == np.sign(stored).tolist()


@needs_store
def test_the_stored_forecast_reproduces_exactly(bank: dict) -> None:
    """`chi_bon_spearman` and the variance baseline, both models, to the last stored digit."""
    tensor, idx_a, idx_b = bank["tensor"], bank["idx_a"], bank["idx_b"]
    per_model = bank["result"]["meta"]["per_model"]
    for model in MODELS:
        scores = bank["scores"][model]
        chi = susceptibility(tensor[idx_a].reshape(-1, 7), scores[idx_a].reshape(-1))
        drift = bon_feature_drift(scores[idx_b], tensor[idx_b], BON_DRIFT_N)
        variance = tensor[idx_a].reshape(-1, 7).var(axis=0)
        stored = per_model[model]
        assert spearman(chi, drift) == pytest.approx(stored["chi_bon_spearman"])
        assert spearman(variance, drift) == pytest.approx(stored["variance_baseline_spearman"])
        assert np.allclose(drift, np.asarray(stored["drift_n16"], dtype=np.float64), rtol=1e-6)


def triple_on(bank: dict, model: str, operator: str):
    """C1's triple on half A of the bank under one operator, through the shipped instrument path."""
    tensor, idx_a = bank["tensor"], bank["idx_a"]
    features = tensor[idx_a].reshape(-1, 7)
    values = bank["scores"][model][idx_a].reshape(-1)
    groups = np.repeat(np.arange(idx_a.size), tensor.shape[1])
    got = susceptibility_triple(
        features, values, bank["names"], group_ids=groups, operator=operator
    )
    assert not isinstance(got, Refusal), got
    return got


# ---------------------------------------------------------------------------
# E17's operator table: within-group primary, pooled as the contrast
# ---------------------------------------------------------------------------


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_within_group_operator_is_the_primary_reading(bank: dict, model: str) -> None:
    """The sampling distribution is the policy's own group, so `C` is within-group.

    Under it `β` and `S` differ by exactly one adjacent transposition on this bank, which is what
    `ρ = +0.9643` means at `k = 7`: Spearman is discrete there and 0.9643 is one swap.
    """
    got = triple_on(bank, model, "within_group")
    assert got.covariance.operator == "within_group"
    assert got.covariance.conditioning == pytest.approx(3.474, abs=5e-3)
    assert got.covariance.dof == 3000  # 4000 responses less 1000 prompt groups
    rho = spearman(got.differential_S_standardised, got.gradient_beta_standardised)
    assert rho == pytest.approx(0.9642857142857143)
    ranks_S = midranks(-got.differential_S_standardised)
    ranks_b = midranks(-got.gradient_beta_standardised)
    assert int(np.max(np.abs(ranks_S - ranks_b))) == 1
    assert len(got.sign_disagreements) <= 1


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_pooled_operator_is_the_contrast_and_it_reorders_decisively(
    bank: dict, model: str
) -> None:
    """The shipped index's operator. Same arithmetic, five rank places of disagreement."""
    got = triple_on(bank, model, "pooled")
    assert got.covariance.operator == "pooled"
    assert got.covariance.conditioning == pytest.approx(3.332, abs=5e-3)
    rho = spearman(got.differential_S_standardised, got.gradient_beta_standardised)
    assert rho == pytest.approx(0.42857142857142855)
    ranks_S = midranks(-got.differential_S_standardised)
    ranks_b = midranks(-got.gradient_beta_standardised)
    assert int(np.max(np.abs(ranks_S - ranks_b))) == 5
    assert got.sign_disagreements == ("len_tokens",)


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_suppressor_is_len_tokens_and_the_direct_effect_goes_to_len_chars(
    bank: dict, model: str
) -> None:
    """The textbook case, on real data: two features at r = 0.974 and the joint fit picks one."""
    got = triple_on(bank, model, "pooled")
    names = list(got.names)
    chars, tokens = names.index("len_chars"), names.index("len_tokens")
    corr = got.covariance.correlation[chars, tokens]
    assert corr == pytest.approx(0.974, abs=2e-3)
    # S says both are strongly and positively selected.
    assert got.differential_S[chars] > 0 and got.differential_S[tokens] > 0
    # beta gives the whole positive direct effect to len_chars and pushes len_tokens negative.
    assert got.gradient_beta[tokens] < 0
    assert got.gradient_beta_standardised[chars] > got.differential_S_standardised[chars]
    assert int(midranks(-got.gradient_beta_standardised)[tokens]) == 7


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_variance_standardisation_is_load_bearing_and_e17_does_not_state_it(
    bank: dict, model: str
) -> None:
    """E17's table is the standardised form. On raw units the same data gives another verdict.

    This is a defect in the erratum rather than in the mathematics: `β` in the features' own units
    is the defined estimator and it is what `Δz = ηGβ` consumes, and it is also not rankable
    across features counted in characters and in proportions. Both forms ship on the reading for
    exactly this reason.
    """
    got = triple_on(bank, model, "within_group")
    raw = spearman(got.differential_S, got.gradient_beta)
    standardised = spearman(got.differential_S_standardised, got.gradient_beta_standardised)
    assert standardised == pytest.approx(0.9642857142857143)
    assert raw < 0.65, raw
    assert standardised - raw > 0.3
    # The two differ only by a positive diagonal rescaling, so signs are identical either way.
    assert np.array_equal(np.sign(got.gradient_beta), np.sign(got.gradient_beta_standardised))


@needs_store
def test_n_D_on_the_raw_covariance_is_uninformative_which_is_why_it_is_not_the_primary(
    bank: dict,
) -> None:
    """The published formula, applied literally, reads 1.002 of 7 on this bank.

    `λ_max` of the raw covariance is the `len_chars` variance and nothing else registers. The
    correlation form is what measures the multicollinearity `n_D` exists to flag, and the raw form
    ships beside it so the choice is auditable.
    """
    got = triple_on(bank, MODELS[0], "within_group")
    assert got.covariance.conditioning_raw == pytest.approx(1.0, abs=0.01)
    assert got.covariance.conditioning > 3.0


# ---------------------------------------------------------------------------
# The clause, first half: the stored forecast recomputed with beta
# ---------------------------------------------------------------------------


def forecast_with(bank: dict, model: str, operator: str) -> dict[str, float]:
    """ρ(S, drift) and ρ(β, drift) on one model under one operator, standardised throughout.

    The drift is divided by the same per-feature standard deviation `β` is multiplied by, so both
    sides of the correlation are in standard deviations of their own feature. Mixing a standardised
    predictor with a raw outcome would rank the outcome by feature scale and the predictor by
    something else.
    """
    tensor, idx_a, idx_b = bank["tensor"], bank["idx_a"], bank["idx_b"]
    scores = bank["scores"][model]
    got = triple_on(bank, model, operator)
    drift = bon_feature_drift(scores[idx_b], tensor[idx_b], BON_DRIFT_N)
    sd = tensor[idx_a].reshape(-1, 7).std(axis=0)
    drift_std = drift / sd
    return {
        "rho_S": spearman(got.differential_S_standardised, drift_std),
        "rho_beta": spearman(got.gradient_beta_standardised, drift_std),
    }


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_forecast_recomputed_with_beta_under_the_specifications_operator(
    bank: dict, model: str
) -> None:
    """The clause, primary reading: `β` loses to `S` by one adjacent swap, and that is the answer.

    Δρ = −0.0357 on both models, which at `k = 7` is one transposition and nothing more. The
    ordering change is reported rather than characterised as an improvement: over 2,000 prompt-level
    bootstrap draws the interval spans zero (−0.7500 to +0.0366 on llama, −0.7143 to +0.0714 on
    qwen), so this subject cannot separate the two forecasters under this operator.
    """
    got = forecast_with(bank, model, "within_group")
    assert got["rho_S"] >= 0.96
    assert got["rho_beta"] >= 0.92
    delta = got["rho_beta"] - got["rho_S"]
    assert delta == pytest.approx(-1 / 28, abs=1e-9)  # exactly one adjacent swap at k = 7


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_forecast_recomputed_with_beta_under_the_shipped_operator(
    bank: dict, model: str
) -> None:
    """The clause, contrast reading: under the pooled operator `β` loses decisively.

    Δρ is −0.5714 on llama and −0.6071 on qwen, and over 2,000 prompt-level bootstrap draws `β`
    beat `S` in 0 of 2,000 on either model.
    """
    got = forecast_with(bank, model, "pooled")
    assert got["rho_S"] >= 0.96
    assert got["rho_beta"] <= 0.45
    assert got["rho_beta"] - got["rho_S"] <= -0.55


@needs_store
@pytest.mark.parametrize("model", MODELS)
def test_the_response_Gbeta_recovers_S_exactly_when_every_direction_is_reachable(
    bank: dict, model: str
) -> None:
    """Why `β` was never the right forecaster here, asserted rather than argued.

    Best-of-n selection on a fixed bank moves the feature mean by reweighting responses that already
    exist, so no direction of feature variance is unreachable and the response operator is `C`
    itself. Then `Gβ = Cβ = S` identically, and the member of C1's triple that forecasts drift is
    the response, which on this subject **is** `S`. `β` can only beat `S` where `G ≠ C`, and that
    needs a policy parameterisation the bank does not have.
    """
    tensor, idx_a = bank["tensor"], bank["idx_a"]
    features = tensor[idx_a].reshape(-1, 7)
    values = bank["scores"][model][idx_a].reshape(-1)
    groups = np.repeat(np.arange(idx_a.size), tensor.shape[1])
    C = feature_covariance(features, bank["names"], group_ids=groups, operator="within_group")
    got = susceptibility_triple(
        features,
        values,
        bank["names"],
        group_ids=groups,
        operator="within_group",
        metric_G=C.matrix,
        response_source="C, the fixed-bank limit where every direction is reachable",
    )
    assert not isinstance(got, Refusal)
    assert got.response is not None
    scale = float(np.max(np.abs(got.differential_S)))
    assert np.allclose(got.response, got.differential_S, rtol=1e-9, atol=1e-12 * scale)
    assert spearman(got.response, got.differential_S) == pytest.approx(1.0)


@needs_store
def test_beta_reorders_so_C1s_kill_condition_does_not_fire(bank: dict) -> None:
    """C1 dies if `β` never reorders features against `S` on stored data. It reorders."""
    moved = 0
    for model in MODELS:
        for operator in ("within_group", "pooled"):
            got = triple_on(bank, model, operator)
            rho = spearman(got.differential_S_standardised, got.gradient_beta_standardised)
            if rho < 1.0 or got.sign_disagreements:
                moved += 1
    assert moved == 4, "beta reordered against S on every model and both operators"


# ---------------------------------------------------------------------------
# The clause, second half: h2 is refused on the bank rather than dropped
# ---------------------------------------------------------------------------


@needs_store
def test_the_h2_half_is_refused_on_the_bank_and_the_refusal_names_what_is_missing(
    bank: dict,
) -> None:
    """`h² = G_ii/C_ii` needs `G = J F⁻¹ Jᵀ`. There is no policy in this store to differentiate.

    The clause asks for the forecasts to be "again filtered by `h²`" and that filter cannot be
    built here at any covariance operator, which E17 states as its own last limit: this is `C`, not
    `G`. The honest discharge is a refusal carrying the reason and the remedy, not a silently
    dropped half, so the refusal is constructed and asserted rather than described in prose.
    """
    got = triple_on(bank, MODELS[0], "within_group")
    C = got.covariance.matrix
    # The only matrix this subject offers in place of G is C itself, and h2 = C_ii/C_ii = 1 for
    # every feature. That is not a measurement: it is the statement that a fixed bank has no
    # unreachable variance because it has no parameters, and it filters nothing.
    degenerate = heritability(C, C, got.names, rung=0, method="C_in_place_of_G")
    assert not isinstance(degenerate, Refusal)
    assert np.allclose(degenerate.h2, 1.0)
    assert degenerate.inert == ()

    refusal = Refusal(
        instrument="FeatureHeritability",
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            "the campaign evidence store holds reward models scored on fixed banks and no "
            "reinforcement-learning training run at all (E18), so there is no policy "
            "to take a Jacobian or a Fisher against and G = J F^-1 J^T does not exist here. "
            "Substituting C gives h2 = 1 on all 7 features, which filters nothing"
        ),
        remedy=(
            "compute G against a policy checkpoint from a real training run: k Jacobian-vector "
            "products plus k Fisher solves by conjugate gradient, reporting the damping lambda and "
            "confirming the solution is stable across a decade of it. On a record with no "
            "checkpoint, read the rung-0 realised estimator over a step window instead."
        ),
        statistics={"n_features": len(got.names), "h2_with_C_for_G": 1.0},
    )
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "E18" in refusal.detail
    assert refusal.is_bounded is False


# ---------------------------------------------------------------------------
# The h2 half resolves against the 200-step record, and refuses there too
# ---------------------------------------------------------------------------

needs_record = pytest.mark.skipif(
    not (LONG_RUN / "steps").exists() and not LONG_RUN.exists(),
    reason=f"the 200-step GRPO record is not at {LONG_RUN}",
)


@pytest.fixture(scope="module")
def ledgers() -> list:
    from reward_lens.measure.ledger.features import SurfaceFeatures
    from reward_lens.measure.ledger.price import (
        learning_rates,
        ledger_series,
        steps_from_run,
        whole_run,
    )
    from reward_lens.record.reader import open_run

    run = open_run(str(LONG_RUN), LONG_ID)
    window = whole_run(run)
    samples = steps_from_run(run, SurfaceFeatures(), window=window)
    return ledger_series(samples, eta_by_step=learning_rates(run, window))


@needs_record
def test_the_record_is_the_one_D14_describes(ledgers: list) -> None:
    assert len(ledgers) == 199  # 200 steps, so 199 consecutive pairs
    assert ledgers[0].names == (
        "response_chars",
        "response_words",
        "mean_word_length",
        "type_token_ratio",
        "n_turns",
    )
    assert ledgers[0].n_groups == 2  # eight rollouts per step at K = 4
    # Every consecutive pair draws a disjoint prompt set, which is half of why the ratio below is
    # meaningless: Delta z is a difference between two task samples as well as between two policies.
    assert float(np.median([led.task_overlap for led in ledgers])) == 0.0


@needs_record
def test_realised_h2_on_the_record_is_out_of_bounds_and_refuses(ledgers: list) -> None:
    """The rung-0 estimator returns 1.2e+04 against an upper bound of 1, and refuses.

    `C ⪰ G` bounds `h²` in [0, 1] as a theorem, so this is not a large heritability and it is not an
    outlier: the numerator and the denominator are not related by the breeder's equation on this
    record. The number that says so is the selection share, 9.6e-07 on `response_chars`, and it is
    in the refusal because it is what a reader acts on.
    """
    inst = RealisedHeritability.from_ledgers(ledgers, enforce_bounds=False)
    reading = inst.compute()
    assert not isinstance(reading, Refusal)
    by_name = dict(zip(reading.names, reading.h2))
    assert by_name["response_chars"] > 1e3
    assert by_name["response_words"] > 1e3
    assert by_name["mean_word_length"] > 1e3
    # The two features with no within-group spread never entered the ledger's selection term, so
    # they carry no estimate at all. Undefined is not out of bounds and they are named separately.
    assert np.isnan(by_name["type_token_ratio"])
    assert np.isnan(by_name["n_turns"])
    assert reading.undefined == ("type_token_ratio", "n_turns")
    share = dict(zip(reading.names, reading.selection_share))
    assert share["response_chars"] < 1e-5

    refusal = refuse_out_of_bounds(reading)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ENVELOPE_VIOLATED
    assert set(refusal.statistics["out_of_bounds"]) == {
        "response_chars",
        "response_words",
        "mean_word_length",
    }
    assert refusal.statistics["min_selection_share"] < 1e-5
    # The remedy names the subject the claim actually needs rather than restating the failure.
    assert "share their prompts" in refusal.remedy
    assert "rung 2" in refusal.remedy

    # And the instrument refuses by default: the number is never handed back unguarded.
    assert isinstance(RealisedHeritability.from_ledgers(ledgers).compute(), Refusal)


@needs_record
def test_the_within_group_C_on_the_record_is_rank_deficient_and_beta_refuses(
    ledgers: list,
) -> None:
    """D14's first constraint, measured: eight rollouts in two groups is six contrasts for five
    features, and two of the five never vary, so `C⁻¹S` does not exist at a single step.

    The refusal names the constant features, because dropping them by name is the fix and a
    pseudo-inverse would have silently projected the answer onto whatever happened to move.
    """
    from reward_lens.measure.ledger.features import SurfaceFeatures
    from reward_lens.measure.ledger.price import steps_from_run, whole_run
    from reward_lens.record.reader import open_run

    run = open_run(str(LONG_RUN), LONG_ID)
    window = whole_run(run)
    samples = steps_from_run(run, SurfaceFeatures(), window=window)

    step = samples[0]
    C = feature_covariance(
        step.features, step.names, group_ids=step.group_ids, operator="within_group"
    )
    assert C.dof == 6
    assert C.constant_features == ("type_token_ratio", "n_turns")
    S = differential(
        step.features, step.advantages, group_ids=step.group_ids, operator="within_group"
    )
    refusal = selection_gradient(S, C)
    assert isinstance(refusal, Refusal)
    assert refusal.statistics["constant_features"] == ["type_token_ratio", "n_turns"]

    # Pooling the covariance over the whole run raises the degrees of freedom to 1,200 and does not
    # fix it, because the two features are constant in every group of every step. Rank stays 3 of 5.
    stacked = np.vstack([s.features for s in samples])
    gids = np.concatenate(
        [s.group_ids + 1000 * i for i, s in enumerate(samples)]  # keep step groups distinct
    )
    pooled = feature_covariance(stacked, samples[0].names, group_ids=gids, operator="within_group")
    assert pooled.dof == 1200
    assert np.linalg.matrix_rank(pooled.matrix) == 3
    assert pooled.constant_features == ("type_token_ratio", "n_turns")


@needs_record
def test_the_three_varying_features_do_admit_a_beta_once_the_basis_is_stated(
    ledgers: list,
) -> None:
    """Dropping the two constants by name is the stated choice D14 asks for, and it works.

    This is the honest form of the rung-1 reading on this record: a three-feature basis, pooled over
    the run, with the basis on the reading because the basis is part of the claim.
    """
    from reward_lens.measure.ledger.features import SurfaceFeatures
    from reward_lens.measure.ledger.price import steps_from_run, whole_run
    from reward_lens.record.reader import open_run

    run = open_run(str(LONG_RUN), LONG_ID)
    window = whole_run(run)
    samples = steps_from_run(run, SurfaceFeatures(), window=window)
    keep = [
        samples[0].names.index(n) for n in ("response_chars", "response_words", "mean_word_length")
    ]
    names = tuple(samples[0].names[i] for i in keep)

    features = np.vstack([s.features[:, keep] for s in samples])
    advantages = np.concatenate([s.advantages for s in samples])
    gids = np.concatenate([s.group_ids + 1000 * i for i, s in enumerate(samples)])

    got = susceptibility_triple(features, advantages, names, group_ids=gids)
    assert not isinstance(got, Refusal), got
    assert got.basis == names
    assert got.covariance.dof == 1200
    assert got.covariance.operator == "within_group"
    assert np.all(np.isfinite(got.gradient_beta))
    assert 1.0 < got.covariance.conditioning < 3.0
    assert got.covariance.conditioning == pytest.approx(1.704, abs=5e-3)

    # `measure.ledger.features.SurfaceFeatures` puts response_chars and response_words in the basis
    # on the grounds that they "correlate at roughly 0.97 on ordinary prose", which is exactly the
    # near-collinear pair that makes S and beta diverge. On this record they correlate 0.385
    # within-group, and the strongest pair is response_words against mean_word_length at -0.708.
    # A 0.6M-parameter model on a toy length grader does not write ordinary prose, and the
    # divergence this basis was chosen to exercise is weaker here than the docstring implies.
    corr = got.covariance.correlation
    assert corr[0, 1] == pytest.approx(0.385, abs=5e-3)
    assert corr[1, 2] == pytest.approx(-0.708, abs=5e-3)
    # S and beta still disagree in ordering, which is the point that survives: response_words
    # outranks response_chars on beta and not on S.
    assert got.differential_S[0] > got.differential_S[1]
    assert got.gradient_beta[1] > got.gradient_beta[0]


# ---------------------------------------------------------------------------
# The generated invariance tests, and the lint gate
# ---------------------------------------------------------------------------


def test_beta_is_covariant_under_reward_affine_with_weight_one() -> None:
    """`Cov(f, a·r + b) = a·Cov(f, r)` and `C` does not move, so `β` scales by `a`.

    Rule four of the build: no instrument merges without its generated invariance test passing. The
    reading is a vector, so it is projected onto one coefficient, which is what `check_invariance`
    asks for when it cannot reduce a reading to a scalar itself.
    """
    from reward_lens.core.invariance import InvariancePayload, check_invariance
    from reward_lens.measure.indices.chi import SelectionGradientIndex

    rng = np.random.default_rng(11)
    features = rng.standard_normal((240, 4))
    groups = np.repeat(np.arange(60), 4)
    scores = features @ np.array([1.0, -0.5, 0.25, 0.0]) + 0.2 * rng.standard_normal(240)

    def run(_inst, payload: InvariancePayload) -> float:
        got = susceptibility_triple(
            features, payload.scores, tuple("abcd"), group_ids=payload.group_ids
        )
        assert not isinstance(got, Refusal)
        return float(got.gradient_beta[0])

    report = check_invariance(
        SelectionGradientIndex,
        "reward.affine",
        InvariancePayload(scores=scores, group_ids=groups),
        n=32,
        run=run,
    )
    assert report.passed, report.render()
    assert report.relation.status == "covariant" and report.relation.weight == 1.0


def test_beta_is_invariant_under_repr_basis() -> None:
    """The second relation, which E55 made declarable and which therefore now actually runs.

    An orthogonal map acting on both the activations and the bank's directions leaves every feature
    value alone, so every downstream quantity is unchanged. Before E55 this relation was recorded in
    a comment and generated no test.
    """
    from reward_lens.core.invariance import InvariancePayload, check_invariance, resolve_relation
    from reward_lens.measure.indices.chi import SelectionGradientIndex

    rng = np.random.default_rng(12)
    activations = rng.standard_normal((160, 6))
    readouts = rng.standard_normal((3, 6))
    groups = np.repeat(np.arange(40), 4)
    scores = activations @ readouts[0] + 0.1 * rng.standard_normal(160)

    def run(_inst, payload: InvariancePayload) -> float:
        feats = np.asarray(payload.activations) @ np.asarray(payload.readouts).T
        got = susceptibility_triple(feats, scores, tuple("xyz"), group_ids=payload.group_ids)
        assert not isinstance(got, Refusal)
        return float(got.gradient_beta[0])

    assert resolve_relation(SelectionGradientIndex, "repr.basis").status == "invariant"
    report = check_invariance(
        SelectionGradientIndex,
        "repr.basis",
        InvariancePayload(activations=activations, readouts=readouts, group_ids=groups),
        n=32,
        run=run,
    )
    assert report.passed, report.render()


def test_chi_now_declares_both_of_its_relations() -> None:
    """`chi` was `resolve_relation`'s own worked example and could not state the second half."""
    from reward_lens.core.invariance import resolve_relation
    from reward_lens.measure.indices.chi import Chi

    assert resolve_relation(Chi, "reward.affine").status == "covariant"
    assert resolve_relation(Chi, "reward.affine").weight == 1.0
    assert resolve_relation(Chi, "repr.basis").status == "invariant"


def test_every_new_instrument_passes_lint() -> None:
    from reward_lens.measure.base import lint_instrument
    from reward_lens.measure.indices.chi import (
        Chi,
        SelectionConditioning,
        SelectionGradientIndex,
        SelectionResponseIndex,
    )
    from reward_lens.measure.indices.heritability import (
        Autonomy,
        Evolvability,
        FeatureHeritability,
        GConditioning,
    )

    for cls in (
        Chi,
        SelectionGradientIndex,
        SelectionResponseIndex,
        SelectionConditioning,
        FeatureHeritability,
        Evolvability,
        Autonomy,
        GConditioning,
        RealisedHeritability,
    ):
        assert lint_instrument(cls) == [], cls.name


def test_C1_and_C2_cover_every_quantity_their_catalogue_records_name() -> None:
    """The four quantities on C1 and the four on C2, each declared by an instrument that exists."""
    from reward_lens.measure.indices.chi import (
        Chi,
        SelectionConditioning,
        SelectionGradientIndex,
        SelectionResponseIndex,
    )
    from reward_lens.measure.indices.heritability import (
        Autonomy,
        Evolvability,
        FeatureHeritability,
        GConditioning,
    )

    declared = {
        cls.quantity
        for cls in (
            Chi,
            SelectionGradientIndex,
            SelectionResponseIndex,
            SelectionConditioning,
            FeatureHeritability,
            Evolvability,
            Autonomy,
            GConditioning,
            RealisedHeritability,
        )
    }
    assert {
        "selection.differential_S",
        "selection.gradient_beta",
        "selection.response_Gbeta",
        "selection.G_conditioning",
        "selection.heritability_h2",
        "selection.autonomy",
        "selection.evolvability",
    } <= declared


def test_the_response_refuses_without_a_G_rather_than_substituting_C() -> None:
    """`Gβ` with `C` for `G` is `S` wearing a third name, so a missing `G` is a refusal."""
    from reward_lens.measure.indices.chi import SelectionResponseIndex

    got = SelectionResponseIndex().compute(None)  # type: ignore[arg-type]
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "Do not pass C here" in got.remedy


def test_the_differential_matches_the_ledgers_own_estimator() -> None:
    """C1's `S` and F1's `S` are one estimator, checked on the same arrays rather than assumed."""
    from reward_lens.measure.ledger.price import selection_differential

    rng = np.random.default_rng(13)
    features = rng.standard_normal((200, 3))
    advantages = rng.standard_normal(200)
    groups = np.repeat(np.arange(50), 4)
    mine = differential(features, advantages, group_ids=groups, operator="within_group")
    theirs = selection_differential(features, advantages, groups, ("a", "b", "c")).value
    assert np.allclose(mine, theirs, rtol=1e-12, atol=0.0)
