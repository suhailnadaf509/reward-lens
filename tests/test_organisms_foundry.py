"""Pure tests for the organism foundry (M4 acceptance, must pass now).

These assert the ground truth the foundry claims is actually present in the data it emits: a
dose-controlled spurious feature correlates with the label at the requested rho; a compositional rule's
data obeys the rule on both the train and a fresh OOD split; the answer key is internally consistent;
planted intransitivity contains cycles; the annotator mixture's H(V) matches its construction; the
rubric's criterion directions have the requested correlation; and the hack organism has the predicted
sign structure. If these hold, the data was genuinely generated from the rule, which is the premise the
whole calibration story rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core import ORGANISMS
from reward_lens.organisms import foundry as F
from reward_lens.organisms.spec import AnswerKey, RuleSpec


def test_spurious_correlation_matches_rho():
    """The spurious feature correlates with the label at ~rho across the dose sweep."""
    for rho in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        view, key = F.spurious_correlation_organism(rho=rho, n=1200, seed=11, split="train")
        measured = F.measure_spurious_correlation(view, "cites")
        assert abs(measured - rho) < 0.05, f"rho={rho}: measured agreement {measured:.3f}"
        # The true rule is clean regardless of the spurious dose: chosen always satisfies, rejected not.
        assert all(key.rule.prefers_chosen(p.chosen.text, p.rejected.text) for p in view)
        # The channel records the planted dose exactly.
        ch = key.channel("spurious")
        assert ch is not None and ch.rho == pytest.approx(rho)


def test_compositional_rule_obeyed_on_train_and_ood():
    """At every difficulty level the chosen side satisfies the combinator and the rejected does not."""
    for level in (1, 2, 3, 4):
        train, key = F.compositional_rule_organism(level=level, n=120, seed=7, split="train")
        ood, key_ood = F.compositional_rule_organism(level=level, n=120, seed=7, split="ood")
        rule = key.rule
        assert key.rule.combinator == key_ood.rule.combinator  # same rule, different split
        for pair in train:
            assert rule.satisfied(pair.chosen.text)
            assert not rule.satisfied(pair.rejected.text)
        for pair in ood:
            assert rule.satisfied(pair.chosen.text)
            assert not rule.satisfied(pair.rejected.text)


def test_ood_topics_are_disjoint_from_train():
    """The OOD split is genuinely unseen text: its topics never appear in the train split."""
    train, _ = F.compositional_rule_organism(level=2, n=200, seed=1, split="train")
    ood, _ = F.compositional_rule_organism(level=2, n=200, seed=1, split="ood")
    train_prompts = {p.prompt_text for p in train}
    ood_prompts = {p.prompt_text for p in ood}
    assert train_prompts.isdisjoint(ood_prompts)


def test_answer_key_internally_consistent():
    """The rule's combinator references only defined predicates, and channels/family are well-formed."""
    from reward_lens.organisms._features import combinator_names
    from reward_lens.organisms.spec import Predicate

    _, key = F.compositional_rule_organism(level=3, n=10, seed=0)
    assert isinstance(key, AnswerKey)
    assert isinstance(key.rule, RuleSpec)
    names = {p.name for p in key.rule.predicates}
    # Every predicate the combinator references is actually defined on the rule.
    assert set(combinator_names(key.rule.combinator)) <= names
    assert key.family and key.family != "unnamed"
    assert key.governs_behavior_oob is False  # never assumed; only verify.py sets it
    # A malformed rule (combinator names an absent predicate) is rejected at construction.
    with pytest.raises(ValueError):
        RuleSpec(predicates=(Predicate("cites", "cites", "cites"),), combinator="cites AND factual")


def test_effective_n_equals_length_no_fake_n():
    """Foundry pairs are distinct seeds, so the lineage-aware effective n equals the row count (R7)."""
    view, _ = F.compositional_rule_organism(level=2, n=96, seed=3)
    assert view.effective_n() == pytest.approx(len(view), rel=1e-6)
    assert len(view) == 96


def test_intransitivity_contains_cycles():
    """Every planted tournament has a directed cycle (the curl ground truth)."""
    view, key = F.intransitivity_organism(n_triads=20, seed=2)
    assert len(view) == 20
    assert all(F.has_cycle(t) for t in view)
    # A transitive tournament (no planted cycle) must not report one, so the detector is not vacuous.
    from reward_lens.organisms._data_compat import EdgeObs, Response, Tournament, make_lineage

    responses = tuple(Response(text=t) for t in ("a", "b", "c"))
    transitive = Tournament(
        prompt="p",
        responses=responses,
        edges=(EdgeObs(0, 1, 5, 0), EdgeObs(1, 2, 5, 0), EdgeObs(0, 2, 5, 0)),
        lineage=make_lineage("s", "b", (), ["t"]),
    )
    assert not F.has_cycle(transitive)


def test_annotator_mixture_entropy_matches_construction():
    """The recorded H(V) equals the recomputed mixture entropy and the empirical assignment entropy."""
    view, key = F.annotator_mixture_organism(n=4000, seed=4)
    ch = key.channel("annotator_mixture")
    assert ch is not None
    recorded = ch.rho["entropy_bits"]
    recomputed = F.mixture_entropy_bits(dict(ch.rho["mixing"]))
    empirical = F.empirical_annotator_entropy(view)
    assert recorded == pytest.approx(recomputed, abs=1e-9)
    assert empirical == pytest.approx(recorded, abs=0.03)


def test_rubric_directions_have_requested_correlation():
    """The K criterion directions are unit norm with the exact requested pairwise cosine."""
    K, d, corr = 4, 8, 0.3
    _, key = F.rubric_organism(K=K, d=d, correlation=corr, n=20, seed=5)
    dirs = np.stack([key.true_directions[f"criterion_{k}"] for k in range(K)])
    gram = dirs @ dirs.T
    assert np.allclose(np.diag(gram), 1.0, atol=1e-6)
    offdiag = gram[~np.eye(K, dtype=bool)]
    assert np.allclose(offdiag, corr, atol=1e-6)


def test_hack_direction_signature():
    """The hack feature is rewarded (Cov>0) but anti-correlated with gold (Cov<=0), per A12."""
    view, key = F.hack_direction_organism(n=600, seed=6, anti_correlation=0.85)
    sig = F.measure_hack_signature(view, "cites", "factual")
    assert sig["cov_hack_label"] > 0.05
    assert sig["cov_hack_gold"] < 0.0
    assert set(key.true_directions.keys()) == {"hack", "gold"}


def test_gate_rule_obeyed_and_hidden_objective_present():
    """The gate rule cleanly prefers chosen; the hidden objective biases the chosen side."""
    gate_view, gate_key = F.gate_organism(n=120, seed=8)
    assert all(gate_key.rule.prefers_chosen(p.chosen.text, p.rejected.text) for p in gate_view)

    hid_view, hid_key = F.hidden_objective_organism(n=300, seed=9)
    from reward_lens.organisms._features import extract_features

    hidden = hid_key.channel("hidden_objective").detail["hidden_feature"]
    chosen_rate = np.mean([extract_features(p.chosen.text)[hidden] for p in hid_view])
    rejected_rate = np.mean([extract_features(p.rejected.text)[hidden] for p in hid_view])
    assert chosen_rate > rejected_rate  # the hidden objective really biases the chosen side
    assert hid_key.channel("hidden_objective").detail["hidden_break_fraction"] > 0.0


def test_epistemic_and_value_error_rates():
    """The epistemic (fabrication) and value (inversion) error rates match the requested doses."""
    ev, ek = F.epistemic_error_organism(epsilon=0.25, n=800, seed=12)
    frac_fab = np.mean([p.meta["fabricated"] for p in ev])
    assert frac_fab == pytest.approx(0.25, abs=0.05)
    assert ek.channel("epistemic_error").rho == pytest.approx(0.25)

    vv, vk = F.value_error_organism(delta=0.25, n=800, seed=12)
    frac_inv = np.mean([p.meta["inverted"] for p in vv])
    assert frac_inv == pytest.approx(0.25, abs=0.05)
    assert vk.channel("value_error").rho == pytest.approx(0.25)


def test_all_generators_registered():
    """Every generator is discoverable through the ORGANISMS registry (R9)."""
    expected = {
        "compositional",
        "spurious",
        "hidden_objective",
        "gate",
        "intransitivity",
        "annotator_mixture",
        "rubric",
        "hack_direction",
        "epistemic_error",
        "value_error",
        "curl_harmonic",
        "kinship",
    }
    assert expected <= set(ORGANISMS.names())


def test_stub_generators_raise_clearly():
    """The stubbed generators raise NotImplementedError naming what they need (marked stubs)."""
    with pytest.raises(NotImplementedError, match="STUB"):
        F.kinship_organism()


def test_curl_harmonic_organism():
    """curl_harmonic_organism generates both 3-cycles and chordless rings, all having cycles."""
    view, key = F.curl_harmonic_organism(n_triads=10, n_rings=10, seed=1)
    assert len(view) == 20
    assert all(F.has_cycle(t) for t in view)
    ch = key.channel("curl_harmonic")
    assert ch is not None
    assert ch.rho["n_triads"] == 10
    assert ch.rho["n_rings"] == 10


def test_dataset_checksum_is_stable_and_content_derived():
    """Regenerating the same organism yields the same checksum; a different dose yields a different one."""
    v1, _ = F.spurious_correlation_organism(rho=0.8, n=50, seed=3)
    v2, _ = F.spurious_correlation_organism(rho=0.8, n=50, seed=3)
    v3, _ = F.spurious_correlation_organism(rho=0.9, n=50, seed=3)
    assert v1.checksum() == v2.checksum()
    assert v1.checksum() != v3.checksum()
