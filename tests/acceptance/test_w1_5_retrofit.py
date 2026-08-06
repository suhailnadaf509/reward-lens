"""Acceptance: the shipped observables and indices, retyped onto the instrument contract.

What this file has to show is "a test enumerating the registry asserts every instrument has all
four", the four being a quantity id, an invariance group, an envelope and baselines.
`lint_instrument` is that check, plus the rule that an instrument's quantity must be registered, so
`test_every_instrument_passes_lint` below is the gate and the rest of this module exists to make its
failures readable.

Counts, measured rather than quoted. Eleven observables in `measure/battery/` and eighteen indices
in `measure/indices/`, twenty-nine together. Fourteen `LINEAR_READOUT` declaration sites inside
`measure/` before the retrofit; twelve after, two having been dropped from instruments that never
reach a readout vector.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.invariance import (
    GROUPS,
    check_invariance,
    parse_group_field,
    resolve_relation,
)
from reward_lens.core.quantity import QUANTITIES, BiasStatement, load_quantities
from reward_lens.core.types import Access, Capability, Component, Phase, Substrate
from reward_lens.measure.base import declared_access, declared_capabilities, lint_instrument

torch = pytest.importorskip("torch", reason="the battery requires the white-box extra")

from reward_lens.measure import battery as battery_pkg  # noqa: E402
from reward_lens.measure import indices as index_pkg  # noqa: E402

# ---------------------------------------------------------------------------
# The population under test
# ---------------------------------------------------------------------------

#: The eighteen index classes, in the order `measure/indices/__init__.py` documents them.
INDEX_NAMES = (
    "KUI",
    "Distortion",
    "CoverageDisparity",
    "TeacherCompatibility",
    "TailIndex",
    "VerificationScore",
    "StyleShare",
    "ReceiptReliance",
    "Skepticism",
    "Coherence",
    "DarkReward",
    "InterpCoverage",
    "Chi",
    "VCE",
    "Legibility",
    "EvalAwareness",
    "RobustnessSNR",
    "Contested",
)

MEASURE_DIR = pathlib.Path(battery_pkg.__file__).resolve().parent.parent


def battery_instruments() -> list:
    return [cls() for cls in battery_pkg.BATTERY]


def index_instruments() -> list:
    return [getattr(index_pkg, name)() for name in INDEX_NAMES]


def all_instruments() -> list:
    return battery_instruments() + index_instruments()


@pytest.fixture(scope="module", autouse=True)
def _registry() -> None:
    """The quantity registry is loaded on demand, not at import, so load it once here."""
    load_quantities()


# ---------------------------------------------------------------------------
# The counts, measured
# ---------------------------------------------------------------------------


def test_measured_population_is_eleven_and_eighteen() -> None:
    """Eleven battery observables, eighteen indices, twenty-nine together.

    Two counts were in circulation, 28 and 29. The count here is the population, measured off the
    tree rather than transcribed, and it is 29.
    """
    assert len(battery_instruments()) == 11
    assert len(index_instruments()) == 18
    assert len(all_instruments()) == 29
    names = [i.name for i in all_instruments()]
    assert len(set(names)) == 29, "two instruments share a name"


def test_linear_readout_declaration_count_after_the_drop() -> None:
    """Twelve `LINEAR_READOUT` capability declarations remain in `measure/`, down from fourteen.

    The two dropped are `ConflictMatrix`, which never touches a readout at all, and `Chi`, which
    reaches `find_readout` only through `readout_site` to decide where to capture and never reads a
    vector. Every other declarer reaches a vector, four of them transitively: `CircuitJaccard` runs
    `DirectLinearAttribution`, and `PatchGrid`, `PathEffect` and `ConceptDoseResponse` go through
    `run_patched_scores`, which reads `signal.readout(readout).vector`.
    """
    sites = [
        (p.relative_to(MEASURE_DIR).as_posix(), n)
        for p in _retrofit_sources()
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.match(r"\s*capabilities = Capability.*LINEAR_READOUT", line)
    ]
    assert len(sites) == 12, f"expected twelve declaration sites, found {sites}"

    dropped = {"ConflictMatrix", "Chi"}
    for inst in all_instruments():
        has = bool(declared_capabilities(inst) & Capability.LINEAR_READOUT)
        if inst.name in dropped:
            assert not has, f"{inst.name} still declares LINEAR_READOUT"


def _retrofit_sources() -> list[pathlib.Path]:
    """The modules the retrofit owns: the battery and the index library."""
    return sorted((MEASURE_DIR / "battery").glob("*.py")) + sorted(
        (MEASURE_DIR / "indices").glob("*.py")
    )


def _capability_valued_requires(paths) -> list[str]:
    return [
        f"{p.relative_to(MEASURE_DIR).as_posix()}:{n}"
        for p in paths
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.match(r"\s{4}requires = Capability", line)
    ]


def test_no_capability_valued_requires_remains_in_the_retrofitted_packages() -> None:
    """This half of the rename: nothing in the battery or the index library uses the old name."""
    offenders = _capability_valued_requires(_retrofit_sources())
    assert offenders == [], f"still declaring a Capability under `requires`: {offenders}"


def test_no_capability_valued_requires_remains_anywhere_in_measure() -> None:
    """The removal condition for the two compatibility accessors in `measure/base.py`.

    The instrument contract gives the name `requires` to the AccessMatrix and 2.0.1 gave it to the Capability
    flags. `declared_capabilities` and `declared_access` read both spellings during the transition
    and their docstrings name this count reaching zero as when they get deleted, so it is asserted
    rather than assumed. This test is wider than the retrofit's own path set on purpose: the
    accessors are shared, so a module outside the battery and the index library keeps them alive
    for everyone.
    """
    owned = set(_retrofit_sources())
    offenders = _capability_valued_requires(sorted(MEASURE_DIR.rglob("*.py")))
    outside = [o for o in offenders if (MEASURE_DIR / o.rsplit(":", 1)[0]).resolve() not in owned]
    assert offenders == [], f"still declaring a Capability under `requires`: {offenders}. " + (
        f"None of it is in the battery or the index library; {outside} belong to another "
        f"package and have to be renamed there before the accessors can go."
        if outside and len(outside) == len(offenders)
        else ""
    )


# ---------------------------------------------------------------------------
# The gate: every instrument passes lint
# ---------------------------------------------------------------------------


def test_every_instrument_passes_lint() -> None:
    """Every instrument in `measure/` has all four, and lint agrees.

    `lint_instrument` checks the four declarations the contract requires: a registered
    quantity, a non-empty baseline tuple, an envelope, and an invariance group. It returns findings
    rather than raising so the retype could land in one commit; an empty list for every instrument
    is what closes the retrofit.
    """
    findings = {i.name: lint_instrument(i) for i in all_instruments()}
    failed = {n: [f.render() for f in fs] for n, fs in findings.items() if fs}
    unregistered = sorted(
        {i.quantity for i in all_instruments() if i.quantity and i.quantity not in QUANTITIES}
    )
    assert not failed, (
        f"{len(failed)} of 29 instruments fail lint.\n"
        + "\n".join(line for fs in failed.values() for line in fs)
        + (
            f"\n\nAll of it is one cause: {len(unregistered)} quantity ids that no row in "
            f"spec/QUANTITIES.yaml carries, because the registry holds the 85 catalogue "
            f"instruments and not the shipped corpus. Register these and this test goes green "
            f"with no further edit here:\n  " + "\n  ".join(unregistered)
            if unregistered
            else ""
        )
    )


def test_declarations_present_even_where_the_quantity_row_is_missing() -> None:
    """Localises the failure above; it does not replace it.

    This asserts the four declarations exist and are well formed, which is the half of the work
    that lives in `measure/`. The half that lives in `spec/QUANTITIES.yaml` is the quantity row,
    and when that is missing this test passes while `test_every_instrument_passes_lint` fails,
    which is exactly the information a reader of a red run needs. It is not a softer gate: the gate
    is above.
    """
    for inst in all_instruments():
        assert inst.quantity, f"{inst.name} declares no quantity"
        # `selection.differential_S` carries a capital in its local part, as the registry prints
        # it, so the id pattern allows one rather than the id being renamed to fit a test.
        assert re.fullmatch(r"[a-z_]+\.[A-Za-z_0-9]+", inst.quantity), (
            f"{inst.name} declares {inst.quantity!r}, which is not a dotted registry id"
        )
        assert inst.baselines, f"{inst.name} declares no baseline"
        assert inst.envelope is not None, f"{inst.name} declares no envelope"
        assert inst.invariance, f"{inst.name} declares no invariance group"
        assert parse_group_field(inst.invariance), (
            f"{inst.name} declares invariance {inst.invariance!r}, which resolves to no group in "
            f"{sorted(GROUPS)}"
        )
        assert inst.invariance_relation is not None, (
            f"{inst.name} declares a group and no relation to it, so the generated test would "
            f"default to invariant and pass for the wrong reason"
        )


def test_baseline_ids_follow_the_bank_convention() -> None:
    """Every baseline id is a `baseline.*` token, and the ones the bank carries resolve against it.

    `stats/baselines/__init__.py` keys `BASELINES` by `BaselineID` "so an instrument's `baselines`
    tuple names entries that exist rather than strings somebody typed". Only one of the six the bank
    ships, `baseline.length`, is a control any of these twenty-nine instruments can use; the rest of
    what they name are per-instrument negative controls (a self-patch, a norm-matched random
    re-injection, a shuffled axis label) which are not general dumb baselines and do not belong in a
    six-entry bank. So this asserts the naming convention, which is what keeps the two vocabularies
    mergeable, and asserts the overlap resolves. Whether the per-instrument controls also get
    registered is a decision for whoever owns `stats/baselines/`.
    """
    from reward_lens.stats.baselines import BASELINES

    for inst in all_instruments():
        for bid in inst.baselines:
            assert bid.startswith("baseline."), (
                f"{inst.name} names baseline {bid!r}, which is not a `baseline.*` id"
            )

    named = {b for inst in all_instruments() for b in inst.baselines}
    assert named & set(BASELINES) == {"baseline.length"}


def test_every_instrument_declares_access_substrates_and_phases() -> None:
    """The other three of the six section-4.2 declarations, which lint does not cover."""
    for inst in all_instruments():
        access = declared_access(inst)
        assert access, f"{inst.name} declares no access matrix"
        for component, level in access.items():
            assert isinstance(component, Component)
            assert isinstance(level, Access) and level != Access.NONE
        assert inst.substrates, f"{inst.name} declares no substrates"
        assert all(isinstance(s, Substrate) for s in inst.substrates)
        assert inst.phases, f"{inst.name} declares no phases"
        assert all(isinstance(p, Phase) for p in inst.phases)
        assert inst.rung >= 0


def test_every_declared_regime_condition_is_measurable() -> None:
    """`EnvelopeSpec` enforces this at construction; asserting it here pins the mapping.

    A condition in `requires` that is absent from `measured_by` is a precondition nobody can check,
    which reads as rigour and enforces nothing. The construction-time rule means an instrument
    carrying such an envelope could not have imported, so this test is a regression guard on the
    shared `MEASURED_BY` table rather than a second implementation of the rule.
    """
    for inst in all_instruments():
        env = inst.envelope
        assert env is not None
        assert env.requires, f"{inst.name} declares an empty envelope without a justification"
        for condition in env.requires:
            assert isinstance(condition, RegimeCondition)
            assert condition in env.measured_by
            assert env.measured_by[condition], (
                f"{inst.name} names {condition.name} with an empty measuring quantity"
            )


# ---------------------------------------------------------------------------
# Chi, registered honestly rather than fixed
# ---------------------------------------------------------------------------


def test_chi_is_registered_as_the_shipped_differential_at_rung_zero() -> None:
    """`Chi` estimates `selection.differential_S`, at rung 0, with a bias statement that says why.

    The shipped index computes `Cov(f, r)` on a fixed base-policy bank at zero optimisation
    pressure. That is not fixed here: C1 is a Phase 5 package. What is done here is registering it
    for what it is, so the bias travels with every reading instead of living in a document.
    """
    chi = index_pkg.Chi()
    assert chi.quantity == "selection.differential_S"
    assert chi.quantity in QUANTITIES, "the one quantity of the 29 that the registry does carry"
    assert chi.rung == 0

    bias = chi.BIAS
    assert isinstance(bias, BiasStatement)
    assert bias.why.strip(), "a bias with no `why` tells a reader nothing they can act on"
    assert bias.direction == "unknown"

    why = bias.why.lower()
    # The two things wrong with it, both named in plain words rather than gestured at.
    assert "marginal covariance" in why
    assert "beta" in why  # the gradient it is not
    assert "base-policy bank" in why and "zero optimisation pressure" in why


def test_chi_envelope_matches_the_catalogue_record_for_c1() -> None:
    """The two conditions the merged catalogue carries for C1, and no silently added third."""
    env = index_pkg.Chi().envelope
    assert env is not None
    assert env.requires == frozenset(
        {RegimeCondition.LINEAR_RESPONSE, RegimeCondition.GROUP_NONDEGENERATE}
    )


def test_chi_is_covariant_not_invariant_under_a_reward_rescale() -> None:
    """`Cov(f, a*r + b) = a*Cov(f, r)`, so the declared relation has to carry the weight.

    Declaring this one invariant would be the mis-declaration `Relation` exists to catch: the
    generated test would then assert equality and fail on the first draw with `a != 1`.
    """
    chi = index_pkg.Chi()
    assert chi.invariance == "reward.affine"
    assert chi.invariance_relation is not None
    # Read through `resolve_relation` rather than off the attribute. `chi` now declares the mapping
    # form, because it transforms two ways: covariant with weight 1 under `reward.affine` and
    # invariant under `repr.basis`, which it used to record in a comment for want of anywhere to put
    # it.
    affine = resolve_relation(chi, "reward.affine")
    assert affine.status == "covariant"
    assert affine.weight == 1.0
    assert resolve_relation(chi, "repr.basis").status == "invariant"


# ---------------------------------------------------------------------------
# The generated invariance test, one per instrument
# ---------------------------------------------------------------------------


def _payload(seed: int = 0):
    """One payload every probe below reads, in the vocabulary the seven groups share.

    ``scores`` is 2n rewards laid out as the chosen block followed by the rejected block, with
    ``group_ids`` naming the pair, so a probe recovers a per-pair delta as ``s[:n] - s[n:]`` and a
    group action that adds a constant to every score leaves that delta alone. ``activations`` is the
    (n, d) matrix and ``readouts`` the (k, d) directions, which `repr.basis` rotates together.
    """
    from reward_lens.core.invariance import InvariancePayload

    rng = np.random.default_rng(seed)
    n, d, k = 24, 8, 3
    activations = rng.standard_normal((n, d))
    readouts = rng.standard_normal((k, d))
    chosen = activations @ readouts[0] + 0.3 * rng.standard_normal(n)
    rejected = chosen - 0.8 - 0.2 * rng.standard_normal(n)
    return InvariancePayload(
        scores=np.concatenate([chosen, rejected]),
        group_ids=np.concatenate([np.arange(n), np.arange(n)]),
        activations=activations,
        readouts=readouts,
    )


def _delta(payload) -> np.ndarray:
    s = np.asarray(payload.scores, dtype=np.float64)
    half = s.size // 2
    return s[:half] - s[half:]


def _probes() -> dict:
    """The probe each instrument's generated test evaluates, and why it is the right functional.

    `check_invariance` asserts a relation on a scalar, so each entry projects the instrument's own
    mathematics onto the one number the relation is declared about. Where a module exposes a pure
    function that is the science, the probe calls it. Where the battery exposes no pure function
    (the patching instruments compute through the runtime), the probe is the arithmetic core the
    reading is built from, which for all of those is an inner product against the readout.
    """
    from reward_lens.measure.battery.bias import cohens_d
    from reward_lens.measure.battery.circuit import jaccard
    from reward_lens.measure.battery.geometry import cosine_matrix
    from reward_lens.measure.battery.lens import crystallization_layer
    from reward_lens.measure.battery.snr import power_snr
    from reward_lens.measure.indices.chi import susceptibility
    from reward_lens.measure.indices.coherence import max_offdiagonal_coherence
    from reward_lens.measure.indices.contested import contested_direction
    from reward_lens.measure.indices.dark_reward import dark_reward
    from reward_lens.measure.indices.distortion import linear_sensitivity
    from reward_lens.measure.indices.eval_awareness import eval_awareness_probe
    from reward_lens.measure.indices.kui import Property, kui_from_properties
    from reward_lens.measure.indices.legibility import legibility_frontier
    from reward_lens.measure.indices.receipt_reliance import receipt_reliance
    from reward_lens.measure.indices.skepticism import skepticism
    from reward_lens.measure.indices.snr import robustness_snr
    from reward_lens.measure.indices.style_share import style_share
    from reward_lens.measure.indices.tail import tail_estimate
    from reward_lens.measure.indices.teacher_compatibility import teacher_compatibility
    from reward_lens.measure.indices.vce import value_convergence_excess
    from reward_lens.measure.indices.verification_score import verification_score

    def acts_dot_readout(p) -> float:
        """The one operation the lens, the attribution and both patching instruments are made of."""
        return float(np.sum(np.asarray(p.activations) @ np.asarray(p.readouts)[0]))

    def lens(p) -> float:
        layers = np.arange(np.asarray(p.activations).shape[0])
        curve = np.asarray(p.activations) @ np.asarray(p.readouts)[0]
        return float(crystallization_layer(curve, layers))

    def circuit(p) -> float:
        strength = np.abs(np.asarray(p.activations) @ np.asarray(p.readouts).T).mean(axis=0)
        top = set(np.argsort(strength)[::-1][:2].tolist())
        return jaccard(top, {0, 1})

    def cosines(p) -> float:
        m = cosine_matrix(np.asarray(p.readouts))
        return float(m[~np.eye(m.shape[0], dtype=bool)].mean())

    def conflict(p) -> float:
        acts = np.asarray(p.activations)
        thirds = acts.shape[0] // 3
        terms = np.stack([acts[i * thirds : (i + 1) * thirds].mean(axis=0) for i in range(3)])
        m = cosine_matrix(terms)
        return float(m[~np.eye(3, dtype=bool)].mean())

    def four_rewards(p) -> tuple[float, float]:
        """Two reward differences, each of two scores, so an additive shift cancels in both."""
        s = np.asarray(p.scores, dtype=np.float64)
        q = s.size // 4
        return float(s[:q].mean() - s[q : 2 * q].mean()), float(
            s[2 * q : 3 * q].mean() - s[3 * q :].mean()
        )

    def kui(p) -> float:
        readouts = np.asarray(p.readouts)
        props = [
            Property(name=f"p{i}", decodability=0.5 + 0.1 * i, direction=readouts[i])
            for i in range(readouts.shape[0])
        ]
        return float(kui_from_properties(props, readouts[0])["kui"][0])

    def legibility(p) -> float:
        acts = np.asarray(p.activations)
        n = acts.shape[0]
        report = legibility_frontier(acts[:, :4], _delta(p)[:n], [1.0, 2.0, 3.0, 4.0])
        return float(report["fidelity_at_knee"])  # type: ignore[arg-type]

    def evalaware(p) -> float:
        acts = np.asarray(p.activations)
        labels = (np.arange(acts.shape[0]) % 2).astype(int)
        return float(eval_awareness_probe(acts, labels, seed=0)["balanced_accuracy"])

    def contested(p) -> float:
        acts = np.asarray(p.activations)
        dis = np.linspace(0.0, 1.0, acts.shape[0])
        return float(contested_direction(acts, dis)["correlation"])  # type: ignore[arg-type]

    return {
        # -- reward.affine: the group acts on `scores` --------------------
        "BiasBattery": lambda p: float(cohens_d(_delta(p))),
        "PromptSNR": lambda p: float(power_snr(_delta(p))),
        "RobustnessSNR": lambda p: float(
            robustness_snr(np.asarray(p.scores), np.asarray(p.group_ids))["snr"]
        ),
        "TailIndex": lambda p: float(tail_estimate(np.asarray(p.scores))["shape_xi"]),
        "DarkReward": lambda p: float(
            dark_reward(np.asarray(p.scores), np.tile(np.asarray(p.activations)[:, :2], (2, 1)))
        ),
        "Legibility": legibility,
        "Skepticism": lambda p: float(
            skepticism(
                float(np.asarray(p.scores)[: p.scores.size // 2].mean()),
                float(np.asarray(p.scores)[p.scores.size // 2 :].mean()),
            )
        ),
        "ReceiptReliance": lambda p: float(receipt_reliance(*four_rewards(p))),
        "VerificationScore": lambda p: float(verification_score(*reversed(four_rewards(p)))),
        # -- repr.basis: the group acts on `activations` and `readouts` ----
        "LensCrystallization": lens,
        "DirectLinearAttribution": acts_dot_readout,
        "PatchGrid": acts_dot_readout,
        "PathEffect": acts_dot_readout,
        "ConceptDoseResponse": lambda p: float(
            np.dot(
                np.asarray(p.readouts)[1] / np.linalg.norm(np.asarray(p.readouts)[1]),
                np.asarray(p.readouts)[0],
            )
        ),
        "ConflictMatrix": conflict,
        "CircuitJaccard": circuit,
        "FeatureRewardAlignment": lambda p: float(
            np.max(np.asarray(p.readouts) @ np.asarray(p.readouts)[0])
        ),
        "MultiObjectiveGeometry": cosines,
        "Coherence": lambda p: max_offdiagonal_coherence(cosine_matrix(np.asarray(p.readouts))),
        "Contested": contested,
        "Distortion": lambda p: float(
            linear_sensitivity(np.asarray(p.readouts), np.asarray(p.readouts)[0])[1]
        ),
        "EvalAwareness": evalaware,
        "KUI": kui,
        "StyleShare": lambda p: style_share(
            np.asarray(p.activations)[0],
            np.asarray(p.readouts)[:2],
            np.asarray(p.readouts)[0],
        ),
        "TeacherCompatibility": lambda p: teacher_compatibility(
            np.asarray(p.readouts)[0], np.asarray(p.activations)
        ),
        "VCE": lambda p: float(
            value_convergence_excess(
                float(np.abs(np.dot(np.asarray(p.readouts)[0], np.asarray(p.readouts)[1]))),
                float(np.abs(np.dot(np.asarray(p.readouts)[0], np.asarray(p.readouts)[2]))),
            )["vce"]
        ),
        # -- reward.affine, covariant weight 1 -----------------------------
        "Chi": lambda p: float(
            susceptibility(np.tile(np.asarray(p.activations)[:, :1], (2, 1)), np.asarray(p.scores))[
                0
            ]
        ),
        # -- trivial: nothing acts, and the generated test says so ---------
        "CoverageDisparity": lambda p: 0.0,
        "InterpCoverage": lambda p: 0.0,
    }


@pytest.mark.parametrize("inst", all_instruments(), ids=lambda i: i.name)
def test_generated_invariance_check_passes(inst) -> None:
    """One generated property test per instrument, under the group it declares.

    Standing rule 4: no instrument merges without this. An instrument declaring `trivial` passes
    vacuously because nothing acts on its reading, which is an honest pass and is reported as a skip
    inside the report; an instrument declaring nothing never reaches here, because
    `test_declarations_present_even_where_the_quantity_row_is_missing` fails first.
    """
    probes = _probes()
    assert inst.name in probes, f"no invariance probe registered for {inst.name}"
    probe = probes[inst.name]
    payload = _payload()
    groups = parse_group_field(inst.invariance)
    assert groups, f"{inst.name} declares {inst.invariance!r}, which names no group"

    for gid in groups:
        # `relation` is deliberately not passed. Forcing one relation for every group an instrument
        # declares is what the mapping form of `invariance_relation` exists to stop: `chi` is
        # covariant with weight 1 under `reward.affine` and invariant under `repr.basis`, and a
        # single relation cannot say both. Left unset, `check_invariance` calls `resolve_relation`
        # and gets the right one per group, which is the behaviour under test.
        report = check_invariance(
            inst,
            gid,
            payload,
            n=16,
            run=lambda _inst, p: probe(p),
        )
        assert report.passed, report.render()


def test_the_trivial_group_pass_is_a_skip_and_not_a_measurement() -> None:
    """`trivial` passes because nothing acts, and the report says so rather than claiming a check."""
    for inst in all_instruments():
        if inst.invariance != "trivial":
            continue
        report = check_invariance(
            inst,
            "trivial",
            _payload(),
            relation=inst.invariance_relation,
            run=lambda _i, _p: 0.0,
        )
        assert report.passed
        assert report.skipped, f"{inst.name} claims a trivial-group check it did not run"
