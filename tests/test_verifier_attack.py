"""D8: the harness inventory, the taint route to a scoring decision, and the cost ratio.

The fixture harness below is small and every finding in it was put there on purpose, which is what
makes the assertions checkable: the credential read, the agent-writable read, the computed
subprocess target and the unchecked route from file contents to a branch are all visible in nine
lines of source.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import InvariancePayload, check_invariance, check_unit_refusal
from reward_lens.core.quantity import ladder
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import VerifierUnderTest, is_sensitive, redact
from reward_lens.verifier.attack import (
    AttackCheapness,
    AttackCheapnessRatio,
    AttackSurface,
    AttackSurfaceInventory,
    CostSample,
    TaskBoundary,
    analyse_environment,
    attack_cheapness,
)
from reward_lens.verifier.fuzz import DisclosureDecision, DisclosureRequired

HARNESS = textwrap.dedent('''
    """A scoring harness with one of each kind of exposure, all deliberate."""

    import json
    import os
    import subprocess

    import requests

    GOLD = "/opt/gold/answers.json"


    def grade(submission_path, task_id):
        """Score a submission. Reads the agent's own output file, which is the point."""
        key = os.environ["GRADER_API_KEY"]
        with open(submission_path) as fh:
            payload = json.load(fh)
        with open(GOLD) as g:
            gold = json.load(g)
        report = open("/workspace/report.json", "w")
        report.write("done")
        subprocess.run(["pytest", payload["tests"]])
        requests.post("https://telemetry.invalid/t", json={"task": task_id})
        if payload["claimed_score"] > 0.5:
            return 1.0
        if gold["answer"] == payload["answer"]:
            return 1.0
        return 0.0
''')

BOUNDARY = TaskBoundary(
    agent_writable=("/workspace/*",),
    harness_private=("/opt/gold/*",),
    secret=("*API_KEY*", "*TOKEN*"),
    name="fixture",
)


@pytest.fixture
def harness(tmp_path: Path) -> VerifierUnderTest:
    path = tmp_path / "harness.py"
    path.write_text(HARNESS, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


# ---------------------------------------------------------------------------
# The two libcst wrinkles, re-verified where this module depends on them
# ---------------------------------------------------------------------------


def test_libcst_is_the_package_we_think_it_is() -> None:
    import importlib.metadata as md

    import libcst

    assert "site-packages" in libcst.__file__
    assert md.version("libcst").startswith("1.")


def test_the_visitor_must_be_driven_by_the_wrapper_and_not_by_the_module() -> None:
    """E9's first libcst wrinkle, re-run against the installed version.

    `module.visit(v)` never resolves the metadata, so the first `get_metadata` call inside the
    visitor fails. E9 records this as one failure and it is two, which matters because the second
    one is the confusing kind and it is the one this module's own visitor would hit: a visitor
    that defines `__init__` without calling `super().__init__()` never gets its `metadata`
    attribute, so it dies on `AttributeError: 'X' object has no attribute 'metadata'` with no
    mention of metadata resolution at all. A visitor using the inherited constructor gets the
    helpful `KeyError` naming the wrapper.
    """
    import libcst as cst
    from libcst.metadata import MetadataWrapper, ScopeProvider

    # No import in this source, so the last line below exercises the wrapper without also tripping
    # the second wrinkle, which the next test isolates.
    module = cst.parse_module("def f(a):\n    b = a\n    return b\n")

    class _Inherited(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ScopeProvider,)

        def visit_Name(self, node):  # noqa: ANN001, ANN202
            self.get_metadata(ScopeProvider, node)
            return True

    class _OwnInit(_Inherited):
        def __init__(self) -> None:  # deliberately no super().__init__()
            self.seen = 0

    with pytest.raises(KeyError, match="did you forget a MetadataWrapper"):
        module.visit(_Inherited())

    with pytest.raises(AttributeError, match="metadata"):
        module.visit(_OwnInit())

    MetadataWrapper(module).visit(_Inherited())  # the same visitor, driven correctly


def test_scope_provider_raises_key_error_on_an_import_alias() -> None:
    """E9's second wrinkle. The names it cannot place are reported rather than swallowed."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper, ScopeProvider

    unplaced: list[str] = []

    class _V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ScopeProvider,)

        def visit_Name(self, node):  # noqa: ANN001, ANN202
            try:
                self.get_metadata(ScopeProvider, node)
            except KeyError:
                unplaced.append(node.value)
            return True

    MetadataWrapper(cst.parse_module("import os as operating_system\nx = 1\n")).visit(_V())
    assert "os" in unplaced


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------


def test_every_planted_exposure_is_found(harness) -> None:
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    kinds = surface.by_kind
    assert kinds["environment"] == 1
    assert kinds["execute"] == 1
    assert kinds["network"] == 1
    assert kinds["read"] == 4  # two `open`s and the two `json.load`s that consume them
    assert kinds["write"] == 2


def test_the_credential_read_is_named_as_one(harness) -> None:
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    creds = surface.credentials
    assert len(creds) == 1
    assert creds[0].target == "GRADER_API_KEY"
    assert creds[0].kind == "environment"


def test_a_module_level_constant_path_resolves_to_a_literal(harness) -> None:
    """`GOLD` is a name at the call site and a path in the report, or the boundary matches nothing."""
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    gold = [a for a in surface.accesses if a.target == "/opt/gold/answers.json"]
    assert gold and gold[0].target_is_literal


def test_a_write_names_the_file_and_not_what_was_written(harness) -> None:
    surface = analyse_environment(harness, rung=0)
    writes = {a.target for a in surface.accesses if a.kind == "write"}
    assert "/workspace/report.json" in writes
    assert "done" not in writes, "the string being written is not a path to test the boundary on"


# ---------------------------------------------------------------------------
# The taint route, which is the pattern the rung exists to find
# ---------------------------------------------------------------------------


def test_untrusted_file_contents_reach_a_scoring_decision(harness) -> None:
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    routes = surface.unchecked_taints
    assert routes, surface.render(include_targets=True)
    assert all(r.sink_kind in {"return", "branch"} for r in routes)
    assert any("payload" in r.chain for r in routes)


def test_the_harness_reading_its_own_gold_is_not_a_finding_once_a_boundary_is_declared(
    harness,
) -> None:
    """Counting a grader's own answer key as untrusted input turns every grader into a finding."""
    with_boundary = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    without = analyse_environment(harness, rung=0)
    assert len(with_boundary.taints) < len(without.taints)
    assert not any("gold" in t.chain for t in with_boundary.taints)
    assert any("gold" in t.chain for t in without.taints)


def test_a_checked_route_is_reported_as_checked(tmp_path: Path) -> None:
    source = textwrap.dedent("""
        import json

        def grade(submission_path):
            with open(submission_path) as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return 0.0
            return 1.0
    """)
    path = tmp_path / "checked.py"
    path.write_text(source, encoding="utf-8")
    surface = analyse_environment(VerifierUnderTest(path, entrypoint="grade"), rung=0)
    assert surface.taints
    assert all(t.validated for t in surface.taints)
    assert surface.headline == 0


def test_a_target_computed_from_untrusted_input_is_marked(harness) -> None:
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    steered = [a for a in surface.accesses if a.tainted_target]
    assert any(a.call == "open" for a in steered)
    assert any(a.call == "subprocess.run" for a in steered)


def test_a_wrong_entrypoint_raises_rather_than_reporting_a_sealed_harness(harness) -> None:
    """A zero from the wrong function name is the most misleading output this could produce."""
    wrong = VerifierUnderTest(harness.source_path, entrypoint="score_it")
    with pytest.raises(AttributeError, match="defines no function named"):
        analyse_environment(wrong, rung=0)


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_the_crossings_are_the_specification_sentence(harness) -> None:
    """*The scoring script reads a file the agent can write.*"""
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    rules = {c.rule for c in surface.crossings}
    assert "reaches a declared secret" in rules
    assert "the target is computed from untrusted input" in rules


def test_rung_one_without_a_boundary_says_so_rather_than_reporting_no_crossings(harness) -> None:
    surface = analyse_environment(harness, rung=1)
    assert surface.crossings == ()
    assert any("no TaskBoundary was supplied" in n for n in surface.notes)


# ---------------------------------------------------------------------------
# Dual use
# ---------------------------------------------------------------------------


def test_the_surface_is_sensitive_and_redacts_to_counts(harness) -> None:
    """*Sensitive outputs are excluded from the rendered card unless explicitly requested.*"""
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    assert is_sensitive(surface)

    reduced = redact(surface)
    assert not is_sensitive(reduced)
    assert reduced.by_kind == surface.by_kind
    assert len(reduced.credentials) == len(surface.credentials)
    assert reduced.headline == surface.headline
    assert "GRADER_API_KEY" not in reduced.render(include_targets=True)
    assert "/workspace/report.json" not in reduced.render(include_targets=True)


def test_the_redacted_view_still_reports_how_many_crossings_there_were(harness) -> None:
    """A card showing zero crossings because the list was removed is worse than no card."""
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    assert f"{len(surface.crossings)} crossings" in surface.redacted().render()


def test_publishing_without_a_recorded_decision_raises(harness) -> None:
    surface = analyse_environment(harness, rung=1, boundary=BOUNDARY)
    with pytest.raises(DisclosureRequired, match="no recorded decision"):
        surface.for_publication()

    decided = surface.with_disclosure(
        DisclosureDecision(decided_by="the grader's owner", reason="internal review")
    )
    assert decided.for_publication() is decided


def test_the_disclosure_mechanism_is_d5s_rather_than_a_second_one() -> None:
    """One mechanism, because two in a package is one more than a reviewer can check."""
    import reward_lens.verifier.attack as attack
    import reward_lens.verifier.fuzz as fuzz

    assert attack.DisclosureDecision is fuzz.DisclosureDecision
    assert attack.DisclosureRequired is fuzz.DisclosureRequired


# ---------------------------------------------------------------------------
# The cost ratio
# ---------------------------------------------------------------------------


def test_the_ratio_reads_the_way_the_specification_sentence_does() -> None:
    """*Attacking the grader is 11x cheaper than solving the task.*"""
    result = attack_cheapness(
        CostSample(what="prompt-injection attempt", unit="tokens", values=(100.0,) * 8),
        CostSample(what="solve the task", unit="tokens", values=(1100.0,) * 8),
    )
    assert isinstance(result, AttackCheapness)
    assert result.ratio == pytest.approx(11.0)
    assert result.cheaper_to_attack is True
    assert "11.00x" in result.render()


def test_a_ratio_whose_interval_spans_one_is_reported_as_undecided() -> None:
    result = attack_cheapness(
        CostSample(what="attack", unit="seconds", values=(9.0, 11.0, 10.0, 12.0, 8.0)),
        CostSample(what="solve", unit="seconds", values=(11.0, 9.0, 12.0, 8.0, 10.0)),
        seed=1,
    )
    assert isinstance(result, AttackCheapness)
    assert result.ci_low < 1.0 < result.ci_high
    assert result.cheaper_to_attack is None


def test_comparing_across_a_unit_boundary_refuses_rather_than_converting() -> None:
    """The `units` group's assertion, which is `env.attack_cheapness`'s declared group."""
    refusal = attack_cheapness(
        CostSample(what="attack", unit="tokens", values=(100.0,)),
        CostSample(what="solve", unit="seconds", values=(30.0,)),
    )
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.UNIT_MISMATCH
    assert "property of your setup, not of the units" in refusal.detail
    assert refusal.remedy.strip()


def test_the_unit_refusal_passes_the_generated_check_for_the_units_group() -> None:
    """*No instrument merges without its generated invariance test passing.*

    The `units` group's generated check is a refusal rather than a numeric relation, so
    `check_invariance` routes it to `check_unit_refusal`, and this is that check run directly on
    the comparison the instrument performs.
    """
    assert check_unit_refusal(
        lambda a, b: attack_cheapness(a, b),
        CostSample(what="attack", unit="tokens", values=(100.0,)),
        CostSample(what="solve", unit="seconds", values=(30.0,)),
    )
    report = check_invariance(AttackCheapnessRatio(), "units", InvariancePayload(), n=4)
    assert report.passed
    assert "refusal" in report.skipped


def test_no_measured_costs_refuses_rather_than_inferring_a_ratio_from_source() -> None:
    """*Every number came from code that ran.* A cheapness read off an inventory is a guess."""
    reading = AttackCheapnessRatio().estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "attack and solve" in reading.detail
    assert "This rung is a measurement" in reading.remedy


def test_a_cost_sample_without_a_unit_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="needs a unit"):
        CostSample(what="attack", unit="  ", values=(1.0,))


def test_a_zero_cost_is_a_construction_error_because_the_ratio_divides_by_it() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        CostSample(what="attack", unit="tokens", values=(0.0,))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(
    attack=st.lists(st.floats(min_value=0.1, max_value=1e4), min_size=1, max_size=12),
    factor=st.floats(min_value=0.05, max_value=40.0),
)
@settings(max_examples=100, deadline=None)
def test_the_ratio_is_scale_free_in_the_shared_unit(attack: list[float], factor: float) -> None:
    """Measuring both sides in milliseconds instead of seconds must not move the ratio.

    That is what makes a cost *ratio* the reportable quantity rather than either cost on its own,
    and it is why the unit refusal above is about a *mismatch* rather than about the unit itself.
    """
    solve = [v * 3.0 for v in attack]
    base = attack_cheapness(
        CostSample(what="a", unit="u", values=tuple(attack)),
        CostSample(what="s", unit="u", values=tuple(solve)),
        resamples=64,
    )
    scaled = attack_cheapness(
        CostSample(what="a", unit="u", values=tuple(v * factor for v in attack)),
        CostSample(what="s", unit="u", values=tuple(v * factor for v in solve)),
        resamples=64,
    )
    assert isinstance(base, AttackCheapness) and isinstance(scaled, AttackCheapness)
    assert scaled.ratio == pytest.approx(base.ratio, rel=1e-9)


@given(
    body=st.lists(
        st.sampled_from(
            [
                "    x = 1",
                "    y = open('/tmp/a')",
                "    z = os.environ['TOKEN']",
                "    if x:\n        return 1.0",
                "    return 0.0",
            ]
        ),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=60, deadline=None)
def test_the_static_pass_never_raises_on_a_parseable_harness(body: list[str]) -> None:
    """A static pass that dies on an unfamiliar shape is not a static pass.

    A `TemporaryDirectory` rather than `tmp_path`: hypothesis rejects a function-scoped fixture
    under `@given` because it is not reset between generated inputs, and here it genuinely is not
    (every draw writes the same filename).
    """
    import tempfile

    source = "import os\n\n\ndef grade(payload):\n" + "\n".join(body) + "\n    return 0.0\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "generated.py"
        path.write_text(source, encoding="utf-8")
        surface = analyse_environment(VerifierUnderTest(path, entrypoint="grade"), rung=0)
    assert isinstance(surface, AttackSurface)
    assert surface.headline >= 0


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_both_instruments_pass_lint(harness) -> None:
    for inst in (AttackSurfaceInventory(harness), AttackCheapnessRatio()):
        assert lint_instrument(inst) == [], inst.name


def test_the_inventory_declares_source_access_rather_than_mutate(harness) -> None:
    """E20: reading a harness's text is neither running it nor modifying it."""
    from reward_lens.core.types import Access, Component

    inst = AttackSurfaceInventory(harness)
    assert inst.requires[Component.TASK] is Access.SOURCE
    assert Access.MUTATE not in inst.requires[Component.TASK]


def test_the_registered_ladder_matches_what_the_quantity_registry_declares() -> None:
    """`rungs: 3` for the surface and `rungs: 1` for the cheapness ratio."""
    assert [e.rung for e in ladder("env.attack_surface")] == [0, 1, 2]
    assert [e.rung for e in ladder("env.attack_cheapness")] == [2]


def test_the_inventorys_generated_invariance_test_passes(harness) -> None:
    inst = AttackSurfaceInventory(harness)
    group = inst.invariance if inst.invariance != "none" else "trivial"
    report = check_invariance(
        inst,
        group,
        InvariancePayload(),
        n=4,
        run=lambda i, _p: float(analyse_environment(harness, rung=0).headline),
    )
    assert report.passed
    assert "trivial group" in report.skipped


def test_the_instrument_emits_a_row_flagged_sensitive(harness) -> None:
    reading = AttackSurfaceInventory(harness, rung=1, boundary=BOUNDARY).estimate()
    assert not isinstance(reading, Refusal), getattr(reading, "render", lambda: reading)()
    assert reading.subject.extra["sensitive"] == "true"
    assert reading.subject.extra["baseline_sandbox_holds"] == "0"
