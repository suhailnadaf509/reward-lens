"""Acceptance: the docs build reads the registry, and lint rule two is enforceable.

The instrument contract states two lint rules. The first, an `Instrument` whose quantity is not
registered fails at import, has been enforced since Phase 0. The second, **a `Quantity` with no estimator
fails the docs build with a message naming it as an open research target rather than a bug**, had
nothing to enforce it: no part of the documentation build read the registry, so the rule was a
sentence in a specification.

`docs/gen_catalogue.py` is what closes it, and these tests are the definition of done for that.
They import the generator directly rather than shelling out to mkdocs, so they run in the ordinary
test environment with no documentation dependencies installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"


def _load_generator():
    """Import `docs/gen_catalogue.py`, which is not on the package path by design."""
    path = DOCS_DIR / "gen_catalogue.py"
    if not path.exists():  # pragma: no cover - only in a source-less install
        pytest.skip("docs/ is not present in this checkout")
    spec = importlib.util.spec_from_file_location("_rl_gen_catalogue", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


@pytest.fixture(scope="module")
def view(gen):
    return gen.read_registry()


# ---------------------------------------------------------------------------
# Lint rule two
# ---------------------------------------------------------------------------


def test_ledger_exists_and_is_not_empty(gen):
    """The ratchet has a file. Without one the rule silently exempts everything."""
    assert gen.LEDGER_PATH.exists(), "docs/open-quantities.txt is the ratchet; it must exist"
    assert gen.load_ledger(), "an empty ledger exempts nothing and also records nothing"


def test_every_open_quantity_is_recorded(gen, view):
    """The live rule: no quantity may be open without being named in the ledger."""
    notes = gen.check_ledger(view, gen.load_ledger())
    assert any("lint rule two" in n for n in notes)


def test_an_unrecorded_open_quantity_fails_the_build(gen, view):
    """The rule has teeth. Drop one entry and the build must stop, naming the quantity.

    This is the test that distinguishes an enforced rule from a rendered page. The message has to
    name the quantity and call it an open research target, because that wording is the rule: the
    point is that an unbuilt estimator reads as roadmap rather than as a defect.
    """
    if not view.open_quantities:  # pragma: no cover - the day every quantity is built
        pytest.skip("no open quantities left, which would be excellent news")
    victim = sorted(view.open_quantities)[0]
    thinned = gen.load_ledger() - {victim}
    with pytest.raises(gen.LintFailure) as exc:
        gen.check_ledger(view, thinned)
    message = str(exc.value)
    assert victim in message, "the failure must name the quantity, not just count it"
    assert "open research target" in message
    assert "--write-ledger" in message, "a lint failure with no remedy is a dead end"


def test_a_closed_ledger_entry_does_not_fail_the_build(gen, view):
    """The ratchet may shrink. An estimator landing must never turn the docs red."""
    notes = gen.check_ledger(view, gen.load_ledger() | {"quantity.that.was.built"})
    assert any("can be deleted" in n for n in notes)


# ---------------------------------------------------------------------------
# The render comes from the registry, not from a transcription
# ---------------------------------------------------------------------------


def test_catalogue_loader_and_catalogue_file_agree(gen):
    """`read_registry` cross-checks the two and raises if they diverge. Assert it ran clean."""
    view = gen.read_registry()
    assert view.n_instruments > 0
    assert set(view.rows) == {i.id for i in view.instruments}


def test_pages_render_and_carry_the_catalogue_fields(gen, view):
    """Every instrument's page carries the fields the catalogue declares for it."""
    pages = gen.build_pages(view, gen.load_ledger())
    assert "catalogue/index.md" in pages
    assert "refusals.md" in pages

    body = "\n".join(pages[uri] for uri in pages if uri.startswith("catalogue/series-"))
    for inst in view.instruments:
        assert f"### {inst.id}. {inst.name}" in body, f"{inst.id} has no section"
    for label in ("Quantity", "Access", "Substrates", "Phases", "Envelope", "Invariance group"):
        assert f"| {label} |" in body, f"the fact table dropped {label}"
    assert "**The ladder.**" in body
    assert "**Kill condition.**" in body
    assert "**What a reading would say.**" in body
    assert "**Baselines.**" in body


def test_no_page_pins_a_count_that_the_registry_owns(gen, view):
    """Counts are read at build time, so the pages track a catalogue that is still moving."""
    pages = gen.build_pages(view, gen.load_ledger())
    index = pages["catalogue/index.md"]
    assert f"{view.n_instruments} instruments" in index
    assert f"{len(view.quantities)} quantities" in index


def test_open_page_names_every_open_quantity(gen, view):
    page = gen.build_pages(view, gen.load_ledger())["catalogue/open.md"]
    for qid in view.open_quantities:
        assert f"`{qid}`" in page, f"{qid} is open and is not named on the roadmap page"


def test_the_polymorphic_open_field_is_normalised(gen):
    """E14: six rows store `quantities` as the bare string OPEN.

    Iterating that string yields four single-character ids. Rendering them would put four phantom
    quantities on a user-facing page, which is the exact shape of the bug E14 records.
    """
    assert gen._as_list("OPEN") == []
    assert gen._as_list(None) == []
    assert gen._as_list(["a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# The refusal reference
# ---------------------------------------------------------------------------


def test_refusal_page_covers_every_reason_exactly_once(gen):
    from reward_lens.core.reading import REASON_MEANING, RefusalReason

    page = gen._refusals_page()
    for reason in RefusalReason:
        assert f"### `{reason.name}`" in page, f"{reason.name} has no section"
        assert page.count(f"### `{reason.name}`") == 1
        assert " ".join(REASON_MEANING[reason].split()) in page
        assert reason.name in gen.WHAT_TO_DO


def test_refusal_page_says_what_to_do_for_every_reason(gen):
    """A reason with a meaning and no instruction is a dead end with a name."""
    from reward_lens.core.reading import RefusalReason

    assert set(gen.WHAT_TO_DO) == {r.name for r in RefusalReason}
    for name, text in gen.WHAT_TO_DO.items():
        assert len(text) > 120, f"{name}: an instruction this short is not an instruction"


def test_a_seventeenth_reason_would_stop_the_docs_build(gen, monkeypatch):
    """The count is not asserted; the coverage is.

    A seventeenth `RefusalReason` has been requested more than once and is a maintainer decision.
    If one lands, this is what happens: the docs build stops until somebody writes down what a
    user holding it should do. That is the behaviour worth having, rather than a page that
    silently documents sixteen of seventeen.
    """
    thinned = dict(gen.WHAT_TO_DO)
    thinned.pop("VOID")
    monkeypatch.setattr(gen, "WHAT_TO_DO", thinned)
    with pytest.raises(gen.LintFailure) as exc:
        gen._refusals_page()
    assert "VOID" in str(exc.value)


def test_reason_meaning_covers_the_enum(gen):
    """`REASON_MEANING` is the source the page renders from, so a gap in it is a gap on the page."""
    from reward_lens.core.reading import REASON_MEANING, RefusalReason

    assert set(REASON_MEANING) == set(RefusalReason)
    assert len(RefusalReason) == 17, "seventeen reasons"


# ---------------------------------------------------------------------------
# The build environment
# ---------------------------------------------------------------------------


def test_a_broken_module_is_reported_as_broken_not_as_an_open_target(gen):
    """A missing optional extra is expected; a NameError in the tree is not.

    Folding the second into the first would make a half-written module look like an unbuilt
    instrument, and send whoever reads the roadmap page looking in the wrong place.
    """
    from reward_lens.core.extras import ExtraRequiredError

    assert gen._is_optional_extra(ExtraRequiredError("needs white-box"))
    assert gen._is_optional_extra(ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert not gen._is_optional_extra(NameError("name 'dataclass' is not defined"))
    assert not gen._is_optional_extra(
        ModuleNotFoundError("No module named 'reward_lens.nope'", name="reward_lens.nope")
    )


def test_generate_is_the_one_entry_point_the_docs_build_uses(gen):
    """`generate` runs the lint and the render together, which is what `docs/hooks.py` calls."""
    pages, notes = gen.generate(check=True)
    assert len(pages) >= 18
    assert any("lint rule two" in n for n in notes)
