"""The Atlas: a population registry and a leaderboard view over the store (M7).

The Atlas is the population-scale artifact. Atlas-v0, built here, is the honest floor the design
asks for (RK6): a registry of the standard ten-model population and a leaderboard that is a *view*
over stored Evidence, plus the budgeted-sweep orchestration that the real population sweep will use.
It computes no fresh numbers (I5); the leaderboard reads the latest Evidence of each observable per
model out of the store and tabulates it, and an uncalibrated cell is flagged exactly as a card flags
an uncalibrated index (gate 1 at the render layer). The population-science observables the design
names for the Atlas (herding, attack-transfer risk, universality/VCE, monoculture) are studies that
land here later, one at a time and each budgeted; none is built now, per the RK6 mitigation.

Two honesty points are load-bearing and stated in code rather than assumed:

- The registry is torch-free and hashes no weights, so every standard entry's fingerprint is
  *declared* (derived from the model's Hugging Face repo id) and its lineage carries
  ``provenance_tier = "card-claimed"``. A card-claimed fingerprint is a placeholder that the
  GPU-gated sweep replaces with the verified weight hash from ``runtime.fingerprint`` once real
  weights are loaded. Nothing here pretends a declared id is a weight-verified one.
- ``sweep`` orchestrates and budgets the real population sweep but does not run it. Loading and
  scoring ten reward models is GPU work; the method plans the sweep, prices it against a budget
  from Evidence the store already holds (R13), and refuses to fabricate a run. Executing it raises
  a clearly marked error rather than returning invented numbers.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from reward_lens.core.provenance import Cost
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.core.types import ModelFP, content_hash

ProvenanceTier = Literal["weights-verified", "card-claimed"]


# ---------------------------------------------------------------------------
# Lineage and entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelLineage:
    """A reward model's declared lineage.

    Mirrors the ``SignalMeta.lineage`` fields the runtime records at load, held here as a typed
    object so the Atlas can slice the population by base model, data provenance, and release date
    without loading weights. ``provenance_tier`` is the truth-in-advertising field: ``card-claimed``
    means the fact was read off a model card and no weights were hashed to confirm it, and
    ``weights-verified`` means a fingerprint over the actual safetensors backs it. Kinship, the
    population axes, and the monoculture index are all computed off these fields later (RK9), which
    is why they are collected from day one even though Atlas-v0 does not yet consume them.
    """

    base_model: str
    training_data: str
    release_date: str = ""
    provenance_tier: ProvenanceTier = "card-claimed"

    def to_meta_dict(self) -> dict[str, Any]:
        """Render to the ``SignalMeta.lineage`` dict shape the runtime uses."""
        return {
            "base_model": self.base_model,
            "training_data": self.training_data,
            "release_date": self.release_date,
            "provenance_tier": self.provenance_tier,
        }

    @classmethod
    def from_meta_dict(cls, meta: dict[str, Any]) -> "ModelLineage":
        """Build from a ``SignalMeta.lineage`` dict, tolerating missing optional fields."""
        return cls(
            base_model=meta.get("base_model", ""),
            training_data=meta.get("training_data", ""),
            release_date=meta.get("release_date", ""),
            provenance_tier=meta.get("provenance_tier", "card-claimed"),
        )


def declared_fingerprint(repo_id: str) -> ModelFP:
    """Derive a declared ``ModelFP`` from a Hugging Face repo id.

    Atlas-v0 is torch-free and hashes no weights, so a registry entry needs a stable id before any
    model is loaded. This derives one deterministically from the repo id and tags it ``mfp:`` so it
    slots into a ``SubjectRef`` uniformly. It is a *declared* id: the paired entry always carries
    ``provenance_tier = "card-claimed"``, and the GPU-gated sweep overwrites it with the real
    weight-hash fingerprint when it loads the model. Two registrations of the same repo id agree.
    """
    return ModelFP(content_hash({"declared_repo": repo_id}, "mfp"))


@dataclass(frozen=True)
class AtlasEntry:
    """One reward model in the population.

    An entry is identity plus lineage: the ``fingerprint`` that keys its Evidence in the store, a
    human ``name``, the ``repo_id`` the declared fingerprint derives from, the model ``paradigm``
    (which population slices care about), and the ``lineage``. ``weights_verified`` is a read of the
    lineage tier, surfaced as a property so a leaderboard can mark which fingerprints are backed by a
    real weight hash and which are card-claimed placeholders.
    """

    fingerprint: ModelFP
    name: str
    repo_id: str
    lineage: ModelLineage
    paradigm: str = "classifier"

    @property
    def weights_verified(self) -> bool:
        return self.lineage.provenance_tier == "weights-verified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "name": self.name,
            "repo_id": self.repo_id,
            "paradigm": self.paradigm,
            "weights_verified": self.weights_verified,
            "lineage": self.lineage.to_meta_dict(),
        }


# The standard ten-model population. Lineage here is *declared* from public
# model cards; every entry is card-claimed because Atlas-v0 hashes no weights. The four
# campaign RMs plus Skywork-v0.2 anchor the existing battery; GRM/INF-ORM/URM/QRM/Nemotron are
# placeholders carried with their known lineage so the population axes exist before any sweep runs.
_STANDARD_POPULATION: tuple[dict[str, Any], ...] = (
    {
        "name": "Skywork-Reward-Llama-3.1-8B-v0.1",
        "repo_id": "Skywork/Skywork-Reward-Llama-3.1-8B",
        "paradigm": "classifier",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "training_data": "Skywork Reward Data Collection (80K preference pairs), v0.1",
        "release_date": "2024-09",
    },
    {
        "name": "Skywork-Reward-Llama-3.1-8B-v0.2",
        "repo_id": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "paradigm": "classifier",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "training_data": "Skywork Reward Preference 80K, v0.2 (decontaminated)",
        "release_date": "2024-10",
    },
    {
        "name": "ArmoRM-Llama3-8B",
        "repo_id": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        "paradigm": "multi-objective",
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "training_data": "19-objective mixture (UltraFeedback, HelpSteer, BeaverTails, and others)",
        "release_date": "2024-06",
    },
    {
        "name": "Skywork-Reward-Gemma-2-27B",
        "repo_id": "Skywork/Skywork-Reward-Gemma-2-27B",
        "paradigm": "classifier",
        "base_model": "google/gemma-2-27b-it",
        "training_data": "Skywork Reward Data Collection (80K preference pairs)",
        "release_date": "2024-09",
    },
    {
        "name": "Tulu-3-8B-RM",
        "repo_id": "allenai/Llama-3.1-Tulu-3-8B-RM",
        "paradigm": "classifier",
        "base_model": "meta-llama/Llama-3.1-8B",
        "training_data": "Tulu 3 preference mixture (on-policy and off-policy)",
        "release_date": "2024-11",
    },
    {
        "name": "GRM-Llama3-8B",
        "repo_id": "Ray2333/GRM-Llama3-8B-rewardmodel-ft",
        "paradigm": "classifier",
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "training_data": "reward mixture with hidden-state regularization (generalizable RM)",
        "release_date": "2024-08",
    },
    {
        "name": "INF-ORM-Llama3.1-70B",
        "repo_id": "infly/INF-ORM-Llama3.1-70B",
        "paradigm": "classifier",
        "base_model": "meta-llama/Llama-3.1-70B-Instruct",
        "training_data": "INF outcome-reward preference collection",
        "release_date": "2024-12",
    },
    {
        "name": "URM-LLaMa-3.1-8B",
        "repo_id": "LxzGordon/URM-LLaMa-3.1-8B",
        "paradigm": "uncertainty-aware",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "training_data": "uncertainty-aware reward mixture (attribute-regression head)",
        "release_date": "2024-10",
    },
    {
        "name": "QRM-Llama3.1-8B",
        "repo_id": "nicolinho/QRM-Llama3.1-8B",
        "paradigm": "distributional",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "training_data": "quantile reward mixture (distributional head over reward)",
        "release_date": "2024-10",
    },
    {
        "name": "Llama-3.1-Nemotron-70B-Reward",
        "repo_id": "nvidia/Llama-3.1-Nemotron-70B-Reward",
        "paradigm": "classifier",
        "base_model": "meta-llama/Llama-3.1-70B-Instruct",
        "training_data": "HelpSteer2 preference data",
        "release_date": "2024-10",
    },
)


# ---------------------------------------------------------------------------
# Leaderboard: the view over stored Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderboardCell:
    """One (model, observable) cell of the leaderboard, read from the store (I5).

    Carries the stored value's short representation, the Evidence's trust and calibration, and the
    metered GPU-seconds that produced it (R13). ``validated`` mirrors the card: a cell is validated
    only if its Evidence carries a calibration reference, and an unvalidated cell renders distinctly.
    """

    model: str
    observable: str
    value_repr: str
    trust: str
    calibrated: bool
    evidence_id: str
    gpu_seconds: float

    @property
    def validated(self) -> bool:
        return self.calibrated

    @property
    def flag(self) -> str:
        """The calibration flag string used across the table renders."""
        return "calibrated" if self.calibrated else "uncalibrated"


@dataclass
class Leaderboard:
    """A population comparison table assembled from stored Evidence.

    A view, never a computation: every cell is the latest Evidence of one observable about one model.
    ``cells`` is keyed by ``(model_name, observable)``; a missing key means the store held no Evidence
    for that pair, which the renders show as an empty, explicitly-absent cell rather than a zero.
    """

    models: list[str]
    observables: list[str]
    cells: dict[tuple[str, str], LeaderboardCell] = field(default_factory=dict)
    total_gpu_seconds: float = 0.0

    def cell(self, model: str, observable: str) -> LeaderboardCell | None:
        return self.cells.get((model, observable))

    @property
    def present(self) -> list[LeaderboardCell]:
        """Every cell the store actually had Evidence for."""
        return list(self.cells.values())

    @property
    def flagged(self) -> list[LeaderboardCell]:
        """The uncalibrated cells, flagged exactly as a card flags an unvalidated index (gate 1)."""
        return [c for c in self.cells.values() if not c.validated]

    @property
    def table(self) -> Any:
        """A pandas DataFrame of the stored values, models as rows and observables as columns.

        Absent (model, observable) pairs render as an empty string so the table shows a real gap
        rather than an invented number. Use ``calibration_table`` for the parallel flag grid.
        """
        import pandas as pd

        data = {
            obs: [
                (self.cells[(m, obs)].value_repr if (m, obs) in self.cells else "")
                for m in self.models
            ]
            for obs in self.observables
        }
        return pd.DataFrame(data, index=self.models)

    @property
    def calibration_table(self) -> Any:
        """A pandas DataFrame of per-cell calibration flags parallel to ``table``.

        Each cell is ``"calibrated"``, ``"uncalibrated"``, or ``""`` where the store held nothing.
        This is the grid a reader consults to see which numbers stand behind a scorecard.
        """
        import pandas as pd

        data = {
            obs: [
                (self.cells[(m, obs)].flag if (m, obs) in self.cells else "") for m in self.models
            ]
            for obs in self.observables
        }
        return pd.DataFrame(data, index=self.models)

    def to_json(self) -> str:
        payload = {
            "models": self.models,
            "observables": self.observables,
            "total_gpu_seconds": self.total_gpu_seconds,
            "cells": [
                {
                    "model": c.model,
                    "observable": c.observable,
                    "value": c.value_repr,
                    "trust": c.trust,
                    "calibrated": c.calibrated,
                    "flag": c.flag,
                    "evidence_id": c.evidence_id,
                    "gpu_seconds": c.gpu_seconds,
                }
                for c in self.cells.values()
            ],
            "uncalibrated_cells": [
                {"model": c.model, "observable": c.observable} for c in self.flagged
            ],
        }
        return json.dumps(payload, indent=2)

    def to_html(self) -> str:
        return _render_leaderboard_html(self)


def _value_repr(value: Any) -> str:
    """Short human representation of a stored value, matching the card's convention."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_value_repr(v)}" for k, v in list(value.items())[:3]) + "}"
    return type(value).__name__


_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; }
td.absent { color: #999; font-style: italic; }
td.uncalibrated { background: #fff4f4; color: #7a2b2b; }
td.uncalibrated::after { content: " (unvalidated)"; font-size: 0.7em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
.tier-card-claimed { color: #7a2b2b; }
.tier-weights-verified { color: #2b6a2b; }
"""


def _render_leaderboard_html(lb: Leaderboard) -> str:
    head = (
        "<tr><th>Model</th>"
        + "".join(f"<th>{html.escape(o)}</th>" for o in lb.observables)
        + "</tr>"
    )
    rows = []
    for m in lb.models:
        tds = [f"<td>{html.escape(m)}</td>"]
        for obs in lb.observables:
            c = lb.cells.get((m, obs))
            if c is None:
                tds.append('<td class="absent">no evidence</td>')
            elif c.validated:
                tds.append(f"<td>{html.escape(c.value_repr)}</td>")
            else:
                tds.append(f'<td class="uncalibrated">{html.escape(c.value_repr)}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    n_flagged = len(lb.flagged)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Atlas leaderboard</title>
<style>{_CSS}</style></head>
<body>
<h1>Atlas leaderboard</h1>
<p>{len(lb.models)} models, {len(lb.observables)} observables,
{len(lb.present)} measured cells, {n_flagged} uncalibrated.
Total metered GPU-seconds: {lb.total_gpu_seconds:.2f}.</p>
<p>Cells with no stored Evidence read "no evidence"; uncalibrated cells are shaded and marked
unvalidated. Every value is a stored measurement, not a fresh computation.</p>
<table>
<thead>{head}</thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
</body></html>"""


# ---------------------------------------------------------------------------
# Budgeted sweep orchestration (GPU-gated)
# ---------------------------------------------------------------------------


class SweepGatedError(RuntimeError):
    """Raised when the real population sweep is asked to execute (GPU-gated).

    The sweep loads and scores real reward models, which is GPU work the design keeps off this
    torch-free artifacts layer. ``sweep`` builds and prices the plan; asking it to run raises this
    rather than fabricating scores.
    """


@dataclass(frozen=True)
class SweepCell:
    """One planned unit of sweep work: run ``observable`` on ``fingerprint``."""

    fingerprint: str
    observable: str
    estimated: Cost
    estimate_source: str  # "prior-evidence" or "unestimated"


@dataclass
class SweepPlan:
    """A budgeted, non-executed plan for the population sweep (R13).

    ``cells`` is the cartesian product of the requested signals and battery observables. Each cell's
    ``estimated`` cost is read from the store when a prior Evidence for that pair exists (a view over
    metered cost, I5/R13) and is otherwise left unestimated. ``within_budget`` compares the estimated
    GPU-seconds against the budget; the runner refuses to exceed a budget without an explicit
    override, and executing the plan at all is GPU-gated.
    """

    cells: list[SweepCell]
    budget: Cost
    executed: bool = False

    @property
    def estimated_total(self) -> Cost:
        total = Cost()
        for c in self.cells:
            total = total + c.estimated
        return total

    @property
    def n_estimated(self) -> int:
        return sum(1 for c in self.cells if c.estimate_source == "prior-evidence")

    @property
    def within_budget(self) -> bool:
        """True if the estimated GPU-seconds fit the budget. Unestimated cells count as zero.

        An unestimated plan (no priors in the store) is trivially within budget by GPU-seconds; the
        honest reading is that its cost is unknown until the battery has run once and metered itself,
        which is exactly what ``n_estimated`` surfaces.
        """
        if self.budget.gpu_seconds <= 0:
            return True
        return self.estimated_total.gpu_seconds <= self.budget.gpu_seconds

    def summary(self) -> str:
        est = self.estimated_total
        return (
            f"{len(self.cells)} sweep cells ({self.n_estimated} priced from stored cost), "
            f"estimated {est.gpu_seconds:.1f} GPU-seconds against a budget of "
            f"{self.budget.gpu_seconds:.1f}; within budget: {self.within_budget}. "
            "GPU-gated: run the real sweep on hardware, this layer only plans and prices it."
        )


# ---------------------------------------------------------------------------
# The Atlas
# ---------------------------------------------------------------------------


class Atlas:
    """The population registry and its leaderboard view (Atlas-v0).

    Construct with ``Atlas.standard()`` for the ten-model population, or empty and register entries.
    The registry is keyed by fingerprint; a second registration of the same fingerprint replaces the
    entry (the sweep uses this to upgrade a card-claimed entry to a weights-verified one).
    """

    def __init__(self, entries: Iterable[AtlasEntry] = ()) -> None:
        self._entries: dict[str, AtlasEntry] = {}
        for e in entries:
            self.register(e)

    @classmethod
    def standard(cls) -> "Atlas":
        """The standard ten-model population with declared, card-claimed lineage."""
        entries = []
        for spec in _STANDARD_POPULATION:
            lineage = ModelLineage(
                base_model=spec["base_model"],
                training_data=spec["training_data"],
                release_date=spec["release_date"],
                provenance_tier="card-claimed",
            )
            entries.append(
                AtlasEntry(
                    fingerprint=declared_fingerprint(spec["repo_id"]),
                    name=spec["name"],
                    repo_id=spec["repo_id"],
                    lineage=lineage,
                    paradigm=spec["paradigm"],
                )
            )
        return cls(entries)

    # -- registry ------------------------------------------------------------

    def register(self, entry: AtlasEntry) -> None:
        self._entries[entry.fingerprint] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AtlasEntry]:
        return iter(self._entries.values())

    def __contains__(self, fingerprint: str) -> bool:
        return fingerprint in self._entries

    @property
    def entries(self) -> list[AtlasEntry]:
        return list(self._entries.values())

    def by_fingerprint(self, fingerprint: str) -> AtlasEntry | None:
        return self._entries.get(fingerprint)

    def by_name(self, name: str) -> AtlasEntry | None:
        for e in self._entries.values():
            if e.name == name:
                return e
        return None

    # -- leaderboard ---------------------------------------------------------

    def leaderboard(
        self,
        store: EvidenceStore | None = None,
        observables: Iterable[str] | None = None,
    ) -> Leaderboard:
        """Assemble the population leaderboard as a view over the store.

        For every registered model and every requested observable, query the store for the most
        recent Evidence about that (observable, model) pair and record its value, trust, calibration,
        and metered cost. If ``observables`` is not given, the columns are the distinct observable
        names the store holds for this population, so a bare call tabulates whatever has been measured.
        Nothing is computed: an absent pair is an absent cell, and an uncalibrated cell is flagged.
        """
        store = store if store is not None else default_store()
        models = [e.name for e in self._entries.values()]
        obs_list = (
            list(observables) if observables is not None else self._observables_in_store(store)
        )

        cells: dict[tuple[str, str], LeaderboardCell] = {}
        total_gpu = 0.0
        for entry in self._entries.values():
            for obs in obs_list:
                found = store.find(observable=obs, signal=entry.fingerprint, latest=True)
                if not found:
                    continue
                ev = found[-1]
                gpu = ev.provenance.cost.gpu_seconds
                total_gpu += gpu
                cells[(entry.name, obs)] = LeaderboardCell(
                    model=entry.name,
                    observable=obs,
                    value_repr=_value_repr(ev.value),
                    trust=ev.trust.name,
                    calibrated=ev.is_calibrated,
                    evidence_id=ev.id,
                    gpu_seconds=gpu,
                )
        return Leaderboard(
            models=models,
            observables=obs_list,
            cells=cells,
            total_gpu_seconds=total_gpu,
        )

    def _observables_in_store(self, store: EvidenceStore) -> list[str]:
        """Distinct observable names the store holds for any model in this population."""
        fps = set(self._entries)
        found: set[str] = set()
        for ev in store.find():
            if fps.intersection(ev.subject.signals):
                found.add(ev.observable)
        return sorted(found)

    def export_leaderboard(
        self,
        store: EvidenceStore | None = None,
        observables: Iterable[str] | None = None,
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Export the leaderboard to JSON and HTML.

        Returns a dict with the rendered ``json`` and ``html`` strings and the ``Leaderboard`` itself.
        When ``out_dir`` is given, writes ``leaderboard.json`` and ``leaderboard.html`` under it and
        adds their paths to the returned dict. The strings are the single source; the files are a
        convenience for the site export, which is the intended publishing vehicle.
        """
        lb = self.leaderboard(store=store, observables=observables)
        json_str = lb.to_json()
        html_str = lb.to_html()
        result: dict[str, Any] = {"leaderboard": lb, "json": json_str, "html": html_str}
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            json_path = out / "leaderboard.json"
            html_path = out / "leaderboard.html"
            json_path.write_text(json_str, encoding="utf-8")
            html_path.write_text(html_str, encoding="utf-8")
            result["json_path"] = json_path
            result["html_path"] = html_path
        return result

    # -- sweep (GPU-gated) ---------------------------------------------------

    def sweep(
        self,
        signals: Iterable[str],
        battery: Iterable[str],
        budget: Cost,
        store: EvidenceStore | None = None,
        execute: bool = False,
    ) -> SweepPlan:
        """Plan and price the population sweep; do not run it (R13, GPU-gated).

        Builds the cartesian product of ``signals`` (model fingerprints) and ``battery`` (observable
        names) into a ``SweepPlan``. Each cell's estimated cost is read from the store when a prior
        Evidence for that pair carries metered GPU-seconds, so the estimate is itself a view over
        recorded cost rather than a guess. ``execute=True`` raises ``SweepGatedError``: loading and
        scoring the real population is GPU work this layer refuses to fake. The returned plan reports
        whether it fits the budget so an operator can size the run before dispatching it on hardware.
        """
        store = store if store is not None else default_store()
        cells: list[SweepCell] = []
        for fp in signals:
            for obs in battery:
                prior = store.find(observable=obs, signal=fp, latest=True)
                if prior:
                    est = Cost(gpu_seconds=prior[-1].provenance.cost.gpu_seconds)
                    source = "prior-evidence"
                else:
                    est = Cost()
                    source = "unestimated"
                cells.append(
                    SweepCell(fingerprint=fp, observable=obs, estimated=est, estimate_source=source)
                )
        plan = SweepPlan(cells=cells, budget=budget)
        if execute:
            raise SweepGatedError(
                "the real Atlas sweep loads and scores reward models on a GPU and is not run from "
                "the artifacts layer. Dispatch the plan on hardware via the study runner (each cell "
                "is a battery Observable under a frozen sweep study), then read the results back "
                "through leaderboard(). " + plan.summary()
            )
        return plan


__all__ = [
    "ProvenanceTier",
    "ModelLineage",
    "declared_fingerprint",
    "AtlasEntry",
    "LeaderboardCell",
    "Leaderboard",
    "SweepGatedError",
    "SweepCell",
    "SweepPlan",
    "Atlas",
]
