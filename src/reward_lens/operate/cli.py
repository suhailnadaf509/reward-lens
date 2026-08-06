"""The ``reward-lens`` command line.

The operator surface over the kernel and the artifacts layer. The commands split cleanly along the
line the design draws: the ones that are views over the evidence store run here and now with no
model and no GPU (``card``, ``scoreboard``, ``claims``, ``atlas export``, and the planning half of
``atlas sweep`` and ``study freeze``), and the ones that touch a real reward model (``score``,
``serve``, ``audit``, ``organism train``/``score``, ``study run``) dispatch to the exact kernel call
that does the work and are marked GPU-gated. GPU-gated is not a stub that lies: the command names the
call it would make and refuses rather than fabricating a number, and where an ``--execute`` flag
exists it will make the real call on hardware. The module imports nothing heavier than typer and the
torch-free artifacts and studies layers; every model-touching import happens lazily inside a command
body, so ``import reward_lens.operate`` stays torch-free.
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from reward_lens.access import (
    ProbeBudget,
    capability_report,
    classify_substrate,
    http_endpoint,
    resolve_access,
    resolve_phase,
)
from reward_lens.artifacts.atlas import Atlas
from reward_lens.artifacts.card import build_card
from reward_lens.artifacts.claims import check_files
from reward_lens.core.provenance import Cost
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.studies.scoreboard import Scoreboard

# Exit code for a GPU-gated command asked to do model work it cannot do here.
GPU_GATED_EXIT = 3

app = typer.Typer(
    name="reward-lens",
    help="Operator surface for the reward-lens kernel: cards, scoreboard, claims, and the Atlas.",
    no_args_is_help=True,
    add_completion=False,
)
study_app = typer.Typer(
    help="Freeze, run, and report frozen studies (gate 3).", no_args_is_help=True
)
atlas_app = typer.Typer(help="The reward-model population Atlas.", no_args_is_help=True)
organism_app = typer.Typer(help="The ground-truth organism foundry.", no_args_is_help=True)
app.add_typer(study_app, name="study")
app.add_typer(atlas_app, name="atlas")
app.add_typer(organism_app, name="organism")

console = Console()
err_console = Console(stderr=True)


def _store(path: Optional[Path]) -> EvidenceStore:
    """Resolve the evidence store: the given directory, or the configured default."""
    return EvidenceStore(path) if path is not None else default_store()


def _gpu_gated(operation: str, dispatch: str, detail: str = "") -> None:
    """Print the GPU-gated notice for a model-touching command and exit (never fabricates).

    Names the exact kernel call the operation dispatches to, so the notice is a pointer to the real
    work rather than a dead end, and exits with ``GPU_GATED_EXIT``. This is the honest behavior the
    design requires: a torch-free operator layer refuses model work instead of inventing a result.
    """
    body = (
        f"[bold]{operation}[/bold] needs a loaded reward model and a GPU.\n"
        "It is GPU-gated on this torch-free operator layer.\n\n"
        f"Dispatches to: [cyan]{dispatch}[/cyan]"
    )
    if detail:
        body += f"\n\n{detail}"
    err_console.print(Panel.fit(body, title="GPU-gated", border_style="yellow"))
    raise typer.Exit(code=GPU_GATED_EXIT)


# ---------------------------------------------------------------------------
# Pure commands: views over the store
# ---------------------------------------------------------------------------


@app.command()
def card(
    signal: str = typer.Argument(..., help="The model fingerprint (mfp:...) to build a card for."),
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
    fmt: str = typer.Option("json", "--format", help="json or html."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write to this file instead of stdout."),
) -> None:
    """Build an RM Card for a signal: a view over every stored Evidence about it."""
    c = build_card(signal, _store(store))
    text = c.to_html() if fmt == "html" else c.to_json()
    if out is not None:
        out.write_text(text, encoding="utf-8")
        console.print(f"Wrote {fmt} card for {signal} to {out}")
    else:
        typer.echo(text)


@app.command()
def scoreboard(
    path: Optional[Path] = typer.Option(None, "--path", help="Persisted scoreboard JSON."),
) -> None:
    """Print the theorem scoreboard: standing theorems and candidate laws."""
    sb = Scoreboard(path)
    typer.echo(sb.render_markdown())


@app.command()
def claims(
    files: List[Path] = typer.Argument(..., help="Documents to check for unbound numeric claims."),
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
) -> None:
    """Check documents against the store; exit nonzero if any number is unbound.

    A claim tagged with an Evidence id that the store does not contain, a claimed value that
    disagrees with the stored one, or a dangling ``ev:`` reference is a failure. This is the CI
    entry point that keeps a manuscript from claiming a number the store cannot back.
    """
    report = check_files([str(f) for f in files], _store(store))
    typer.echo(report.render())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def capabilities(
    record: Optional[Path] = typer.Option(None, "--record", help="A run record directory."),
    grader: Optional[str] = typer.Option(
        None, "--grader", help="A grader scoring endpoint, as a URL."
    ),
    policy: Optional[Path] = typer.Option(
        None, "--policy", help="A policy checkpoint directory; adds POLICY: FORWARD."
    ),
    env: Optional[Path] = typer.Option(
        None, "--env", help="An environment source tree; adds TASK: MUTATE."
    ),
    substrate: Optional[str] = typer.Option(
        None, "--substrate", help="Declare the grader's substrate instead of classifying it."
    ),
    phase: Optional[str] = typer.Option(
        None, "--phase", help="Declare the phase instead of reading it from the record."
    ),
    probe: int = typer.Option(
        0,
        "--probe",
        help="Authorise up to this many real grader calls to establish QUERY and REPLICATE. "
        "Three calls settle both. The default of 0 makes none.",
    ),
    show_all: bool = typer.Option(
        False, "--all", help="List every catalogued instrument that is not built yet."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """What you could learn about this run, and what it would cost to learn more.

    Resolves the four typing dimensions from what you supplied, then prints which quantities are
    reachable at what rung and price, and for everything else a refusal carrying an instruction.

    No model is loaded and no compute runs. No grader call is made either unless ``--probe`` gives
    a budget above zero, because `REPLICATE` cannot be established without calling the endpoint and
    calling somebody's endpoint costs them money.
    """
    endpoint: object | None = grader
    if grader is not None and probe > 0 and grader.startswith(("http://", "https://")):
        endpoint = http_endpoint(grader)

    access = resolve_access(
        record=record,
        grader=endpoint,
        policy=policy,
        environment=env,
        probe=ProbeBudget(calls=probe),
    )
    substrate_reading = classify_substrate(None, declared=substrate)
    phase_resolution = resolve_phase(record=record, declared=phase)
    report = capability_report(access, substrate_reading, phase_resolution, None)
    if as_json:
        import json

        typer.echo(json.dumps(report.to_dict(), indent=2))
        return
    typer.echo(report.render(show_not_built=show_all))


@atlas_app.command("export")
def atlas_export(
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
    observables: Optional[str] = typer.Option(
        None,
        "--observables",
        help="Comma-separated observable names; defaults to those in the store.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Directory to write leaderboard.{json,html}."
    ),
) -> None:
    """Export the Atlas leaderboard to JSON and HTML: a view over the store."""
    obs = [o.strip() for o in observables.split(",") if o.strip()] if observables else None
    result = Atlas.standard().export_leaderboard(store=_store(store), observables=obs, out_dir=out)
    if out is not None:
        console.print(f"Wrote {result['json_path']} and {result['html_path']}")
    else:
        typer.echo(result["json"])


# ---------------------------------------------------------------------------
# Atlas sweep: plans purely, executes GPU-gated
# ---------------------------------------------------------------------------


@atlas_app.command("sweep")
def atlas_sweep(
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
    battery: str = typer.Option(
        "BiasBattery,DistortionV2,RobustnessSNR",
        "--battery",
        help="Comma-separated observable names to sweep.",
    ),
    gpu_budget: float = typer.Option(0.0, "--gpu-budget", help="GPU-seconds budget for the sweep."),
    execute: bool = typer.Option(False, "--execute", help="Run the real sweep (GPU-gated)."),
) -> None:
    """Plan and price the population sweep over the standard Atlas.

    Builds the cartesian product of the standard population and the requested battery into a plan,
    pricing each cell from any prior metered cost in the store. Planning runs here; ``--execute``
    dispatches the real sweep, which is GPU work and is gated.
    """
    atlas = Atlas.standard()
    obs = [o.strip() for o in battery.split(",") if o.strip()]
    fps = [e.fingerprint for e in atlas.entries]
    plan = atlas.sweep(fps, obs, Cost(gpu_seconds=gpu_budget), store=_store(store), execute=False)
    console.print(plan.summary())
    if execute:
        _gpu_gated(
            "atlas sweep --execute",
            "reward_lens.studies.run_study(sweep_spec) over each battery Observable, then leaderboard()",
            detail=f"{len(plan.cells)} cells planned; run them on hardware and read results back.",
        )


# ---------------------------------------------------------------------------
# Studies: freeze is pure; run and report dispatch to the runner
# ---------------------------------------------------------------------------


def _load_spec(spec_path: str):
    """Load a StudySpec from a ``module:attr`` or ``module.attr`` dotted path."""
    import importlib

    from reward_lens.studies.spec import StudySpec

    if ":" in spec_path:
        module_name, _, attr = spec_path.partition(":")
    else:
        module_name, _, attr = spec_path.rpartition(".")
    if not module_name or not attr:
        raise typer.BadParameter(f"spec '{spec_path}' must be a module:attr or module.attr path")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr, None)
    if not isinstance(obj, StudySpec):
        raise typer.BadParameter(f"'{spec_path}' does not resolve to a StudySpec")
    return obj


@study_app.command("freeze")
def study_freeze(
    spec: str = typer.Argument(..., help="Dotted path to a StudySpec (module:attr)."),
) -> None:
    """Freeze a study spec: hash it and record the git sha, yielding its StudyID (gate 3)."""
    from reward_lens.studies.freeze import freeze

    frozen = freeze(_load_spec(spec))
    console.print(f"Frozen study [bold]{frozen.study_id}[/bold]")
    console.print(f"  spec hash: {frozen.spec_hash}")
    console.print(f"  git sha:   {frozen.git_sha}")
    console.print(f"  frozen at: {frozen.frozen_at}")


@study_app.command("run")
def study_run(
    spec: str = typer.Argument(..., help="Dotted path to a StudySpec (module:attr)."),
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
    execute: bool = typer.Option(False, "--execute", help="Run the analysis (may be GPU-gated)."),
) -> None:
    """Run a frozen study end to end.

    Freezes the spec, then dispatches to ``run_study``, whose analysis function resolves and measures
    against real subjects. Most analyses touch a model, so running is GPU-gated by default;
    ``--execute`` performs the real dispatch.
    """
    from reward_lens.studies.freeze import freeze

    frozen = freeze(_load_spec(spec))
    console.print(f"Frozen study [bold]{frozen.study_id}[/bold] (analysis: {frozen.spec.analysis})")
    if not execute:
        _gpu_gated(
            "study run",
            f"reward_lens.studies.run_study({frozen.study_id})",
            detail="The analysis resolves subjects and measures them; pass --execute to run it.",
        )
    from reward_lens.studies.runner import run_study

    frozen, result = run_study(frozen, store=_store(store))
    console.print(f"Outcomes: {result.outcomes}")
    if result.killed:
        console.print(f"[bold red]Kill criterion fired:[/bold red] {result.killed_by}")


@study_app.command("report")
def study_report(
    spec: str = typer.Argument(..., help="Dotted path to a StudySpec (module:attr)."),
    store: Optional[Path] = typer.Option(None, "--store", help="Evidence store directory."),
    execute: bool = typer.Option(False, "--execute", help="Run then render (may be GPU-gated)."),
) -> None:
    """Render a study report.

    A report is a view over a completed run, so it runs the study to obtain the result and then
    renders it. Running is GPU-gated by default for the same reason ``study run`` is; ``--execute``
    performs the real dispatch and prints the markdown report.
    """
    from reward_lens.studies.freeze import freeze

    frozen = freeze(_load_spec(spec))
    if not execute:
        _gpu_gated(
            "study report",
            f"run_study({frozen.study_id}) then reward_lens.studies.render_report(...)",
            detail="A report renders a completed run; pass --execute to run and render.",
        )
    from reward_lens.studies.report import render_report
    from reward_lens.studies.runner import run_study

    s = _store(store)
    frozen, result = run_study(frozen, store=s)
    typer.echo(render_report(frozen, result, s))


# ---------------------------------------------------------------------------
# Model-touching commands: dispatch to the kernel, GPU-gated
# ---------------------------------------------------------------------------


@app.command()
def score(
    signal: str = typer.Argument(..., help="Signal to load and score (repo id or fingerprint)."),
) -> None:
    """Score inputs with a reward model (GPU-gated).

    Dispatches to the signals loader and the RewardSignal protocol's ``score``. Loading an 8B reward
    model and running a forward pass is GPU work this layer does not do; the real call is named below.
    """
    _gpu_gated(
        "score",
        f"reward_lens.signals.load_signal({signal!r}).score(view) -> Evidence[Scores]",
    )


@app.command()
def serve(
    signal: str = typer.Argument(..., help="Signal to serve as a reward endpoint."),
) -> None:
    """Serve a reward model as an RL-loop-compatible endpoint (GPU-gated)."""
    _gpu_gated(
        "serve",
        f"reward_lens.signals.serve.serve(load_signal({signal!r})) (OpenRLHF/TRL/veRL proxy)",
        detail="The reward server holds a model resident on the GPU; run it on hardware.",
    )


@app.command()
def audit(
    signal: str = typer.Argument(..., help="Signal to audit in the blind auditing game."),
    organism: Optional[str] = typer.Option(
        None, "--organism", help="Organism id to audit against."
    ),
) -> None:
    """Run the blind auditing game against a signal or organism (GPU-gated).

    The auditing game hands an operator a signal plus reward-lens and scores whether they can name
    the planted rule. It needs a loaded model and an organism with an answer key, so it is gated.
    """
    _gpu_gated(
        "audit",
        "reward_lens.organisms.game.AuditingGame(signal, organism).run()",
        detail="Blind operators (human or MCP agent) play against a planted-rule organism.",
    )


@organism_app.command("make")
def organism_make(
    spec: str = typer.Argument(..., help="Dotted path to a RuleSpec, or a rule name."),
) -> None:
    """Generate an organism dataset from a rule spec (GPU-gated at training).

    Dispatches to the foundry. Data generation is CPU work but the foundry and its training recipes
    live behind torch; the make/train/score sequence is gated here and run on hardware.
    """
    _gpu_gated(
        "organism make",
        f"reward_lens.organisms.foundry.generate({spec!r}) -> organism dataset + AnswerKey",
    )


@organism_app.command("train")
def organism_train(
    spec: str = typer.Argument(..., help="Organism id or spec to train."),
) -> None:
    """Train an organism reward model with a planted rule (GPU-gated)."""
    _gpu_gated(
        "organism train",
        f"reward_lens.organisms.train.train({spec!r}) (LoRA/full-FT recipe)",
        detail="Training a reward model with a planted channel is GPU work; run it on hardware.",
    )


@organism_app.command("score")
def organism_score(
    spec: str = typer.Argument(..., help="Organism id to score with the battery."),
) -> None:
    """Score an organism with the battery and produce its MethodScorecard (GPU-gated)."""
    _gpu_gated(
        "organism score",
        f"reward_lens.organisms.scorecard.score({spec!r}) -> MethodScorecard (instrument ROC)",
    )


def main() -> None:
    """Console-script entry point (``reward-lens``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
