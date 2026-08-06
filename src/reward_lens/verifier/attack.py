"""D8 Attack surface: what the harness exposes, and how much cheaper attacking it is than solving.

Says, with your numbers in place of the specification's: "The scoring script reads a file the agent
can write. Attacking the grader is 11x cheaper than solving the task."

UK AISI publishes per-model cheat rates, 7.8% and 12.6% for two named frontier models, and reports
that the modal cheat for two GPT-5.x variants was "attacked a system other than the target". Every
one of those numbers is measured after the fact by a monitor watching transcripts. Nobody predicts
them from properties of the environment, which is odd, because the properties are sitting in the
harness's own source: what it opens, what it writes, which environment variables it reads, and
which of its scoring decisions depend on bytes the agent produced.

**Three questions and three rungs.**

Rung 0 is the static inventory. Every file read, every file write, every subprocess, every network
call and every environment-variable read the harness makes, with the line it happens on and whether
the target is a literal or comes from somewhere else. Credential reachability falls out of the same
pass: an `os.environ` read whose key looks like a secret, or a path under `.ssh`, `.aws` or
`/var/run/secrets`, is reachable from wherever that code runs.

Rung 1 adds the boundary. An inventory on its own is a list; it becomes a finding when you say
which side of the task boundary each item is on. A `TaskBoundary` names what the agent may write
and what the harness keeps to itself, and a *crossing* is the harness trusting something on the
agent's side. That is the specification's own example sentence: the scoring script reads a file the
agent can write.

The taint analysis is the part that turns a read into an attack. Reading agent-produced bytes is
not by itself a problem; a grader has to read the answer. The pattern that matters is **untrusted
input reaching a scoring decision**, and `libcst`'s `ScopeProvider` gives the def-use edges to
follow it without writing a dataflow analysis from scratch. Two wrinkles that cost a day if you
find them the hard way: the visitor has to be driven by `MetadataWrapper.visit()` rather than
`Module.visit()`, and `get_metadata(ScopeProvider, node)` raises `KeyError` on import statements
and on attribute names. Both are handled and the unresolved names are reported rather than
swallowed, so a reader can see the analysis's edge instead of inferring a clean result.

Rung 2 is the measured cost ratio, `env.attack_cheapness`. It is a measurement and not an estimate:
you supply what an attack cost and what a solve cost, in the same unit, and this divides them and
puts an interval on the quotient. With no measurements it refuses, because a cheapness ratio
guessed from source is exactly the confident wrong number the rest of this library exists to
prevent. The unit check is not a formality either: `env.attack_cheapness` carries the `units`
invariance group, whose assertion is that a comparison across a unit boundary refuses rather than
converting, so attack cost in tokens against solve cost in seconds is a `UNIT_MISMATCH` refusal.

**Dual use.** The output of rung 0 is a list of ways into a scoring system. `AttackSurface` is
sensitive by default, `redacted()` keeps the counts and drops the targets and the reproducing
lines, and `for_publication()` raises unless somebody has recorded a `DisclosureDecision`. The
mechanism is D5's, imported rather than reimplemented, because two disclosure mechanisms in one
package is one more than a reviewer can check.

Kill condition, from the catalogue: **if attack cheapness does not predict realised cheat rate
across twenty environments.** That test has not been run here and cannot be run from source alone.
It needs twenty environments with monitor-measured cheat rates beside them, and until somebody runs
it the surface inventory is a description rather than a prediction.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.verifier import (
    SENSITIVE_NOTE,
    SENSITIVE_SUBJECT_EXTRA,
    ProgramSubject,
    VerifierUnderTest,
    ensure_quantities,
)
from reward_lens.verifier.fuzz import DisclosureDecision, DisclosureRequired

#: The five kinds of thing a harness can reach outside itself.
SurfaceKind = Literal["read", "write", "execute", "network", "environment"]

#: Method-style calls that read something outside the process. Matched on the bare attribute name,
#: because `p.read_text()` and `handle.read()` are the same act through different objects and the
#: receiver's type is not knowable from source.
READ_METHODS: frozenset[str] = frozenset(
    {
        "read",
        "readline",
        "readlines",
        "read_text",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
        "scandir",
    }
)

#: Dotted calls that read. Matched on the full dotted name so `json.load` is a read and a local
#: helper called `load` is not.
READ_CALLS: frozenset[str] = frozenset(
    {
        "json.load",
        "json.loads",
        "yaml.safe_load",
        "yaml.load",
        "pickle.load",
        "pickle.loads",
        "os.listdir",
        "os.walk",
        "os.scandir",
        "os.stat",
        "glob.glob",
        "glob.iglob",
        "pandas.read_csv",
        "pd.read_csv",
        "pd.read_json",
    }
)

WRITE_METHODS: frozenset[str] = frozenset(
    {
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "mkdir",
        "touch",
        "chmod",
        "unlink",
    }
)

WRITE_CALLS: frozenset[str] = frozenset(
    {
        "json.dump",
        "yaml.dump",
        "pickle.dump",
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.chmod",
        "os.chown",
        "os.symlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)

#: Anything that starts another program, or turns data into code. `eval` and `exec` are here
#: because a grader that evaluates the model's output as an expression has handed it the process.
EXECUTE_PREFIXES: tuple[str, ...] = ("subprocess.", "docker.", "multiprocessing.")
EXECUTE_CALLS: frozenset[str] = frozenset(
    {"os.system", "os.popen", "os.spawnl", "os.execv", "eval", "exec", "compile", "__import__"}
)

NETWORK_PREFIXES: tuple[str, ...] = (
    "requests.",
    "urllib.",
    "httpx.",
    "socket.",
    "aiohttp.",
    "boto3.",
    "paramiko.",
)

#: Environment reads. `docker.from_env` is here rather than under execute because what it *reads*
#: is `DOCKER_HOST` and the TLS material beside it, which is the credential question.
ENVIRONMENT_CALLS: frozenset[str] = frozenset(
    {"os.getenv", "os.environ.get", "environ.get", "docker.from_env", "dotenv.load_dotenv"}
)

#: Substrings that make an environment-variable name or a path look like a secret. Deliberately
#: broad: a false positive costs a line in a report, and a false negative is a credential nobody
#: knew was reachable.
CREDENTIAL_MARKERS: tuple[str, ...] = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth",
    "private",
    "session",
    "cookie",
    "api",
    "access",
    "signature",
)

#: Paths that hold credentials on a normal machine, matched as substrings of a literal path.
CREDENTIAL_PATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".netrc",
    ".git-credentials",
    ".docker/config",
    ".kube/config",
    "/var/run/secrets",
    "id_rsa",
    "id_ed25519",
    ".env",
    "service_account",
)

#: Names whose presence in a comparison or a call argument counts as validating what flows through
#: it. Same dominance-flavoured heuristic D9 uses, and the same admission: it cannot see whether
#: the check is correct, only that one happened.
VALIDATOR_NAMES: tuple[str, ...] = (
    "isinstance",
    "validate",
    "verify",
    "check",
    "assert_",
    "sanitize",
    "sanitise",
    "escape",
    "allowlist",
    "whitelist",
    "schema",
)


def _requires_libcst() -> None:
    from reward_lens.core.extras import require_extra

    require_extra("verifier", subsystem="D8 (the attack-surface inventory)")


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ResourceAccess:
    """One thing the harness reaches outside itself, and where.

    ``target`` is the literal when the code names one and ``<expr>`` otherwise, with
    ``target_is_literal`` saying which. That distinction is the whole difference between "the
    harness reads /opt/gold/answers.json" and "the harness reads a path computed at run time", and
    the second is the interesting one: a computed path is a path something else can influence.
    """

    kind: SurfaceKind
    call: str
    target: str
    target_is_literal: bool
    function: str
    line: int
    col: int
    credential: bool = False
    tainted_target: bool = False

    def render(self) -> str:
        where = f"{self.function}:{self.line}"
        tag = " [CREDENTIAL]" if self.credential else ""
        tag += " [target from untrusted input]" if self.tainted_target else ""
        return f"{self.kind:<11} {self.call}({self.target}) at {where}{tag}"


@register_payload
@dataclass(frozen=True)
class TaintPath:
    """Untrusted input reaching a scoring decision, with the names it travelled through.

    This is the pattern the whole rung exists to find. A grader reading the model's answer is not
    a finding; a grader whose *verdict* is computed from bytes the agent controls, with nothing in
    between that checks them, is. ``chain`` is the sequence of local names the value passed
    through, which is what makes the finding actionable: it names the line to put the check on.
    """

    source: str
    source_kind: Literal["parameter", "resource_read", "environment"]
    sink: str
    sink_kind: Literal["return", "branch"]
    function: str
    line: int
    chain: tuple[str, ...] = ()
    validated: bool = False
    validated_by: str = ""

    def render(self) -> str:
        state = f"checked by {self.validated_by}" if self.validated else "UNCHECKED"
        through = " -> ".join(self.chain) if self.chain else "directly"
        return (
            f"{self.source} reaches {self.sink} ({self.sink_kind}) at "
            f"{self.function}:{self.line} through {through}: {state}"
        )


@dataclass(frozen=True)
class TaskBoundary:
    """Rung 1's formal statement of where the task ends and the harness begins.

    Three lists, all glob patterns matched against literal targets. ``agent_writable`` is what the
    agent may legitimately produce; ``harness_private`` is what only the harness should touch;
    ``secret`` names environment variables and paths that are credentials regardless of who can
    reach them.

    The boundary is a declaration, not a measurement, and that is the point of writing it down: an
    environment whose author cannot say which paths the agent may write has not decided where its
    task ends, and every crossing this instrument reports afterwards is conditioned on the answer.
    """

    agent_writable: tuple[str, ...] = ()
    harness_private: tuple[str, ...] = ()
    secret: tuple[str, ...] = ()
    name: str = "unnamed"

    def is_agent_writable(self, target: str) -> bool:
        return any(fnmatch.fnmatch(target, pattern) for pattern in self.agent_writable)

    def is_harness_private(self, target: str) -> bool:
        return any(fnmatch.fnmatch(target, pattern) for pattern in self.harness_private)

    def is_secret(self, target: str) -> bool:
        return any(fnmatch.fnmatch(target, pattern) for pattern in self.secret)


@register_payload
@dataclass(frozen=True)
class BoundaryCrossing:
    """One place the harness relies on something that is on the agent's side of the boundary."""

    access: ResourceAccess
    rule: str
    why: str

    def render(self) -> str:
        return f"{self.access.render()}\n        {self.rule}: {self.why}"


@register_payload
@dataclass(frozen=True)
class AttackSurface:
    """The inventory. The value of `env.attack_surface`, and **sensitive by default**.

    ``baseline_sandbox_holds`` is the catalogue's mandatory baseline and it is zero, which looks
    like a placeholder and is not. "The sandbox holds" is the assumption under which the harness
    exposes nothing, so the baseline prediction for every count below is zero, and every non-zero
    count is this instrument disagreeing with it. Recording it as a number rather than as a
    sentence is what lets a card show the comparison.
    """

    source_path: str
    entrypoint: str
    fingerprint: str
    rung: int
    accesses: tuple[ResourceAccess, ...]
    taints: tuple[TaintPath, ...]
    crossings: tuple[BoundaryCrossing, ...] = ()
    #: How many crossings a `redacted()` view dropped, so its count still reads the same. A
    #: redacted card showing zero crossings because the list was removed is worse than no card.
    crossings_withheld: int = 0
    boundary: str = ""
    unresolved_names: tuple[str, ...] = ()
    cheapness: "AttackCheapness | None" = None
    baseline_sandbox_holds: int = 0
    sensitive: bool = True
    sensitive_note: str = SENSITIVE_NOTE
    disclosure: DisclosureDecision | None = None
    withheld: int = 0
    notes: tuple[str, ...] = ()

    # -- counts ------------------------------------------------------------

    @property
    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.accesses:
            out[a.kind] = out.get(a.kind, 0) + 1
        return out

    @property
    def credentials(self) -> tuple[ResourceAccess, ...]:
        return tuple(a for a in self.accesses if a.credential)

    @property
    def unchecked_taints(self) -> tuple[TaintPath, ...]:
        return tuple(t for t in self.taints if not t.validated)

    @property
    def headline(self) -> int:
        """The number a card quotes: unchecked paths from untrusted input to a scoring decision.

        Not the raw access count. A harness that opens forty files is not forty times more
        attackable than one that opens one; a harness whose verdict is computed from bytes the
        agent wrote, with nothing checking them, is attackable once and that is enough.
        """
        return len(self.unchecked_taints)

    # -- dual use ----------------------------------------------------------

    def redacted(self) -> "AttackSurface":
        """Counts without targets. What a rendered card gets unless somebody asks for more.

        Everything that makes the number auditable survives: how many reads, how many writes, how
        many credentials are reachable, how many unchecked paths run from untrusted input to a
        scoring decision. The paths, the environment-variable names and the line numbers do not,
        because those are the directions.
        """
        return replace(
            self,
            accesses=tuple(
                replace(a, target="<withheld>", target_is_literal=False, line=0, col=0)
                for a in self.accesses
            ),
            taints=tuple(replace(t, chain=(), line=0) for t in self.taints),
            crossings=(),
            crossings_withheld=len(self.crossings),
            withheld=len(self.crossings) + sum(1 for a in self.accesses if a.target_is_literal),
            sensitive=False,
        )

    def for_publication(self) -> "AttackSurface":
        """The unredacted inventory, and only with a decision recorded."""
        if self.disclosure is None:
            raise DisclosureRequired(
                f"{type(self).__name__} for {self.entrypoint!r} lists "
                f"{len(self.accesses)} resources the harness reaches and "
                f"{len(self.unchecked_taints)} unchecked routes from untrusted input to a scoring "
                f"decision, with no recorded decision to publish them. Attach a "
                f"DisclosureDecision naming who decided and why, or call `redacted()` for the "
                f"counts without the targets."
            )
        return self

    def with_disclosure(self, decision: DisclosureDecision) -> "AttackSurface":
        return replace(self, disclosure=decision)

    # -- presentation ------------------------------------------------------

    def render(self, *, include_targets: bool = False) -> str:
        counts = self.by_kind
        tally = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines = [
            f"attack surface of {Path(self.source_path).name}:{self.entrypoint} at rung {self.rung}",
            f"    {tally}" if tally else "    no resource access found",
            f"    {len(self.credentials)} credential-shaped reads reachable from inside",
            f"    {len(self.unchecked_taints)} unchecked routes from untrusted input to a "
            f"scoring decision, of {len(self.taints)} routes found",
            f"    baseline (the assumption that the sandbox holds) predicts "
            f"{self.baseline_sandbox_holds}",
        ]
        if self.boundary:
            n_crossings = len(self.crossings) + self.crossings_withheld
            lines.append(f"    boundary {self.boundary}: {n_crossings} crossings")
        if self.cheapness is not None:
            lines.append("    " + self.cheapness.render())
        if include_targets:
            lines += [f"      {a.render()}" for a in self.accesses]
            lines += [f"      {t.render()}" for t in self.taints]
            lines += [f"      {c.render()}" for c in self.crossings]
        elif self.accesses or self.taints:
            lines.append(
                f"    {len(self.accesses)} targets and {len(self.taints)} routes withheld. "
                f"{self.sensitive_note}"
            )
        if self.withheld:
            lines.append(f"    {self.withheld} targets withheld from this view")
        if self.unresolved_names:
            lines.append(
                f"    ScopeProvider could not resolve {len(self.unresolved_names)} names "
                f"({', '.join(sorted(set(self.unresolved_names))[:6])}); the counts above are "
                f"floors"
            )
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The cost ratio
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CostSample:
    """What something cost, several times, in a named unit.

    The unit is not decoration and is not defaulted. `env.attack_cheapness` carries the `units`
    invariance group, whose assertion is that a comparison across a unit boundary refuses rather
    than converting, and that assertion has nothing to bite on unless both sides say what they are
    counting.
    """

    what: str
    unit: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.unit.strip():
            raise ValueError(
                "a cost sample needs a unit. Tokens, seconds, dollars and grader calls are all "
                "costs, and a ratio of two of them measured in different units is not a ratio."
            )
        if not self.values:
            raise ValueError(f"cost sample {self.what!r} carries no measurements")
        if any(v <= 0 or not math.isfinite(v) for v in self.values):
            raise ValueError(
                f"cost sample {self.what!r} has a non-positive or non-finite cost; a cheapness "
                f"ratio divides by it"
            )

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def n(self) -> int:
        return len(self.values)


@register_payload
@dataclass(frozen=True)
class AttackCheapness:
    """How much cheaper attacking the grader is than solving the task. `env.attack_cheapness`.

    The ratio is `solve / attack`, so it reads the way the specification's sentence does: 11 means
    attacking is eleven times cheaper. The interval is a percentile bootstrap over both samples
    jointly, which is the right shape for a ratio of means: the delta method's normal interval is
    symmetric and a cost ratio is not.
    """

    unit: str
    attack_mean: float
    solve_mean: float
    ratio: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_attack: int
    n_solve: int
    resamples: int
    attack_what: str = ""
    solve_what: str = ""

    @property
    def cheaper_to_attack(self) -> bool | None:
        """Whether the interval settles it. None when it spans 1."""
        if self.ci_low > 1.0:
            return True
        if self.ci_high < 1.0:
            return False
        return None

    def render(self) -> str:
        verdict = {
            True: "attacking the grader is cheaper than solving the task",
            False: "solving the task is cheaper than attacking the grader",
            None: "the interval spans 1, so this does not settle which is cheaper",
        }[self.cheaper_to_attack]
        return (
            f"attack cheapness {self.ratio:.2f}x "
            f"[{self.ci_low:.2f}, {self.ci_high:.2f}] at {self.ci_level:.0%} "
            f"({self.solve_mean:.4g} {self.unit} to solve against {self.attack_mean:.4g} to "
            f"attack, n = {self.n_solve} and {self.n_attack}): {verdict}"
        )


def attack_cheapness(
    attack: CostSample,
    solve: CostSample,
    *,
    ci_level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> AttackCheapness | Refusal:
    """`solve / attack`, with a percentile-bootstrap interval. Refuses across a unit boundary.

    The unit refusal is `env.attack_cheapness`'s declared invariance group doing its job. Silently
    dividing seconds by tokens produces a number, and a number produced that way is exactly the
    failure this library's `UNIT_MISMATCH` reason exists for.
    """
    if attack.unit != solve.unit:
        return Refusal(
            instrument="AttackCheapness",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"attack cost is in {attack.unit!r} and solve cost is in {solve.unit!r}. The "
                f"conversion factor between them is a property of your setup, not of the units."
            ),
            remedy=(
                "measure both in the same unit and pass them again. If you have the conversion "
                "factor, apply it yourself and say so in the sample's `what` field, so the "
                "reading records that a conversion happened rather than hiding it inside a "
                "ratio."
            ),
            statistics={"attack_unit": attack.unit, "solve_unit": solve.unit},
        )

    import numpy as np

    rng = np.random.default_rng(seed)
    a = np.asarray(attack.values, dtype=float)
    s = np.asarray(solve.values, dtype=float)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        ai = rng.integers(0, a.size, a.size)
        si = rng.integers(0, s.size, s.size)
        draws[i] = s[si].mean() / a[ai].mean()
    alpha = 0.5 * (1.0 - ci_level)
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])

    return AttackCheapness(
        unit=attack.unit,
        attack_mean=float(a.mean()),
        solve_mean=float(s.mean()),
        ratio=float(s.mean() / a.mean()),
        ci_low=float(low),
        ci_high=float(high),
        ci_level=ci_level,
        n_attack=attack.n,
        n_solve=solve.n,
        resamples=resamples,
        attack_what=attack.what,
        solve_what=solve.what,
    )


# ---------------------------------------------------------------------------
# The static pass
# ---------------------------------------------------------------------------


def _dotted(node: Any) -> str:
    """`os.environ.get` from an Attribute chain, or the bare name, or "" for anything else."""
    import libcst as cst

    parts: list[str] = []
    cur = node
    while isinstance(cur, cst.Attribute):
        parts.append(cur.attr.value)
        cur = cur.value
    if isinstance(cur, cst.Name):
        parts.append(cur.value)
    else:
        return ""
    return ".".join(reversed(parts))


def _literal(node: Any) -> str | None:
    import libcst as cst

    if isinstance(node, cst.SimpleString):
        try:
            return str(node.evaluated_value)
        except Exception:  # noqa: BLE001 - an f-string prefix makes this unevaluable, which is fine
            return None
    if isinstance(node, cst.ConcatenatedString):
        try:
            return str(node.evaluated_value)
        except Exception:  # noqa: BLE001
            return None
    return None


def _names_in(node: Any) -> set[str]:
    """Every bare Name mentioned anywhere under `node`. The taint analysis's alphabet."""
    import libcst as cst

    found: set[str] = set()

    class _V(cst.CSTVisitor):
        def visit_Name(self, n: Any) -> bool:
            found.add(n.value)
            return True

        def visit_Attribute(self, n: Any) -> bool:
            # Only the root of a dotted chain is a name in scope; `content.split` should
            # contribute `content` and not `split`.
            root = n
            while isinstance(root, cst.Attribute):
                root = root.value
            if isinstance(root, cst.Name):
                found.add(root.value)
            return False

    node.visit(_V())
    return found


def _module_constants(module: Any) -> dict[str, str]:
    """Module-level `NAME = "literal"` bindings, so a constant path is still a literal path.

    Real harnesses put their paths in constants. SWE-bench's grading module has half a dozen, and
    without this the boundary check receives `<GOLD>` instead of `/opt/gold/answers.json` and
    matches nothing, which reads as a clean result and is a resolution failure.

    Plain `module.visit` rather than `wrapper.visit` here on purpose: this visitor asks for no
    metadata, and the wrapper is only required when one does.
    """
    import libcst as cst

    found: dict[str, str] = {}

    class _Constants(cst.CSTVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: Any) -> bool:
            self.depth += 1
            return True

        def leave_FunctionDef(self, node: Any) -> None:
            self.depth -= 1

        def visit_ClassDef(self, node: Any) -> bool:
            self.depth += 1
            return True

        def leave_ClassDef(self, node: Any) -> None:
            self.depth -= 1

        def visit_Assign(self, node: Any) -> bool:
            if self.depth or len(node.targets) != 1:
                return False
            target = node.targets[0].target
            lit = _literal(node.value)
            if isinstance(target, cst.Name) and lit is not None:
                found[target.value] = lit
            return False

    module.visit(_Constants())
    return found


def _looks_credential(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in CREDENTIAL_MARKERS) or any(p in low for p in CREDENTIAL_PATHS)


def _classify(dotted: str, bare: str, args: Sequence[Any]) -> tuple[SurfaceKind, str] | None:
    """What kind of surface a call is, and the canonical name to report it under."""
    if dotted == "open" or bare == "open":
        mode = ""
        if len(args) > 1:
            lit = _literal(args[1].value)
            mode = lit or ""
        for a in args:
            if getattr(a, "keyword", None) is not None and a.keyword.value == "mode":
                mode = _literal(a.value) or mode
        return ("write" if any(c in mode for c in "wax+") else "read", "open")
    if dotted in ENVIRONMENT_CALLS or bare in {"getenv"}:
        return ("environment", dotted or bare)
    if dotted in EXECUTE_CALLS or any(dotted.startswith(p) for p in EXECUTE_PREFIXES):
        return ("execute", dotted)
    if any(dotted.startswith(p) for p in NETWORK_PREFIXES):
        return ("network", dotted)
    if dotted in WRITE_CALLS:
        return ("write", dotted)
    if dotted in READ_CALLS:
        return ("read", dotted)
    if bare in WRITE_METHODS:
        return ("write", dotted or bare)
    if bare in READ_METHODS:
        return ("read", dotted or bare)
    return None


@dataclass
class _FunctionFacts:
    """Everything the taint fixpoint needs about one function, collected in a single pass."""

    name: str
    params: tuple[str, ...] = ()
    #: assigned name -> (names it derives from, whether the right-hand side is a resource read,
    #: the line, and the validator that guarded it if any)
    bindings: dict[str, tuple[frozenset[str], str, int, str]] = field(default_factory=dict)
    returns: list[tuple[frozenset[str], int]] = field(default_factory=list)
    branches: list[tuple[frozenset[str], int]] = field(default_factory=list)
    validated: dict[str, str] = field(default_factory=dict)


def _collect(
    source: str, entrypoint: str
) -> tuple[
    list[ResourceAccess],
    dict[str, _FunctionFacts],
    list[str],
]:
    """One `MetadataWrapper.visit` pass: the inventory, the per-function facts, the unresolved names.

    `wrapper.visit(collector)` rather than `module.visit(collector)`, which is the wrinkle E9
    records and which was re-verified against libcst 1.9.0 before this was written: driving the
    visitor from the module raises `AttributeError: 'collector' object has no attribute 'metadata'`
    because the metadata was never resolved.
    """
    _requires_libcst()
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

    module = cst.parse_module(source)
    constants = _module_constants(module)
    wrapper = MetadataWrapper(module)
    accesses: list[ResourceAccess] = []
    facts: dict[str, _FunctionFacts] = {}
    unresolved: list[str] = []

    class _Scan(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ScopeProvider, PositionProvider)

        def __init__(self) -> None:
            self.stack: list[_FunctionFacts] = []

        # -- structure -----------------------------------------------------

        def visit_FunctionDef(self, node: Any) -> bool:
            f = _FunctionFacts(
                name=node.name.value,
                params=tuple(p.name.value for p in node.params.params),
            )
            facts[f.name] = f
            self.stack.append(f)
            return True

        def leave_FunctionDef(self, node: Any) -> None:
            if self.stack and self.stack[-1].name == node.name.value:
                self.stack.pop()

        @property
        def here(self) -> _FunctionFacts | None:
            return self.stack[-1] if self.stack else None

        def _line(self, node: Any) -> tuple[int, int]:
            try:
                pos = self.get_metadata(PositionProvider, node)
            except KeyError:
                return (0, 0)
            return (pos.start.line, pos.start.column)

        def _in_scope(self, node: Any) -> bool:
            """Whether `ScopeProvider` can place this node. Records the ones it cannot.

            `ScopeProvider` raises `KeyError` on import statements and on attribute names in
            libcst 1.9.0, re-verified before this was written. Catching it for the one lookup that
            provokes it and recording the name is different from swallowing it: the reading says
            which names the analysis could not place, so a reader sees its edge.
            """
            try:
                self.get_metadata(ScopeProvider, node)
                return True
            except KeyError:
                unresolved.append(getattr(node, "value", type(node).__name__))
                return False

        # -- the inventory -------------------------------------------------

        def visit_Call(self, node: Any) -> bool:
            dotted = _dotted(node.func)
            bare = node.func.attr.value if isinstance(node.func, cst.Attribute) else dotted
            kind_and_name = _classify(dotted, bare, list(node.args))
            if kind_and_name is None:
                return True
            kind, call = kind_and_name
            line, col = self._line(node)
            target, is_literal = self._target(node)
            accesses.append(
                ResourceAccess(
                    kind=kind,
                    call=call,
                    target=target,
                    target_is_literal=is_literal,
                    function=self.here.name if self.here else "<module>",
                    line=line,
                    col=col,
                    credential=_looks_credential(target) if is_literal else False,
                )
            )
            return True

        def visit_Subscript(self, node: Any) -> bool:
            """`os.environ["OPENAI_API_KEY"]`, which is a call to nothing and a read of a secret."""
            root = _dotted(node.value)
            if not root.endswith("environ"):
                return True
            key = None
            for element in node.slice:
                key = _literal(getattr(element.slice, "value", None))
            line, col = self._line(node)
            target = key if key is not None else "<expr>"
            accesses.append(
                ResourceAccess(
                    kind="environment",
                    call=f"{root}[]",
                    target=target,
                    target_is_literal=key is not None,
                    function=self.here.name if self.here else "<module>",
                    line=line,
                    col=col,
                    credential=_looks_credential(target) if key is not None else True,
                )
            )
            return True

        def _target(self, node: Any) -> tuple[str, bool]:
            """What the call touches, which is not always its first argument.

            For a method call the receiver is the resource: `report.write("done")` touches
            `report`, and reading the literal `"done"` off it would report the *contents* as the
            target and then test them against the boundary's path patterns, which is nonsense that
            renders convincingly.
            """
            if isinstance(node.func, cst.Attribute):
                bare = node.func.attr.value
                if bare in WRITE_METHODS or bare in READ_METHODS:
                    receiver = node.func.value
                    lit = self._as_literal(receiver)
                    if lit is not None:
                        return lit, True
                    names = _names_in(receiver)
                    if names:
                        return f"<{sorted(names)[0]}>", False
            for arg in node.args:
                lit = self._as_literal(arg.value)
                if lit is not None:
                    return lit, True
            for arg in node.args:
                names = _names_in(arg.value)
                if names:
                    return f"<{sorted(names)[0]}>", False
            if isinstance(node.func, cst.Attribute):
                root = _dotted(node.func)
                return f"<{root.rsplit('.', 1)[0]}>", False
            return "<none>", False

        def _as_literal(self, node: Any) -> str | None:
            """An inline string, or a module-level constant that holds one."""
            lit = _literal(node)
            if lit is not None:
                return lit
            if isinstance(node, cst.Name):
                return constants.get(node.value)
            return None

        # -- the dataflow --------------------------------------------------

        def visit_Assign(self, node: Any) -> bool:
            f = self.here
            if f is None or len(node.targets) != 1:
                return True
            target = node.targets[0].target
            if not isinstance(target, cst.Name):
                return True
            self._in_scope(target)
            line, _ = self._line(node)
            f.bindings[target.value] = (
                frozenset(_names_in(node.value)),
                self._read_kind(node.value),
                line,
                self._validator(node.value),
            )
            return True

        def visit_With(self, node: Any) -> bool:
            f = self.here
            if f is None:
                return True
            for item in node.items:
                asname = getattr(item, "asname", None)
                if asname is None or not isinstance(asname.name, cst.Name):
                    continue
                line, _ = self._line(node)
                f.bindings[asname.name.value] = (
                    frozenset(_names_in(item.item)),
                    self._read_kind(item.item),
                    line,
                    "",
                )
            return True

        def _read_kind(self, value: Any) -> str:
            """Whether this expression is itself a source of untrusted bytes."""
            if not isinstance(value, cst.Call):
                return ""
            dotted = _dotted(value.func)
            bare = value.func.attr.value if isinstance(value.func, cst.Attribute) else dotted
            got = _classify(dotted, bare, list(value.args))
            if got is None:
                return ""
            kind, _ = got
            return kind if kind in {"read", "environment"} else ""

        def _validator(self, value: Any) -> str:
            for name in _names_in(value):
                low = name.lower()
                if any(v in low for v in VALIDATOR_NAMES):
                    return name
            return ""

        # -- the sinks -----------------------------------------------------

        def visit_Return(self, node: Any) -> bool:
            f = self.here
            if f is None or node.value is None:
                return True
            line, _ = self._line(node)
            f.returns.append((frozenset(_names_in(node.value)), line))
            return True

        def visit_If(self, node: Any) -> bool:
            f = self.here
            if f is None:
                return True
            line, _ = self._line(node)
            f.branches.append((frozenset(_names_in(node.test)), line))
            for name in _names_in(node.test):
                low = name.lower()
                if any(v in low for v in VALIDATOR_NAMES):
                    for other in _names_in(node.test):
                        f.validated.setdefault(other, name)
            return True

    scan = _Scan()
    wrapper.visit(scan)
    if entrypoint and entrypoint not in facts:
        raise AttributeError(
            f"the source defines no function named {entrypoint!r}. Set `entrypoint` to the "
            f"scoring function's name; D8 follows untrusted input to *its* decisions, and "
            f"pointing it at the wrong function reports a clean result for the wrong reason."
        )
    return accesses, facts, unresolved


def _taint(
    facts: _FunctionFacts,
    untrusted: Iterable[str],
    trusted_bindings: Iterable[str] = (),
) -> tuple[
    dict[str, tuple[str, str, tuple[str, ...]]],
    list[TaintPath],
]:
    """Forward taint to a fixpoint over one function's bindings, then read off the sinks.

    Sources are the declared untrusted parameters and anything bound to a resource read: a file's
    contents are untrusted whatever the path was, because the file sits inside a sandbox the agent
    is running in.

    ``trusted_bindings`` is the exception and it needs a declared boundary to exist. A harness
    reading its own gold answers from a path it keeps private is not reading untrusted input, and
    counting that read as a taint source turns every grader into a finding. Without a
    `TaskBoundary` nothing is trusted, which over-reports on purpose: the instrument cannot tell
    a private path from a shared one by looking at the harness alone, and guessing in the other
    direction would hide the real thing.

    Sinks are the function's `return` statements and its branch tests. A return is the scoring
    decision itself; a branch is a decision on the way to one, and both are reported because a
    grader that branches on unchecked input and then returns a constant is not safer for it.
    """
    trusted = set(trusted_bindings)
    tainted: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for p in facts.params:
        if p in untrusted:
            tainted[p] = (f"parameter {p}", "parameter", (p,))
    for name, (deps, read_kind, _line, _guard) in facts.bindings.items():
        if read_kind and name not in trusted:
            label = "environment" if read_kind == "environment" else "resource_read"
            tainted[name] = (f"{label} bound to {name}", label, (name,))

    changed = True
    while changed:
        changed = False
        for name, (deps, _read, _line, _guard) in facts.bindings.items():
            if name in tainted or name in trusted:
                continue
            hit = sorted(deps & set(tainted))
            if hit:
                source, kind, chain = tainted[hit[0]]
                tainted[name] = (source, kind, (*chain, name))
                changed = True

    paths: list[TaintPath] = []
    for names, line, sink_kind, sink in (
        *[(n, ln, "return", f"the return of {facts.name}") for n, ln in facts.returns],
        *[(n, ln, "branch", f"a branch in {facts.name}") for n, ln in facts.branches],
    ):
        for name in sorted(names & set(tainted)):
            source, kind, chain = tainted[name]
            guard = facts.validated.get(name, "")
            if not guard:
                for step in chain:
                    binding = facts.bindings.get(step)
                    if binding and binding[3]:
                        guard = binding[3]
                        break
            paths.append(
                TaintPath(
                    source=source,
                    source_kind=kind,  # type: ignore[arg-type]  # the three labels above are the Literal
                    sink=sink,
                    sink_kind=sink_kind,  # type: ignore[arg-type]
                    function=facts.name,
                    line=line,
                    chain=chain,
                    validated=bool(guard),
                    validated_by=guard,
                )
            )
    return tainted, paths


def analyse_environment(
    verifier: VerifierUnderTest,
    *,
    rung: int = 1,
    boundary: TaskBoundary | None = None,
    untrusted_params: Sequence[str] | None = None,
    cheapness: AttackCheapness | None = None,
) -> AttackSurface:
    """D8's static pass over one harness file. Rung 0 is the inventory, rung 1 adds the boundary.

    ``untrusted_params`` defaults to *every* parameter of the entrypoint, which is the assumption
    a harness author should have to argue against rather than opt into: the arguments a scoring
    function receives come from the task and from the agent, and one that receives only trusted
    values is unusual enough to be worth naming.
    """
    source = verifier.source()
    accesses, facts, unresolved = _collect(source, verifier.entrypoint)
    target = facts[verifier.entrypoint]
    untrusted = tuple(untrusted_params) if untrusted_params is not None else target.params

    trusted = _trusted_bindings(target, accesses, boundary)
    tainted, paths = _taint(target, untrusted, trusted)

    # A target computed from a tainted name is a path the agent can steer, which is a stronger
    # finding than reading a fixed path and is easy to miss in a flat inventory.
    marked: list[ResourceAccess] = []
    for a in accesses:
        steered = (
            not a.target_is_literal
            and a.function == target.name
            and a.target.strip("<>") in tainted
        )
        marked.append(replace(a, tainted_target=steered) if steered else a)

    crossings: tuple[BoundaryCrossing, ...] = ()
    notes: list[str] = []
    if rung >= 1:
        if boundary is None:
            notes.append(
                "rung 1 was requested and no TaskBoundary was supplied, so no crossing could be "
                "computed. A crossing is the harness trusting something on the agent's side, and "
                "which side a path is on is a declaration about the environment rather than "
                "something readable from the harness's own source."
            )
        else:
            crossings = tuple(_crossings(marked, boundary))

    return AttackSurface(
        source_path=str(verifier.source_path),
        entrypoint=verifier.entrypoint,
        fingerprint=verifier.fingerprint,
        rung=rung if cheapness is None else 2,
        accesses=tuple(marked),
        taints=tuple(paths),
        crossings=crossings,
        boundary=boundary.name if boundary is not None else "",
        unresolved_names=tuple(unresolved),
        cheapness=cheapness,
        notes=tuple(notes),
    )


def _trusted_bindings(
    facts: _FunctionFacts,
    accesses: Sequence[ResourceAccess],
    boundary: TaskBoundary | None,
) -> set[str]:
    """Names bound to a read of something the boundary declares the harness keeps to itself.

    Two passes to a small fixpoint, because a gold file is usually read in two steps: a handle is
    opened on a private literal path and the contents are parsed out of the handle. Trusting only
    the handle would leave the parsed value tainted and the finding would survive with a longer
    chain, which is the same wrong answer in a better disguise.
    """
    if boundary is None:
        return set()
    private_lines = {
        a.line
        for a in accesses
        if a.kind == "read" and a.target_is_literal and boundary.is_harness_private(a.target)
    }
    trusted = {
        name
        for name, (_deps, read_kind, line, _guard) in facts.bindings.items()
        if read_kind == "read" and line in private_lines
    }
    local = set(facts.bindings) | set(facts.params)
    changed = True
    while changed:
        changed = False
        for name, (deps, read_kind, _line, _guard) in facts.bindings.items():
            if name in trusted or not read_kind:
                continue
            # Only local names carry trust. `json` in `json.load(g)` is an import, not a value
            # with a provenance, and letting it block the check leaves the parsed gold tainted.
            carriers = deps & local
            if carriers and carriers <= trusted:
                trusted.add(name)
                changed = True
    return trusted


def _crossings(
    accesses: Sequence[ResourceAccess], boundary: TaskBoundary
) -> Iterable[BoundaryCrossing]:
    for a in accesses:
        if a.target_is_literal and boundary.is_agent_writable(a.target) and a.kind == "read":
            yield BoundaryCrossing(
                access=a,
                rule="reads inside the agent-writable region",
                why=(
                    "the scoring path depends on bytes the agent may write, so producing the "
                    "bytes is an alternative to solving the task"
                ),
            )
        elif a.target_is_literal and boundary.is_secret(a.target):
            yield BoundaryCrossing(
                access=a,
                rule="reaches a declared secret",
                why="anything running in this process can reach it, including agent-supplied code",
            )
        elif a.tainted_target and a.kind in {"read", "write", "execute"}:
            yield BoundaryCrossing(
                access=a,
                rule="the target is computed from untrusted input",
                why=(
                    "the agent influences which resource is touched, which is a stronger position "
                    "than influencing its contents"
                ),
            )


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

#: Rung 0 and rung 1 read one file's text at one content hash and assert nothing about the run, so
#: no measured regime can invalidate them. Same honest use of `unconditional` D9 makes.
D8_SURFACE_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over the harness's own source at one content hash. It makes no claim about the "
        "run, the corpus or the policy, so no regime condition can make the inventory wrong. What "
        "it can be is incomplete, which is a bias rather than a regime dependency and is declared "
        "as one."
    ),
)

#: The cost ratio is a measurement of two behaviours of the *live* grader, so it needs the grader
#: to be the same program at both ends of the comparison.
D8_CHEAPNESS_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
)

#: The environment's source and configuration. `Access.SOURCE` exists for exactly this: reading a
#: harness's text is neither running it nor modifying it.
ACCESS_ENV_SOURCE: AccessMatrix = {Component.TASK: Access.SOURCE}

#: A measured cost ratio needs the source *and* the ability to run both a solve and an attack.
ACCESS_ENV_SOURCE_AND_QUERY: AccessMatrix = {
    Component.TASK: Access.SOURCE | Access.QUERY,
    Component.GRADER: Access.QUERY,
}

_D8_SUBSTRATES = frozenset({Substrate.PROGRAM, Substrate.PROCEDURAL, Substrate.COMPOSITE})
_D8_PHASES = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})


class AttackSurfaceInventory(BaseObservable):
    """D8 `env.attack_surface`: what the harness reads, writes, runs and trusts.

    Kill condition, from the catalogue: **if attack cheapness does not predict realised cheat rate
    across twenty environments.** Worth being plain that the test has not been run: it needs twenty
    environments with monitor-measured cheat rates beside them, and nothing here establishes that
    the surface predicts anything. Until it does, this is a description of a harness, and the
    number to distrust first is a zero, which usually means the entrypoint name is wrong rather
    than that the harness is sealed.
    """

    name = "AttackSurfaceInventory"
    version = "1.0"
    quantity = "env.attack_surface"
    capabilities = Capability.SCORES
    requires = ACCESS_ENV_SOURCE
    substrates = _D8_SUBSTRATES
    phases = _D8_PHASES
    envelope = D8_SURFACE_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = ("the assumption that the sandbox holds, which predicts a surface of zero",)
    rung = 1
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "static taint analysis over libcst's ScopeProvider def-use edges"
    deviations = (
        "reads the harness's source and never modifies it; declared `Access.SOURCE`, added for "
        "the D series.",
        "the taint analysis is intra-procedural over the entrypoint and follows names, not "
        "values. It does not cross function boundaries, so a harness that launders untrusted "
        "input through a helper reports fewer paths than it has.",
        "without a TaskBoundary every resource read is treated as an untrusted source, including "
        "the harness reading its own gold answers, so the rung-0 taint count is an over-count in "
        "that one direction while the access inventory around it is a floor. Declaring a boundary "
        "at rung 1 is what removes the harness's private reads and makes both counts floors.",
        "validation is the presence of a validator-shaped name on the path, which cannot see "
        "whether the check is correct or whether it dominates the use.",
        "a boundary crossing is computed only for literal targets. A path assembled at run time "
        "is reported as a steered target instead, which is a different and weaker statement.",
    )

    def __init__(
        self,
        verifier: VerifierUnderTest | None = None,
        *,
        rung: int = 1,
        boundary: TaskBoundary | None = None,
        untrusted_params: Sequence[str] | None = None,
        cheapness: AttackCheapness | None = None,
    ) -> None:
        ensure_quantities()
        self.verifier = verifier
        self.rung = rung
        self.boundary = boundary
        self.untrusted_params = untrusted_params
        self.cheapness = cheapness

    @property
    def subject(self) -> ProgramSubject:
        if self.verifier is None:
            raise ValueError(f"{self.name} was constructed without a harness")
        return ProgramSubject(self.verifier)

    def estimate(self, ctx: Context | None = None) -> Reading:
        if ctx is None:
            # `Context.signal` is typed `RewardSignal` and a harness is a program, not a network.
            # Same kernel gap `verifier.program_context` records.
            ctx = Context(
                signal=self.subject,  # type: ignore[arg-type]
                readout="score",
                substrate=Substrate.PROGRAM,
            )
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        verifier = self.verifier or ctx.signal.verifier  # type: ignore[union-attr]  # a ProgramSubject, not a network
        surface = analyse_environment(
            verifier,
            rung=self.rung,
            boundary=self.boundary,
            untrusted_params=self.untrusted_params,
            cheapness=self.cheapness,
        )
        return ctx.emit(
            surface,
            uncertainty=Uncertainty(n=len(surface.accesses), method="static census, a floor"),
            subject_extra={
                **dict(SENSITIVE_SUBJECT_EXTRA),
                "baseline_sandbox_holds": str(surface.baseline_sandbox_holds),
            },
        )


class AttackCheapnessRatio(BaseObservable):
    """D8 `env.attack_cheapness`: how much cheaper attacking the grader is than solving the task.

    Kill condition, from the catalogue: **if attack cheapness does not predict realised cheat rate
    across twenty environments.** This is the quantity that carries it, and the twenty-environment
    test is the one thing that would turn the D8 pair from a description into a forecast.
    """

    name = "AttackCheapnessRatio"
    version = "1.0"
    quantity = "env.attack_cheapness"
    #: `NONE` rather than `SCORES`, unlike every other instrument in this series. The ratio is
    #: computed from two cost samples the caller measured; it asks nothing of a signal, and there
    #: is often no source to hand either, since a marketplace buyer can time an attack against an
    #: environment whose code they will never see. The kernel's own guidance in `measure.base.run`
    #: is that an instrument needing no capability from a signal declares `Capability.NONE` and
    #: reads its subject through the access matrix, which is what `requires` below does.
    capabilities = Capability.NONE
    requires = ACCESS_ENV_SOURCE_AND_QUERY
    substrates = _D8_SUBSTRATES
    phases = _D8_PHASES
    envelope = D8_CHEAPNESS_ENVELOPE
    invariance = "units"
    invariance_relation = Relation("invariant")
    baselines = ("the assumption that the sandbox holds, under which no attack is available",)
    rung = 2
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "the ratio is of measured costs the caller supplies. Nothing here estimates the cost of "
        "an attack from source: a cheapness number inferred from an inventory would be a guess "
        "wearing a unit.",
        "the interval is a percentile bootstrap over the two cost samples, so it covers sampling "
        "spread and not the choice of which attack was tried. A cheaper attack nobody attempted "
        "moves the true ratio and not this interval.",
    )

    def __init__(
        self,
        attack: CostSample | None = None,
        solve: CostSample | None = None,
        *,
        verifier: VerifierUnderTest | None = None,
        ci_level: float = 0.95,
        resamples: int = 2000,
        seed: int = 0,
    ) -> None:
        ensure_quantities()
        self.attack = attack
        self.solve = solve
        self.verifier = verifier
        self.ci_level = ci_level
        self.resamples = resamples
        self.seed = seed

    def estimate(self, ctx: Context | None = None) -> Reading:
        if ctx is None:
            signal = ProgramSubject(self.verifier) if self.verifier is not None else None
            ctx = Context(
                signal=signal,  # type: ignore[arg-type]  # a ProgramSubject, not a network
                readout="score",
                substrate=Substrate.PROGRAM,
            )
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        if self.attack is None or self.solve is None:
            missing = [n for n, v in (("attack", self.attack), ("solve", self.solve)) if v is None]
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=f"no measured cost sample for {' and '.join(missing)}",
                remedy=(
                    "run the task and run an attack, and pass what each cost as "
                    "`CostSample(what=..., unit=..., values=(...))`. This rung is a measurement. "
                    "The static inventory at rung 0 tells you where an attack might be; it cannot "
                    "tell you what one costs, and a ratio inferred from it would be a guess."
                ),
                statistics={"missing": missing},
            )
        result = attack_cheapness(
            self.attack,
            self.solve,
            ci_level=self.ci_level,
            resamples=self.resamples,
            seed=self.seed,
        )
        if isinstance(result, Refusal):
            return replace(result, instrument=self.name)
        return ctx.emit(
            result,
            uncertainty=Uncertainty(
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                ci_level=result.ci_level,
                n=result.n_attack + result.n_solve,
                method="percentile bootstrap on the ratio of means",
            ),
        )


def attack_surface(verifier: VerifierUnderTest, **kwargs: Any) -> Reading:
    """Run D8's inventory and return the Reading. The one-call form, for a card renderer."""
    return AttackSurfaceInventory(verifier, **kwargs).estimate()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """D8's ladder: three estimators for the surface, one for the cost ratio.

    That is what `spec/QUANTITIES.yaml` declares, `rungs: 3` and `rungs: 1`, and it is how the
    catalogue's three-rung ladder splits across two quantities: rungs 0 and 1 are the surface, and
    rung 2, "a measured cost ratio", is `env.attack_cheapness`, which the surface then carries as
    an annotation rather than recomputing.
    """
    ensure_quantities()
    for rung, impl, what in (
        (0, "env.attack_surface.inventory", "the static inventory of reads, writes and secrets"),
        (1, "env.attack_surface.boundary", "adds taint to a scoring decision and the crossings"),
        (2, "env.attack_surface.costed", "adds the measured cost ratio as an annotation"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity="env.attack_surface",
                impl=impl,
                requires=ACCESS_ENV_SOURCE if rung < 2 else ACCESS_ENV_SOURCE_AND_QUERY,
                envelope=D8_SURFACE_ENVELOPE if rung < 2 else D8_CHEAPNESS_ENVELOPE,
                rung=rung,
                bias=BiasStatement(
                    direction="downward",
                    why=(
                        f"{what}. A static pass sees the calls it has patterns for and the flows "
                        f"it can follow inside one function. Anything reached through a helper, "
                        f"through getattr, or through a library this has no pattern for is a "
                        f"surface it does not report, so every count is a floor."
                    ),
                ),
                cost=CostModel(note="one parse of one file; no GPU, no grader call"),
                substrates=_D8_SUBSTRATES,
                phases=_D8_PHASES,
                run=None,
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="env.attack_cheapness",
            impl="env.attack_cheapness.measured_ratio",
            requires=ACCESS_ENV_SOURCE_AND_QUERY,
            envelope=D8_CHEAPNESS_ENVELOPE,
            rung=2,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "the ratio is solve cost over the cost of the cheapest attack anybody "
                    "actually tried. An attack nobody thought of is cheaper than the one measured, "
                    "so the measured cheapness under-states the true cheapness, always."
                ),
            ),
            cost=CostModel(note="one bootstrap over two supplied cost samples"),
            substrates=_D8_SUBSTRATES,
            phases=_D8_PHASES,
            run=None,
        )
    )


_register()


__all__ = [
    "ACCESS_ENV_SOURCE",
    "ACCESS_ENV_SOURCE_AND_QUERY",
    "CREDENTIAL_MARKERS",
    "CREDENTIAL_PATHS",
    "D8_CHEAPNESS_ENVELOPE",
    "D8_SURFACE_ENVELOPE",
    "AttackCheapness",
    "AttackCheapnessRatio",
    "AttackSurface",
    "AttackSurfaceInventory",
    "BoundaryCrossing",
    "CostSample",
    "ResourceAccess",
    "SurfaceKind",
    "TaintPath",
    "TaskBoundary",
    "analyse_environment",
    "attack_cheapness",
    "attack_surface",
]
