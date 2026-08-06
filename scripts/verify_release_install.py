#!/usr/bin/env python3
"""Build the artifacts, install them into a genuinely fresh interpreter, and check they work.

Run before every tag, and again against TestPyPI and against PyPI after every upload:

    python scripts/verify_release_install.py
    python scripts/verify_release_install.py --index-url https://test.pypi.org/simple/
    python scripts/verify_release_install.py --index-url https://pypi.org/simple/

The three checks a release sequence has to have are that ``import reward_lens`` succeeds, that it
does not drag torch in with it, and that the console script runs. This script has those, and it
does not stop there, for a reason with a date on it.

**2.0.0 shipped an install that imported successfully, reported its version, and failed on first
real use.** Three runtime dependencies were imported at module scope and none of them was in the
dependency list. ``pip install reward-lens`` succeeded. ``import reward_lens`` succeeded.
``reward_lens.__version__`` printed. The first thing anyone actually did with it raised
``ModuleNotFoundError``. An install check that stops at ``import`` would have passed that release,
which is why 2.0.1 exists: to carry the correction to PyPI. So every check below past the first
three exercises a path a user takes rather than the import surface:

* the registry loads its 190 quantities from the JSON the wheel carries, not from a checkout;
* the `capabilities` command renders against a real record on disk, not just ``--help``;
* a missing extra raises a typed error naming an extra ``pip`` can actually install;
* every declared runtime dependency imports and resolves inside the new environment;
* nothing compiled and no YAML reached the wheel, and ``py.typed`` did.

Nothing here needs the repository to be importable and nothing runs against the working tree. The
package under test is whatever landed in the fresh environment, which is the only thing a user
will ever have.

Exit status is 0 only if every check passes. The report goes to stdout and, unless ``--report -``
is given, to ``dist/install-verification.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# What the installed package must measure. These are assertions against numbers rather than
# against non-emptiness: a registry that loads two rows is not "loaded", and a refusal enum that
# has lost a member is a silently narrower contract. spec/CATALOGUE.yaml is the human-editable
# source and the wheel ships only the generated JSON, so these are read from the package.
EXPECT_QUANTITIES = 190
EXPECT_WEDGE = 112
EXPECT_REFUSAL_REASONS = 17

# Distribution names that would mean a compiled dependency reached the base install. This is the
# list .github/workflows/tests.yml greps for in its base-install job, plus PyYAML.
#
# PyYAML is here rather than in the torch group because it is the reason the catalogue exists
# twice. The YAML under spec/ is the source a human edits and it carries the comments saying where
# each row came from; the JSON under src/reward_lens/spec/ is generated from it and is what the
# wheel ships. PyYAML has a compiled extension, so a registry that imported yaml at load time
# would put a C extension in the base install's dependency closure for the sake of reading one
# file at startup.
#
# numpy, scipy, pandas and scikit-learn are compiled and are deliberately not on this list. They
# ship wheels for every version in the support matrix, so they cost an installing user nothing but
# bytes. "No compiled dependency" in this project has always meant the torch family.
FORBIDDEN_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "triton",
    "pyyaml",
)
FORBIDDEN_DIST_PREFIXES = ("nvidia-",)

# Suffixes that must not appear anywhere inside the installed package directory.
FORBIDDEN_SUFFIXES = (".so", ".pyd", ".dylib", ".yaml", ".yml", ".pyx", ".c", ".h")

# The record the `capabilities` command is rendered against. Twelve steps of a real GRPOTrainer
# run: real weights, real sampling, real advantages, one abstention every seventh completion.
DEFAULT_RECORD = REPO / "tests" / "fixtures" / "grpo_run" / "short"


@dataclass
class Check:
    """One assertion, its outcome, and the numbers it measured."""

    name: str
    ok: bool
    detail: str
    facts: list[str] = field(default_factory=list)


@dataclass
class Runner:
    """A fresh virtual environment and the ways of asking it questions."""

    root: Path
    cwd: Path

    @property
    def python(self) -> Path:
        return self.root / ("Scripts" if os.name == "nt" else "bin") / "python"

    def script(self, name: str) -> Path:
        exe = name + (".exe" if os.name == "nt" else "")
        return self.root / ("Scripts" if os.name == "nt" else "bin") / exe

    @property
    def env(self) -> dict[str, str]:
        """The parent environment with everything that could leak a package out of it removed."""
        e = dict(os.environ)
        for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
            e.pop(var, None)
        e["PYTHONNOUSERSITE"] = "1"
        e["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        # Fixed width and no terminal, so the rendered CLI output is comparable between runs.
        e["COLUMNS"] = "100"
        e["TERM"] = "dumb"
        return e

    def run(self, argv: list, timeout: int = 600) -> subprocess.CompletedProcess:
        """Run a command in the fresh environment, from a neutral directory.

        The working directory matters. Run this from the repository root and ``''`` on
        ``sys.path`` would let a source tree answer for the installed package. The layout here is
        ``src/``, so the root does not in fact shadow ``reward_lens`` today, but that is a
        property of the layout rather than of the check, and a check that leans on it is not
        checking the thing it says it checks.
        """
        return subprocess.run(
            [str(a) for a in argv],
            capture_output=True,
            text=True,
            env=self.env,
            cwd=str(self.cwd),
            timeout=timeout,
        )

    def probe(self, source: str, timeout: int = 300) -> tuple:
        """Run a program in a fresh interpreter and read one JSON object off the last line.

        A separate process per probe, deliberately. ``sys.modules`` is what several of these
        assertions are about, and it cannot be observed honestly in a process that has already
        imported half the package to answer an earlier question.
        """
        r = self.run([self.python, "-c", textwrap.dedent(source)], timeout=timeout)
        if r.returncode != 0:
            return False, {}, (r.stdout + r.stderr).strip()[-2000:]
        lines = r.stdout.strip().splitlines()
        try:
            return True, json.loads(lines[-1] if lines else ""), r.stdout.strip()
        except (json.JSONDecodeError, IndexError) as exc:
            return False, {}, f"probe emitted no JSON ({exc}): {r.stdout.strip()[:400]}"


def sh(argv: list, cwd=None, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )


def declared_version() -> str:
    """The version in pyproject.toml, read without tomllib so this also runs on the 3.10 floor."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("pyproject.toml has no version line")
    return m.group(1)


def declared_base_dependencies() -> list:
    """The names in ``[project] dependencies``, and the module each one installs.

    Read with a regular expression rather than tomllib for the same reason as the version: this
    script has to run on the interpreter the maintainer happens to have, and tomllib is 3.11.
    """
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    if not block:
        raise SystemExit("pyproject.toml has no [project] dependencies array")
    names = re.findall(r'"([A-Za-z0-9_.\-]+)[^"]*"', block.group(1))
    special = {"scikit-learn": "sklearn", "pydantic-settings": "pydantic_settings"}
    return [(n, special.get(n, n.replace("-", "_"))) for n in names]


def build_artifacts(out: Path, log: list) -> tuple:
    """Build the wheel and the sdist and return both paths.

    Prefers the invoking interpreter's ``build``. If it has not got one, a throwaway environment
    gets it, rather than installing anything at all into the environment under test.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    builder = sys.executable
    if sh([builder, "-m", "build", "--version"], timeout=120).returncode != 0:
        log.append("the invoking interpreter has no `build`; making a throwaway one")
        bdir = out.parent / ".verify-builder-venv"
        if bdir.exists():
            shutil.rmtree(bdir)
        venv.EnvBuilder(with_pip=True, clear=True).create(bdir)
        bpy = bdir / ("Scripts" if os.name == "nt" else "bin") / "python"
        got = sh([bpy, "-m", "pip", "install", "--quiet", "--upgrade", "build"], timeout=1200)
        if got.returncode != 0:
            raise SystemExit("could not install `build`:\n" + got.stdout + got.stderr)
        builder = str(bpy)

    r = sh([builder, "-m", "build", "--outdir", str(out), str(REPO)], timeout=2400)
    if r.returncode != 0:
        raise SystemExit("build failed:\n" + r.stdout[-4000:] + r.stderr[-4000:])
    log.append(f"built with {builder}")

    wheels = sorted(out.glob("*.whl"))
    sdists = sorted(out.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"expected one wheel and one sdist in {out}, found {wheels} {sdists}")
    return wheels[0], sdists[0]


def make_environment(root: Path) -> Runner:
    """A genuinely fresh environment: no system site packages, no user site, its own pip."""
    if root.exists():
        shutil.rmtree(root)
    venv.EnvBuilder(
        system_site_packages=False, clear=True, with_pip=True, upgrade_deps=False
    ).create(root)
    neutral = root.parent / "cwd"
    neutral.mkdir(exist_ok=True)
    return Runner(root=root, cwd=neutral)


def install(run: Runner, args, wheel, version: str, log: list) -> str:
    """Put the package into the fresh environment and say where it came from.

    Two sources, because the same script has to verify the local artifacts before the tag, the
    TestPyPI upload after the dispatch, and the PyPI upload after the tag. Whether the wheel that
    reaches a user is the one that was built is exactly the thing an index upload can get wrong.

    The version is always pinned with ``==``, never left to the resolver, and that is not
    tidiness. Two things go wrong without it, and both were seen here rather than reasoned about.
    A run against TestPyPI needs ``--extra-index-url`` for the dependencies TestPyPI does not
    carry, and pip will then happily take the *released* package off PyPI instead of the upload
    under test. And every release candidate is a pre-release, which pip will not install from a
    bare requirement at all: the specifier has to name it. Left unpinned, this script installed
    2.0.1 from PyPI and reported, correctly and uselessly, that 2.0.1 has no `capabilities`
    command and ships no ``py.typed``.
    """
    pip = [run.python, "-m", "pip", "install", "--no-cache-dir"]
    if args.index_url:
        spec = f"reward-lens=={version}"
        cmd = pip + ["--index-url", args.index_url]
        for extra in args.extra_index_url:
            cmd += ["--extra-index-url", extra]
        cmd.append(spec)
        source = f"{args.index_url} ({spec})"
    else:
        cmd = pip + [str(wheel)]
        source = str(wheel)

    r = sh(cmd, timeout=2400)
    if r.returncode != 0:
        raise SystemExit("install failed:\n" + (r.stdout + r.stderr)[-6000:])
    log.append("pip install: " + " ".join(str(c) for c in cmd[3:]))
    return source


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def check_import(run: Runner, expected_version: str) -> Check:
    """1 of 3. The package imports, and it is the version and the copy we think it is."""
    ok, data, raw = run.probe(
        """
        import json, reward_lens
        print(json.dumps({"version": reward_lens.__version__, "file": reward_lens.__file__}))
        """
    )
    if not ok:
        return Check("import reward_lens", False, raw)
    inside = str(run.root) in data["file"]
    good = data["version"] == expected_version and inside
    detail = f"version {data['version']}, from {data['file']}"
    if data["version"] != expected_version:
        detail += (
            f"  <-- expected {expected_version}. Every check below is now being run against the"
            f" wrong package."
        )
    if not inside:
        detail += "  <-- NOT inside the fresh environment"
    return Check("import reward_lens", good, detail, [f"__version__ = {data['version']}"])


def check_no_torch(run: Runner) -> Check:
    """2 of 3. Importing the package does not import torch.

    Read out of ``sys.modules`` in a subprocess that has done nothing else, because the only
    honest place to ask this is a process where the answer could still be no. The same probe
    watches transformers, accelerate and yaml: the point is not torch specifically, it is that a
    base install stays a base install.
    """
    ok, data, raw = run.probe(
        """
        import json, sys
        import reward_lens
        watched = ("torch", "transformers", "accelerate", "yaml", "pyarrow", "matplotlib", "trl")
        print(json.dumps({
            "leaked": [m for m in watched if m in sys.modules],
            "modules": len(sys.modules),
        }))
        """
    )
    if not ok:
        return Check("import pulls nothing heavy", False, raw)
    leaked = data["leaked"]
    return Check(
        "import pulls nothing heavy",
        not leaked,
        (
            f"none of torch, transformers, accelerate, yaml, pyarrow, matplotlib or trl in "
            f"sys.modules; {data['modules']} modules loaded"
        )
        if not leaked
        else f"import dragged in {', '.join(leaked)}",
        [f"sys.modules after import: {data['modules']}"],
    )


def check_capabilities_help(run: Runner) -> Check:
    """3 of 3. The console script exists on PATH and answers.

    This is the check 2.0.0 needed and did not have. The entry point resolving is a different
    fact from the package importing: it goes through the distribution's metadata, and it is where
    a missing module-scope dependency surfaces first.
    """
    exe = run.script("reward-lens")
    if not exe.exists():
        return Check("reward-lens capabilities --help", False, f"no console script at {exe}")
    r = run.run([exe, "capabilities", "--help"], timeout=180)
    good = r.returncode == 0 and "Usage" in r.stdout
    return Check(
        "reward-lens capabilities --help",
        good,
        f"exit {r.returncode}, {len(r.stdout)} bytes of help"
        if good
        else (r.stdout + r.stderr)[-1500:],
    )


def check_registry(run: Runner) -> Check:
    """The registry loads, from the JSON the wheel carries, with the counts the catalogue has.

    Against the numbers rather than against non-emptiness. A registry that loaded two rows would
    pass a truthiness check and then refuse every instrument in the library with
    QUANTITY_UNDEFINED, which is a correct refusal for an incorrect reason.
    """
    ok, data, raw = run.probe(
        """
        import json
        from reward_lens.core.quantity import QUANTITIES, load_quantities
        report = load_quantities()
        print(json.dumps({
            "total": len(QUANTITIES),
            "wedge": sum(1 for q in QUANTITIES.values() if q.wedge),
            "trivial": len(report.trivial_group),
            "skipped_open": report.skipped_open,
            "source": report.source,
        }))
        """
    )
    if not ok:
        return Check("quantity registry", False, raw)
    from_wheel = str(run.root) in data["source"] and data["source"].endswith(".json")
    good = (
        data["total"] == EXPECT_QUANTITIES
        and data["wedge"] == EXPECT_WEDGE
        and not data["skipped_open"]
        and from_wheel
    )
    problems = []
    if data["total"] != EXPECT_QUANTITIES:
        problems.append(f"{data['total']} quantities, expected {EXPECT_QUANTITIES}")
    if data["wedge"] != EXPECT_WEDGE:
        problems.append(f"{data['wedge']} in the wedge, expected {EXPECT_WEDGE}")
    if data["skipped_open"]:
        problems.append(f"skipped as OPEN: {data['skipped_open']}")
    if not from_wheel:
        problems.append(f"loaded from {data['source']}, which is not the installed JSON")
    return Check(
        "quantity registry",
        good,
        "; ".join(problems)
        if problems
        else f"{data['total']} quantities, {data['wedge']} in the wedge, from the packaged JSON",
        [
            f"quantities: {data['total']}",
            f"wedge: {data['wedge']}",
            f"trivial invariance group: {data['trivial']}",
            f"catalogue source: {data['source']}",
        ],
    )


def check_refusal_reasons(run: Runner) -> Check:
    """Seventeen refusal reasons, by name.

    Refusal is a return value here, not an exception, so the enum is part of the public contract:
    a caller switches on it. Losing a member narrows what the library is able to say without
    narrowing anything a type checker or a test would notice.
    """
    ok, data, raw = run.probe(
        """
        import json
        from reward_lens.core.reading import RefusalReason
        print(json.dumps({"names": [m.name for m in RefusalReason]}))
        """
    )
    if not ok:
        return Check("RefusalReason members", False, raw)
    n = len(data["names"])
    return Check(
        "RefusalReason members",
        n == EXPECT_REFUSAL_REASONS,
        f"{n} members"
        + ("" if n == EXPECT_REFUSAL_REASONS else f", expected {EXPECT_REFUSAL_REASONS}"),
        [f"reasons: {', '.join(data['names'])}"],
    )


def check_nothing_compiled(run: Runner) -> Check:
    """No compiled dependency and no YAML in the closure, and none of either in the package.

    Two questions with one answer. The distribution list is what CI greps for, so the same defect
    fails in the same words in both places. The file scan is the stricter half: a wheel can carry
    a ``.so`` that no distribution name reveals, and a ``.yaml`` in the package would mean the
    registry had quietly gone back to needing PyYAML at load time.
    """
    ok, data, raw = run.probe(
        """
        import json, pathlib, sys
        import reward_lens
        from importlib.metadata import distributions
        pkg = pathlib.Path(reward_lens.__file__).parent
        installed = sorted({(d.metadata["Name"] or "").lower() for d in distributions()})
        suspects = sorted(
            str(p.relative_to(pkg))
            for p in pkg.rglob("*")
            if p.suffix.lower() in %(suffixes)r
        )
        print(json.dumps({
            "installed": [d for d in installed if d],
            "suspects": suspects,
            "package_dir": str(pkg),
        }))
        """
        % {"suffixes": FORBIDDEN_SUFFIXES}
    )
    if not ok:
        return Check("nothing compiled, no YAML", False, raw)
    bad = [
        d
        for d in data["installed"]
        if d in FORBIDDEN_DISTRIBUTIONS or d.startswith(FORBIDDEN_DIST_PREFIXES)
    ]
    problems = []
    if bad:
        problems.append("compiled or YAML distribution in the closure: " + ", ".join(bad))
    if data["suspects"]:
        problems.append("forbidden file in the package: " + ", ".join(data["suspects"]))
    return Check(
        "nothing compiled, no YAML",
        not problems,
        "; ".join(problems)
        if problems
        else f"{len(data['installed'])} distributions installed, none of them "
        f"{', '.join(FORBIDDEN_DISTRIBUTIONS)} or nvidia-*; no "
        f"{'/'.join(FORBIDDEN_SUFFIXES)} file under {data['package_dir']}",
        [f"distributions installed: {len(data['installed'])}"],
    )


def check_py_typed(run: Runner) -> Check:
    """``py.typed`` shipped, and the spec JSON with it.

    PEP 561 says a checker treats a package with no marker as untyped, and this project's
    ``ignore_missing_imports`` then resolves every name imported from it to ``Any``.
    Under that resolution ``Blind[T]`` is a name that means nothing, the eight-error leakage
    fixture reports one unrelated error and none of the eight, and the barrier that exists to stop
    a label reaching a scorer is enforced for the maintainer and vacuous for every user of the
    wheel. It is an empty file and it is the whole enforcement.
    """
    ok, data, raw = run.probe(
        """
        import json, pathlib
        import reward_lens
        pkg = pathlib.Path(reward_lens.__file__).parent
        print(json.dumps({
            "py_typed": (pkg / "py.typed").is_file(),
            "spec": sorted(p.name for p in (pkg / "spec").glob("*.json")),
        }))
        """
    )
    if not ok:
        return Check("py.typed and the spec JSON", False, raw)
    want = ["CATALOGUE.json", "QUANTITIES.json"]
    good = data["py_typed"] and data["spec"] == want
    return Check(
        "py.typed and the spec JSON",
        good,
        f"py.typed present, spec/{{{', '.join(data['spec'])}}} present"
        if good
        else f"py.typed={data['py_typed']}, spec JSON={data['spec']}, expected {want}",
    )


def check_capabilities_render(run: Runner, record: Path) -> Check:
    """The command renders against a real record, which is a different fact from ``--help``.

    ``--help`` proves the entry point resolves. This proves the code behind it runs: it reads a
    record written by a real GRPOTrainer, resolves the four access dimensions, and prints both
    halves of the report, the reachable quantities and the refusals carrying remedies. A refusal
    here is a success, so what is asserted is that the report has a refusal section at all, not
    that anything was available.
    """
    if not record.is_dir():
        return Check("capabilities against a real record", False, f"no record at {record}")
    exe = run.script("reward-lens")
    r = run.run([exe, "capabilities", "--record", str(record)], timeout=300)
    text = r.stdout
    wanted = ("ACCESS RESOLVED", "REGIME MEASURED", "REFUSED, WITH REMEDY")
    missing = [w for w in wanted if w not in text]

    j = run.run([exe, "capabilities", "--record", str(record), "--json"], timeout=300)
    try:
        payload = json.loads(j.stdout)
        json_ok = isinstance(payload, dict) and "access" in payload
    except json.JSONDecodeError:
        payload, json_ok = {}, False

    good = r.returncode == 0 and not missing and j.returncode == 0 and json_ok
    if good:
        refused = len(payload.get("refused", []) or [])
        available = len(payload.get("available", []) or [])
        detail = (
            f"rendered {len(text)} bytes over {len(text.splitlines())} lines; "
            f"--json parses with {available} available and {refused} refused"
        )
        facts = [f"capabilities on {record.name}: {available} available, {refused} refused"]
    else:
        detail = (
            f"exit {r.returncode}/{j.returncode}, missing sections {missing}, json_ok={json_ok}"
        )
        detail += "\n" + (r.stdout + r.stderr)[-1200:]
        facts = []
    return Check("capabilities against a real record", good, detail, facts)


def check_extras(run: Runner) -> Check:
    """A missing extra raises a typed error naming an extra ``pip`` can install.

    The failure this replaces was not an unhelpful message, it was a dead end with a helpful tone.
    ``loops/integrations/base.py`` used to raise "Install reward-lens[trl]" for three frameworks
    and none of the three was declared in ``pyproject.toml``, so the instruction the error gave
    could not be followed. Doing what the error says has to be a thing that works, so this checks
    both halves in the installed environment: that the error is typed and names an extra, and that
    every extra any error can name is in the distribution's own ``Provides-Extra``, which is where
    ``pip`` looks.
    """
    ok, data, raw = run.probe(
        """
        import json
        from importlib.metadata import metadata
        from reward_lens.core.extras import EXTRA_PROBE, EXTRA_PURPOSE, ExtraRequiredError
        from reward_lens.core.extras import require_extra

        out = {"declared": sorted(metadata("reward-lens").get_all("Provides-Extra") or [])}
        out["named"] = sorted(EXTRA_PROBE)
        out["purposed"] = sorted(EXTRA_PURPOSE)
        try:
            require_extra("sampling", subsystem="reward_lens.policy.vllm")
            out["typed_error"] = None
        except ExtraRequiredError as exc:
            out["typed_error"] = str(exc)
            out["is_import_error"] = isinstance(exc, ImportError)
        try:
            require_extra("gpu", subsystem="somewhere")
            out["unknown_extra"] = "no error"
        except KeyError:
            out["unknown_extra"] = "KeyError"
        except Exception as exc:
            out["unknown_extra"] = type(exc).__name__
        print(json.dumps(out))
        """
    )
    if not ok:
        return Check("extras name something installable", False, raw)
    undeclared = sorted(set(data["named"]) - set(data["declared"]))
    msg = data.get("typed_error") or ""
    problems = []
    if undeclared:
        problems.append(f"named by the code and not installable: {undeclared}")
    if "reward-lens[sampling]" not in msg:
        problems.append("the typed error does not name a pip-installable extra")
    if not data.get("is_import_error"):
        problems.append(
            "ExtraRequiredError is not an ImportError, so `except ImportError` misses it"
        )
    if data.get("unknown_extra") != "KeyError":
        problems.append(f"an unknown extra gave {data.get('unknown_extra')} rather than a KeyError")
    if set(data["purposed"]) != set(data["named"]):
        problems.append("an extra has no stated purpose")
    return Check(
        "extras name something installable",
        not problems,
        "; ".join(problems)
        if problems
        else f"{len(data['named'])} extras named in code, all of the "
        f"{len(data['declared'])} declared; the error names reward-lens[sampling] and is an "
        f"ImportError",
        [f"Provides-Extra: {', '.join(data['declared'])}"],
    )


def check_dependency_identity(run: Runner) -> Check:
    """Every declared dependency imports, and each one's ``__file__`` is in this environment.

    Asserting the path rather than the name, because a name on PyPI is not an identity. Two of
    this project's own candidate dependencies make the point: ``confseq`` has been stale on PyPI
    since 2023-01-26 and cannot be installed here at all, since there is no cp312 wheel and the
    sdist's CMake dies on ``Boost_INCLUDE_DIR-NOTFOUND``, which is why ``monitor/_vendor/cif.py``
    is vendored rather than depended on; and ``rewardspy`` returns 404 on PyPI and exists only as a
    GitHub repository, so an install that appeared to satisfy it would have satisfied it with
    something else. An import that succeeds from a stale copy elsewhere on the path is the same
    class of false pass, and the path is what distinguishes them.
    """
    deps = declared_base_dependencies()
    ok, data, raw = run.probe(
        """
        import importlib, json
        from importlib.metadata import version, PackageNotFoundError
        rows = []
        for dist, module in %(deps)r:
            row = {"dist": dist, "module": module}
            try:
                m = importlib.import_module(module)
                row["file"] = getattr(m, "__file__", None) or "(namespace package)"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            try:
                row["version"] = version(dist)
            except PackageNotFoundError:
                row["version"] = None
            rows.append(row)
        print(json.dumps({"rows": rows}))
        """
        % {"deps": deps}
    )
    if not ok:
        return Check("declared dependencies resolve here", False, raw)
    problems, facts = [], []
    for row in data["rows"]:
        if "error" in row:
            problems.append(f"{row['dist']}: {row['error']}")
            continue
        if str(run.root) not in row["file"]:
            problems.append(f"{row['dist']} imported from outside the environment: {row['file']}")
            continue
        if row["version"] is None:
            problems.append(f"{row['dist']} imports but has no installed distribution metadata")
            continue
        facts.append(f"{row['dist']} {row['version']}")
    return Check(
        "declared dependencies resolve here",
        not problems,
        "; ".join(problems)
        if problems
        else f"all {len(deps)} import, each from inside the environment: " + ", ".join(facts),
        facts,
    )


def check_console_scripts(run: Runner) -> Check:
    """Both entry points are on PATH and both answer ``--help``."""
    results, problems = [], []
    for name in ("reward-lens", "reward-lens-claims"):
        exe = run.script(name)
        if not exe.exists():
            problems.append(f"{name} is not installed")
            continue
        r = run.run([exe, "--help"], timeout=180)
        if r.returncode != 0:
            problems.append(f"{name} --help exited {r.returncode}: {(r.stdout + r.stderr)[-400:]}")
        else:
            results.append(f"{name} ({len(r.stdout)} bytes)")
    return Check(
        "console scripts",
        not problems,
        "; ".join(problems) if problems else "both answer --help: " + ", ".join(results),
    )


def check_metadata(run: Runner, expected_version: str) -> Check:
    """The metadata a user and an index both read: URLs, long description, Python floor.

    The long description is the project page on PyPI. It is worth asserting non-empty because a
    ``readme`` that fails to resolve is not a build error, it is a blank project page, and the
    place that shows up is after the upload.
    """
    ok, data, raw = run.probe(
        """
        import json
        from importlib.metadata import metadata
        md = metadata("reward-lens")
        body = md.get_payload() if hasattr(md, "get_payload") else ""
        print(json.dumps({
            "version": md["Version"],
            "requires_python": md["Requires-Python"],
            "summary": md["Summary"],
            "content_type": md["Description-Content-Type"],
            "urls": md.get_all("Project-URL") or [],
            "classifiers": md.get_all("Classifier") or [],
            "description_len": len(md["Description"] or body or ""),
        }))
        """
    )
    if not ok:
        return Check("distribution metadata", False, raw)
    problems = []
    if data["version"] != expected_version:
        problems.append(f"version {data['version']}, expected {expected_version}")
    if not data["requires_python"]:
        problems.append("no Requires-Python")
    if (data["content_type"] or "").split(";")[0].strip() != "text/markdown":
        problems.append(f"Description-Content-Type is {data['content_type']!r}")
    if data["description_len"] < 500:
        problems.append(
            f"long description is {data['description_len']} bytes, which is a blank page"
        )
    stale = [u for u in data["urls"] if "readthedocs" in u.lower()]
    if stale:
        problems.append(f"a URL still points at readthedocs: {stale}")
    for u in data["urls"]:
        if "github.com/" in u and "github.com/reward-lens/" not in u:
            problems.append(f"a GitHub URL points somewhere other than the project org: {u}")
    return Check(
        "distribution metadata",
        not problems,
        "; ".join(problems)
        if problems
        else f"{data['version']}, Requires-Python {data['requires_python']}, "
        f"{data['description_len']} bytes of markdown long description, "
        f"{len(data['classifiers'])} classifiers, {len(data['urls'])} project URLs",
        [f"Requires-Python: {data['requires_python']}", f"Summary: {data['summary']}"]
        + [f"Project-URL: {u}" for u in data["urls"]],
    )


def check_sdist(sdist: Path) -> Check:
    """The sdist can be unpacked and carries what a rebuild from it needs.

    An sdist is not a courtesy. It is what anyone building for a platform without a wheel gets,
    and what several distribution packagers use exclusively. A wheel that is correct and an sdist
    that is missing the generated JSON would build a package with an empty registry.
    """
    try:
        with tarfile.open(sdist) as tf:
            names = tf.getnames()
    except Exception as exc:
        return Check("sdist contents", False, f"cannot open {sdist.name}: {exc}")
    root = sdist.name[: -len(".tar.gz")]
    required = [
        f"{root}/pyproject.toml",
        f"{root}/LICENSE",
        f"{root}/README.md",
        f"{root}/src/reward_lens/py.typed",
        f"{root}/src/reward_lens/spec/QUANTITIES.json",
        f"{root}/src/reward_lens/spec/CATALOGUE.json",
    ]
    missing = [r for r in required if r not in names]
    return Check(
        "sdist contents",
        not missing,
        f"missing from the sdist: {missing}"
        if missing
        else f"{len(names)} entries, with pyproject, LICENSE, README, py.typed and both spec JSON",
        [f"sdist entries: {len(names)}"],
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def render(checks: list, meta: dict, log: list) -> str:
    width = max(len(c.name) for c in checks) + 2
    lines = [
        "reward-lens release install verification",
        "=" * 60,
        "",
    ]
    for k, v in meta.items():
        lines.append(f"{k + ':':22} {v}")
    lines.append("")
    lines.append("CHECKS")
    lines.append("-" * 60)
    for c in checks:
        lines.append(f"  {'PASS' if c.ok else 'FAIL'}  {c.name.ljust(width)}  {c.detail}")
    facts = [f for c in checks for f in c.facts]
    if facts:
        lines += ["", "MEASURED", "-" * 60]
        lines += [f"  {f}" for f in facts]
    if log:
        lines += ["", "HOW", "-" * 60]
        lines += [f"  {entry}" for entry in log]
    failed = [c for c in checks if not c.ok]
    lines += ["", "-" * 60]
    lines.append(
        f"{len(checks) - len(failed)}/{len(checks)} passed"
        + ("" if not failed else "; FAILED: " + ", ".join(c.name for c in failed))
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Before the tag:
              python scripts/verify_release_install.py

            After the TestPyPI dispatch (--extra-index-url is not optional there: TestPyPI does
            not carry numpy, scipy, pandas or scikit-learn, and without it the install fails for a
            reason that has nothing to do with the release):
              python scripts/verify_release_install.py --index-url https://test.pypi.org/simple/

            After the tag, against the real index:
              python scripts/verify_release_install.py --index-url https://pypi.org/simple/
            """
        ),
    )
    p.add_argument(
        "--index-url",
        help="Install from this index instead of the local dist/. Implies no build.",
    )
    p.add_argument(
        "--extra-index-url",
        action="append",
        default=[],
        help="Additional index for dependencies. Defaults to PyPI when --index-url is TestPyPI.",
    )
    p.add_argument(
        "--version",
        help="Version to require from the index. Defaults to the one in pyproject.toml.",
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="Use the artifacts already in dist/ rather than rebuilding them.",
    )
    p.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD,
        help="Run record to render `capabilities` against.",
    )
    p.add_argument(
        "--report",
        default=str(REPO / "dist" / "install-verification.txt"),
        help="Where to write the report. `-` writes to stdout only.",
    )
    p.add_argument("--keep", action="store_true", help="Keep the environment for inspection.")
    args = p.parse_args(argv)

    version = args.version or declared_version()
    if args.index_url and not args.extra_index_url and "test.pypi.org" in args.index_url:
        args.extra_index_url = ["https://pypi.org/simple/"]

    log: list = []
    started = time.time()
    dist = REPO / "dist"
    wheel = sdist = None

    if args.index_url:
        log.append(f"no build: installing from {args.index_url}")
    elif args.no_build:
        wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
        if not wheels or not sdists:
            raise SystemExit(f"--no-build but {dist} has no wheel and sdist pair")
        wheel, sdist = wheels[-1], sdists[-1]
        log.append(f"reusing {wheel.name} and {sdist.name}")
    else:
        wheel, sdist = build_artifacts(dist, log)

    workspace = Path(tempfile.mkdtemp(prefix="reward-lens-verify-"))
    try:
        run = make_environment(workspace / "venv")
        base = run.probe('import json, sys; print(json.dumps({"v": sys.version.split()[0]}))')
        log.append(f"fresh environment at {run.root} on Python {base[1].get('v', '?')}")
        source = install(run, args, wheel, version, log)

        checks = [
            check_import(run, version),
            check_no_torch(run),
            check_capabilities_help(run),
            check_registry(run),
            check_refusal_reasons(run),
            check_nothing_compiled(run),
            check_py_typed(run),
            check_capabilities_render(run, args.record.resolve()),
            check_extras(run),
            check_dependency_identity(run),
            check_console_scripts(run),
            check_metadata(run, version),
        ]
        if sdist is not None:
            checks.append(check_sdist(sdist))

        meta = {
            "version under test": version,
            "installed from": source,
            "environment": str(run.root),
            "interpreter": f"{base[1].get('v', '?')} ({platform.platform()})",
            "record": str(args.record),
            "elapsed": f"{time.time() - started:.1f}s",
        }
        text = render(checks, meta, log)
    finally:
        if args.keep:
            print(f"\n(environment kept at {workspace})", file=sys.stderr)
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    print(text)
    if args.report != "-":
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"report written to {out}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
