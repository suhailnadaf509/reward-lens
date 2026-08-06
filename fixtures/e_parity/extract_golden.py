"""Extract the E-parity golden numbers from the v1 campaign result CSVs.

The v1 campaign's result CSVs (under ``reward-lens/outputs/v2_20260506_222648_unknown/``) are the
trust anchor for v3: v3 must reproduce v1's verified-clean headline numbers from the cached
activations before it is trusted to produce new ones, and must produce the honest correction of
v1's known-bad numbers. This script reads those CSVs and writes ``golden.json``, the structured
target the E-parity test suite asserts against.

It reads only the CSVs, so it needs nothing but pandas. Recomputing these numbers from the cached
activations, which is the actual E-parity test, needs the runtime and the battery. This module's
job is to state precisely what the targets are.

Run: ``python fixtures/e_parity/extract_golden.py`` from the repository root. The campaign outputs
are not in this repository. By default they are looked for beside it, in the layout the run used;
point ``REWARD_LENS_V1_OUTPUTS`` at wherever you keep them instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_OUTPUTS = (
    Path(__file__).resolve().parents[4] / "reward-lens" / "outputs" / "v2_20260506_222648_unknown"
)
OUTPUTS = Path(os.environ.get("REWARD_LENS_V1_OUTPUTS", str(_DEFAULT_OUTPUTS)))

# The e15..e20 directories carry a long machine-generated prefix; resolve by suffix.
_PREFIX = "final-reward_reward-lens_outputs_v2_20260506_222648_unknown_"


def _dir(name: str) -> Path:
    direct = OUTPUTS / name
    if direct.exists():
        return direct
    prefixed = OUTPUTS / (_PREFIX + name)
    if prefixed.exists():
        return prefixed
    raise FileNotFoundError(f"neither {direct} nor {prefixed} exists")


def extract() -> dict:
    # The run label, not the path it happened to sit at. Whoever regenerates this has the campaign
    # somewhere else, and a golden file that records their home directory is a golden file that
    # produces a diff for everyone who is not them.
    golden: dict = {
        "source": OUTPUTS.name,
        "note": "E-parity targets: the v1 headline numbers v3 must reproduce from cached activations",
    }

    # E04 faithfulness: attribution-vs-patching rank correlation per (model, dimension). The
    # headline is the per-model mean across dimensions; the design names -0.171 / -0.203 /
    # -0.051 / +0.047 for the four campaign models.
    e04 = pd.read_csv(_dir("e04_faithfulness_population") / "e04_faithfulness.csv")
    per_model_mean = e04.groupby("model")["mean_rho"].mean().round(6).to_dict()
    golden["E04"] = {
        "per_model_mean_rho": per_model_mean,
        "per_model_dimension_rho": {
            model: dict(zip(g["dimension"], g["mean_rho"].round(6)))
            for model, g in e04.groupby("model")
        },
        "tolerance": 0.01,
        "faithful_to": "E04 faithfulness (attribution vs patching Spearman)",
    }

    # E02 crystallization depth: fraction of the reward margin resolved by the final layers,
    # per (model, dimension). Headline is the per-model mean fraction.
    e02 = pd.read_csv(_dir("e02_lens_population") / "e02_crystallization.csv")
    golden["E02"] = {
        "per_model_mean_crystal": e02.groupby("model")["mean_crystal_frac"]
        .mean()
        .round(6)
        .to_dict(),
        "per_model_dimension_crystal": {
            model: dict(zip(g["dimension"], g["mean_crystal_frac"].round(6)))
            for model, g in e02.groupby("model")
        },
        "tolerance": 0.01,
    }

    # E15 head path patching: the strongest attention head per (model, dimension) by mean
    # absolute effect. The design flags an L12_H6-class head; we record the actual top heads.
    e15 = pd.read_csv(_dir("e15_head_path_patching") / "e15_top_heads.csv")
    top_head = (
        e15.sort_values("mean_abs_effect", ascending=False).groupby(["model", "dimension"]).first()
    )
    golden["E15"] = {
        "top_head_per_model_dimension": {
            f"{m}|{d}": {"head": row["head"], "effect": round(float(row["mean_abs_effect"]), 4)}
            for (m, d), row in top_head.iterrows()
        },
        "global_top_head": {
            "head": str(e15.sort_values("mean_abs_effect", ascending=False).iloc[0]["head"]),
            "effect": round(float(e15["mean_abs_effect"].max()), 4),
        },
        "tolerance": 0.05,
    }

    # E18 ArmoRM multi-objective: the 19-objective conflict matrix. Record its shape and the
    # off-diagonal statistics; the full matrix is the fixture.
    e18 = pd.read_csv(_dir("e18_armorm_multi_objective") / "e18_objective_conflict.csv")
    golden["E18"] = {
        "conflict_rows": int(len(e18)),
        "columns": list(e18.columns),
        "faithful_to": "E18 ArmoRM 19x19 objective geometry",
    }

    # E19 finetune delta: v0.1 -> v0.2 changes. The headline cos(w_r, w_r) = 0.005 is computed
    # from the reward directions (model weights), not present in these CSVs; recorded as a
    # target to reproduce in M3/M5 from weights. The measured behavioural deltas are here.
    e19_concept = pd.read_csv(_dir("e19_finetune_delta") / "e19_concept_delta.csv")
    e19_crystal = pd.read_csv(_dir("e19_finetune_delta") / "e19_crystal_delta.csv")
    e19_hacking = pd.read_csv(_dir("e19_finetune_delta") / "e19_hacking_delta.csv")
    golden["E19"] = {
        "raw_reward_direction_cosine_target": 0.005,
        "cosine_tolerance": 0.01,
        "cosine_note": "raw cos(w_r^v0.1, w_r^v0.2); RAW_ONLY (gate 2). Reproduce from weights.",
        "concept_slope_delta": dict(zip(e19_concept["concept"], e19_concept["delta"].round(5))),
        "crystal_delta": dict(zip(e19_crystal["dimension"], e19_crystal["delta"].round(5))),
        "hacking_effect_delta": dict(zip(e19_hacking["dimension"], e19_hacking["delta"].round(5))),
    }

    # E20 architecture vs finetune decomposition: per-dimension arch_effect vs ft_effect and the
    # verdict (architecture-dominated / finetune-dominated / mixed).
    e20 = pd.read_csv(_dir("e20_arch_vs_finetune") / "e20_per_dim_diff_decomposition.csv")
    golden["E20"] = {
        "per_dimension": {
            row["dimension"]: {
                "arch_effect": round(float(row["arch_effect"]), 4),
                "ft_effect": round(float(row["ft_effect"]), 4),
                "verdict": row["verdict"],
            }
            for _, row in e20.iterrows()
        },
        "tolerance": 0.05,
    }

    return golden


def main() -> None:
    golden = extract()
    out = Path(__file__).resolve().parent / "golden.json"
    out.write_text(json.dumps(golden, indent=2, default=str))

    # Report the E04 per-model means and check them against the design's stated headline.
    print("E-parity golden numbers extracted to", out)
    print("\nE04 per-model mean rho (design names -0.171 / -0.203 / -0.051 / +0.047):")
    for model, rho in sorted(golden["E04"]["per_model_mean_rho"].items()):
        print(f"  {model:42s} {rho:+.4f}")
    stated = np.array([-0.171, -0.203, -0.051, 0.047])
    measured = np.array(sorted(golden["E04"]["per_model_mean_rho"].values()))
    stated_sorted = np.sort(stated)
    print("\n  sorted measured:", np.round(measured, 3))
    print("  sorted stated:  ", np.round(stated_sorted, 3))
    print("  max abs diff (sorted):", round(float(np.max(np.abs(measured - stated_sorted))), 4))
    print("\nE02 per-model mean crystallization:")
    for model, c in sorted(golden["E02"]["per_model_mean_crystal"].items()):
        print(f"  {model:42s} {c:.4f}")
    print("\nE15 global top head:", golden["E15"]["global_top_head"])
    print("E18 conflict rows:", golden["E18"]["conflict_rows"])


if __name__ == "__main__":
    main()
