#!/usr/bin/env python3
"""Regenerate every empirical figure in the docs, dual-theme, from committed artifacts.

Each figure is built only from data the repository actually produced: the weekend
RewardBench run (weekend_experiment/) and the at-scale v2 sweep (outputs/v2_.../),
both living in the v1 run repository. No 8B model is loaded here; everything runs
on CPU from committed JSON/CSV/JSONL. If a source file is missing the figure is
skipped with a notice rather than faked.

Every figure is saved twice, <name>-light.svg and <name>-dark.svg, with a
transparent background and colors drawn from rl_palette so it sits correctly on
either the light or the dark page.

Run from docs/diagrams:  python make_empirical_figures.py
Outputs land in ../content/assets/figures/<name>-{light,dark}.svg.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from rl_palette import palette, apply_style, SEQUENTIAL, DIVERGING  # noqa: E402

# Committed run artifacts live in the v1 run repository, which is not this repository and is not
# published with it. By default it is looked for beside this checkout; point REWARD_LENS_V1_REPO
# somewhere else if you keep it elsewhere. A missing source skips its figure rather than faking it,
# so this path being wrong costs you figures and never a wrong number.
DATA_REPO = os.environ.get(
    "REWARD_LENS_V1_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "reward-lens")),
)
WEEKEND = os.path.join(DATA_REPO, "weekend_experiment")
V2 = os.path.join(DATA_REPO, "outputs", "v2_20260506_222648_unknown")
# Figures are written into the docs assets tree of the clean repo.
OUT = os.path.abspath(os.path.join(HERE, "..", "content", "assets", "figures"))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "fixtures", "e_parity", "golden.json"))
os.makedirs(OUT, exist_ok=True)

THEMES = ("light", "dark")
SKY = "Skywork-Reward-Llama-3.1-8B-v0.2"   # e04 subdir name
ARM = "ArmoRM-Llama3-8B-v0.1"


# --- small IO + helpers -----------------------------------------------------
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else None


def layer_of(name: str) -> int:
    m = re.search(r"_L(\d+)", name)
    return int(m.group(1)) if m else -1


def seq_cmap(theme):
    return LinearSegmentedColormap.from_list("rl_seq", SEQUENTIAL[theme])


def div_cmap(theme):
    return LinearSegmentedColormap.from_list("rl_div", DIVERGING[theme])


def save(fig, name, theme):
    fig.tight_layout()
    path = os.path.join(OUT, f"{name}-{theme}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"    wrote {name}-{theme}.svg")


def _e04_dir():
    hits = sorted(glob.glob(os.path.join(V2, "*e04*faithfulness*")))
    return hits[0] if hits else None


def per_dim_rho(model_subdir):
    """Mean over pairs of the per-pair Spearman(attribution, patching), by dimension.

    This reproduces the golden E04 per-model / per-dimension rho exactly.
    """
    ed = _e04_dir()
    if not ed:
        return None
    faith = load_jsonl(os.path.join(ed, model_subdir, "faithfulness_per_pair.jsonl"))
    if not faith:
        return None
    by_dim = defaultdict(list)
    for r in faith:
        by_dim[r["dimension"]].append(r["spearman_rho"])
    return {dim: float(np.mean(v)) for dim, v in by_dim.items()}


def pooled_components(model_subdir):
    """Per-component attribution and patch effect, averaged over every pair.

    Returns (names, attribution, patch_effect, layers) aligned on the 64 shared
    components (attribution's leading `embed` term, absent from patching, is dropped).
    """
    ed = _e04_dir()
    if not ed:
        return None
    A = load_jsonl(os.path.join(ed, model_subdir, "attribution_per_pair.jsonl"))
    P = load_jsonl(os.path.join(ed, model_subdir, "patching_per_pair.jsonl"))
    if not A or not P:
        return None
    Aidx = {r["pair_id"]: r for r in A}
    Pidx = {r["pair_id"]: r for r in P}
    ca, cp = defaultdict(list), defaultdict(list)
    for pid in set(Aidx) & set(Pidx):
        amap = dict(zip(Aidx[pid]["component_names"], Aidx[pid]["differential_contributions"]))
        pmap = dict(zip(Pidx[pid]["component_names"], Pidx[pid]["patch_effects"]))
        for name, pv in pmap.items():
            if name in amap:
                ca[name].append(amap[name])
                cp[name].append(pv)
    names = sorted(ca, key=lambda n: (layer_of(n), 0 if n.startswith("attn") else 1))
    attr = np.array([np.mean(ca[n]) for n in names])
    patch = np.array([np.mean(cp[n]) for n in names])
    layers = np.array([layer_of(n) for n in names])
    return names, attr, patch, layers


# --- 1. reward-lens curve: the margin forming across layers -----------------
def fig_lens_curve(theme, c):
    d = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    if not d:
        print("  skip lens-curve (no data)"); return
    pair = d["helpfulness"]["pairs"][0]
    curve = np.array(pair["differential_curve"])
    layers = np.array(pair["layers"])
    cryst = pair["crystallization_layer"]
    final = curve[-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color=c["baseline"], lw=1)
    ax.axhline(final / 2, color=c["muted"], lw=0.8, ls=":", zorder=1)
    ax.plot(layers, curve, color=c["chosen"], lw=2.4, zorder=3)
    ax.scatter(layers, curve, color=c["chosen"], s=15, zorder=4,
               edgecolor=c["surface"], linewidth=0.4)
    ax.axvline(cryst, color=c["series_b"], lw=1.5, ls="--", zorder=2)
    ax.annotate(f"crystallizes at\nlayer {cryst} of 32",
                xy=(cryst, final / 2), xytext=(cryst - 15, final * 0.62),
                color=c["series_b"], fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color=c["series_b"], lw=1.2))
    ax.annotate("half the final margin", xy=(2, final / 2 + 0.4),
                color=c["muted"], fontsize=9, ha="left")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"margin  $\Delta = w_r^\top(h_{\rm chosen}-h_{\rm rejected})$")
    ax.set_title("The margin stays near zero for two-thirds of the network, then forms late",
                 fontsize=11.5, loc="left")
    ax.grid(axis="y", alpha=0.5)
    save(fig, "lens-curve", theme)


# --- 2. crystallization by dimension, two models ----------------------------
def fig_crystallization(theme, c):
    sky = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    arm = load_json(os.path.join(WEEKEND, "armo", "lens_results.json"))
    if not sky or not arm:
        print("  skip crystallization-by-dim (no data)"); return
    dims = ["helpfulness", "safety", "correctness", "verbosity"]
    sky_m = [sky[d]["mean_crystallization_frac"] for d in dims]
    sky_s = [sky[d]["std_crystallization_frac"] for d in dims]
    arm_m = [arm[d]["mean_crystallization_frac"] for d in dims]
    arm_s = [arm[d]["std_crystallization_frac"] for d in dims]
    x = np.arange(len(dims)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w / 2, sky_m, w, yerr=sky_s, color=c["chosen"], label="Skywork",
           error_kw=dict(ecolor=c["muted"], lw=1, capsize=3))
    ax.bar(x + w / 2, arm_m, w, yerr=arm_s, color=c["series_b"], label="ArmoRM",
           error_kw=dict(ecolor=c["muted"], lw=1, capsize=3))
    ax.axhline(1.0, color=c["baseline"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_ylabel("crystallization depth (fraction of layers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Skywork decides late and consistently; ArmoRM earlier and noisier",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.5)
    save(fig, "crystallization-by-dim", theme)


# --- 3. attribution vs patching: the centerpiece ----------------------------
def fig_attr_vs_patch(theme, c):
    if not _e04_dir():
        print("  skip attribution-vs-patching (no e04 dir)"); return
    pooled = pooled_components(SKY)
    sky_rho = per_dim_rho(SKY)
    arm_rho = per_dim_rho(ARM)
    if pooled is None or not sky_rho or not arm_rho:
        print("  skip attribution-vs-patching (missing e04 data)"); return
    names, attr, patch, layers = pooled
    rho_scatter, p_scatter = spearmanr(attr, patch)
    sky_mean = float(np.mean(list(sky_rho.values())))
    arm_mean = float(np.mean(list(arm_rho.values())))

    if theme == THEMES[0]:
        print(f"    [attribution-vs-patching] Panel A pooled Skywork-v0.2 component scatter: "
              f"Spearman rho = {rho_scatter:+.3f}  (n={len(names)}, p={p_scatter:.2g})")
        print(f"    [attribution-vs-patching] Panel B per-dimension mean rho: "
              f"Skywork-v0.2 = {sky_mean:+.3f}, ArmoRM = {arm_mean:+.3f}")
        g = load_json(GOLDEN)
        if g and "E04" in g:
            gm = g["E04"]["per_model_mean_rho"]
            print(f"    [attribution-vs-patching] golden E04 cross-check: "
                  f"Skywork-v0.2 = {gm[SKY]:+.3f}, ArmoRM = {gm[ARM]:+.3f}")
        for dim in sorted(sky_rho, key=lambda k: sky_rho[k]):
            print(f"        {dim:22s} sky={sky_rho[dim]:+.3f}  arm={arm_rho.get(dim, float('nan')):+.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 5.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.08]})

    # Panel A: every component, attribution vs patch effect, colored by layer.
    axA.axhline(0, color=c["line"], lw=0.8)
    axA.axvline(0, color=c["line"], lw=0.8)
    sc = axA.scatter(attr, patch, c=layers, cmap=seq_cmap(theme), s=48,
                     edgecolor=c["surface"], linewidth=0.6, zorder=3, vmin=0, vmax=31)
    cb = fig.colorbar(sc, ax=axA, pad=0.02, fraction=0.046)
    cb.set_label("layer (early to late)", color=c["ink2"])
    cb.ax.tick_params(colors=c["muted"])
    cb.outline.set_edgecolor(c["line"])
    i_a = int(np.argmax(attr)); i_p = int(np.argmax(patch))
    axA.annotate(f"{names[i_a]}  (late: explains)", (attr[i_a], patch[i_a]),
                 textcoords="offset points", xytext=(-8, 30), color=c["ink2"],
                 fontsize=9, ha="right",
                 arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1))
    axA.annotate(f"{names[i_p]}  (early: causes)", (attr[i_p], patch[i_p]),
                 textcoords="offset points", xytext=(22, -4), color=c["ink2"],
                 fontsize=9, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=c["muted"], lw=1))
    axA.set_xlabel("attribution  (observational: signed share of the reward)", color=c["ink2"])
    axA.set_ylabel("patch effect  (causal: change in the margin)", color=c["ink2"])
    axA.set_title(f"Skywork-v0.2 components   (this cloud: rho = {rho_scatter:+.2f})",
                  loc="left", fontsize=11)
    axA.text(0.97, 0.55,
             "each point is one component;\nthe mass hugs both axes",
             transform=axA.transAxes, fontsize=9, color=c["muted"], va="center", ha="right")
    axA.grid(alpha=0.3)

    # Panel B: per-dimension rho for both models, the E04 headline.
    dims = sorted(sky_rho, key=lambda k: sky_rho[k])
    y = np.arange(len(dims))
    for i, dim in enumerate(dims):
        axB.plot([sky_rho[dim], arm_rho.get(dim, np.nan)], [y[i], y[i]],
                 color=c["line"], lw=1.4, zorder=1)
    axB.scatter([sky_rho[d] for d in dims], y, color=c["chosen"], s=52, zorder=3,
                label="Skywork-v0.2")
    axB.scatter([arm_rho.get(d, np.nan) for d in dims], y, color=c["series_b"], s=52,
                zorder=3, label="ArmoRM")
    axB.axvline(0, color=c["baseline"], lw=1.1, zorder=2)
    axB.axvline(sky_mean, color=c["chosen"], lw=1.2, ls="--", zorder=2)
    axB.axvline(arm_mean, color=c["series_b"], lw=1.2, ls="--", zorder=2)
    axB.set_yticks(y); axB.set_yticklabels(dims, fontsize=8.5)
    axB.set_ylim(-0.7, len(dims) + 0.4)
    axB.set_xlabel("Spearman rho  (attribution vs patch effect, per dimension)", color=c["ink2"])
    axB.set_title("robustly negative on Skywork, near zero on ArmoRM", loc="left", fontsize=11)
    axB.text(sky_mean, len(dims) - 0.2, f"mean {sky_mean:+.2f}", color=c["chosen"],
             fontsize=8.5, ha="center", va="bottom", fontweight="bold")
    axB.text(arm_mean, len(dims) - 0.2, f"mean {arm_mean:+.2f}", color=c["series_b"],
             fontsize=8.5, ha="center", va="bottom", fontweight="bold")
    axB.legend(frameon=False, loc="lower right", fontsize=9)
    axB.grid(axis="x", alpha=0.3)

    fig.suptitle("The components that explain the reward are not the ones that cause it",
                 fontsize=12.5, x=0.015, ha="left", y=1.02)
    save(fig, "attribution-vs-patching", theme)


# --- 4 & 5. attribution and patching bars (top components) ------------------
def _hbar(items, color, title, xlabel, name, theme, c):
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(range(len(vals)), vals, color=color, height=0.66)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f}", va="center", ha="left", color=c["ink2"], fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.grid(axis="x", alpha=0.4)
    save(fig, name, theme)


def fig_bars(theme, c):
    pooled = pooled_components(SKY)
    if pooled is None:
        print("  skip attribution-bars / patching-bars (no data)"); return
    names, attr, patch, layers = pooled
    ord_a = np.argsort(-attr)[:10]
    ord_p = np.argsort(-patch)[:10]
    attr_items = [(names[i], float(attr[i])) for i in ord_a]
    patch_items = [(names[i], float(patch[i])) for i in ord_p]
    if theme == THEMES[0]:
        print(f"    [attribution-bars] top attribution (Skywork-v0.2, pair-averaged): "
              f"{[(n, round(v, 2)) for n, v in attr_items[:4]]}")
        print(f"    [patching-bars]    top patch effect (Skywork-v0.2, pair-averaged): "
              f"{[(n, round(v, 2)) for n, v in patch_items[:4]]}")
    _hbar(attr_items, c["tier_observational"],
          "Attribution credits the last MLPs",
          "mean differential contribution to the margin", "attribution-bars", theme, c)
    _hbar(patch_items, c["tier_causal"],
          "Patching says the early layers carry the cause",
          "mean patch effect on the margin", "patching-bars", theme, c)


# --- 6. hacking effect sizes, two models, sign flips ------------------------
def fig_hacking(theme, c):
    hits = sorted(glob.glob(os.path.join(V2, "*e06*hacking*", "e06_hacking_effects.csv")))
    if not hits:
        print("  skip hacking-effects (no data)"); return
    rows = list(csv.DictReader(open(hits[0])))
    order = ["length", "confidence", "formatting", "repetition", "sycophancy"]

    def d_of(model, dim):
        for r in rows:
            if r["model"] == model and r["dimension"] == dim:
                try:
                    return float(r["cohens_d"])
                except (TypeError, ValueError):
                    return np.nan
        return np.nan

    sky = [d_of(SKY, k) for k in order]
    arm = [d_of(ARM, k) for k in order]
    y = np.arange(len(order))[::-1]; h = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axvline(0, color=c["baseline"], lw=1.2, zorder=1)
    ax.barh(y + h / 2, sky, h, color=c["chosen"], label="Skywork", zorder=2)
    ax.barh(y - h / 2, arm, h, color=c["series_b"], label="ArmoRM", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("Cohen's d   (positive = the reward model prefers the hackable variant)")
    ax.set_title("The same surface feature flips sign between models", fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="lower right")
    for k in order:
        s, a = d_of(SKY, k), d_of(ARM, k)
        if s == s and a == a and np.sign(s) != np.sign(a):
            yy = y[order.index(k)]
            ax.text(ax.get_xlim()[1], yy, "flip  ", color=c["rejected"], fontsize=9,
                    va="center", ha="right", fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    save(fig, "hacking-effects", theme)


# --- 7. concept dose-response ----------------------------------------------
def fig_concept(theme, c):
    d = load_json(os.path.join(WEEKEND, "skywork", "concepts_report.json"))
    if not d:
        print("  skip concept-dose-response (no data)"); return
    inter = d["interventions"]
    ranked = sorted(inter.items(), key=lambda kv: -abs(kv[1].get("causal_slope", 0)))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhline(0, color=c["baseline"], lw=1)
    ax.axvline(0, color=c["line"], lw=0.8)
    for name, info in ranked:
        dr = info.get("dose_response", [])
        if not dr:
            continue
        xs = [p["strength"] for p in dr]; ys = [p["delta_reward"] for p in dr]
        top = (name == ranked[0][0])
        ax.plot(xs, ys, color=c["chosen"] if top else c["line"],
                lw=2.6 if top else 1.4, zorder=3 if top else 2,
                marker="o" if top else None, ms=5)
        if top:
            ax.text(xs[-1], ys[-1], f"  {name}", color=c["ink"], fontsize=10.5,
                    va="center", fontweight="bold")
    ax.text(1.9, 1.35, "other concepts", color=c["muted"], fontsize=9.5, va="center", ha="left")
    slope = ranked[0][1]["causal_slope"]; cos = ranked[0][1]["alignment_cosine"]
    ax.set_xlabel("strength added along the concept direction")
    ax.set_ylabel("change in reward")
    ax.set_title("Push a concept into the activation, the reward follows\n"
                 f"'{ranked[0][0]}' moves reward {slope:.2f} per unit  "
                 f"(cosine with $w_r$ = {cos:.2f})", fontsize=11, loc="left")
    ax.grid(alpha=0.4)
    save(fig, "concept-dose-response", theme)


# --- 8. cross-model formation overlay --------------------------------------
def fig_cross_model(theme, c):
    sky = load_json(os.path.join(WEEKEND, "skywork", "lens_results.json"))
    arm = load_json(os.path.join(WEEKEND, "armo", "lens_results.json"))
    if not sky or not arm:
        print("  skip cross-model-overlay (no data)"); return

    def mean_formation(res):
        curves = []
        for p in res["helpfulness"]["pairs"]:
            v = np.array(p["differential_curve"], dtype=float)
            if abs(v[-1]) < 1e-6:
                continue
            curves.append(v / v[-1])
        return np.mean(curves, axis=0)

    sky_c, arm_c = mean_formation(sky), mean_formation(arm)
    frac = np.linspace(0, 1, len(sky_c))
    r = float(np.corrcoef(sky_c, arm_c)[0, 1])
    if theme == THEMES[0]:
        print(f"    [cross-model-overlay] mean-curve correlation r = {r:+.3f}")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.5, color=c["muted"], lw=0.8, ls=":")
    ax.plot(frac, sky_c, color=c["chosen"], lw=2.4, label="Skywork")
    ax.plot(frac, arm_c, color=c["series_b"], lw=2.4, label="ArmoRM")
    ax.set_xlabel("fractional depth (layer / total layers)")
    ax.set_ylabel("mean normalized margin (0 to final)")
    ax.set_ylim(-0.1, 1.08)
    ax.set_title(f"Both leave the decision to the late layers   (curve correlation r = {r:.2f})",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.4)
    save(fig, "cross-model-overlay", theme)


# --- 9. ArmoRM 19-objective cosine heatmap ----------------------------------
def fig_armo_cosine(theme, c):
    hits = sorted(glob.glob(os.path.join(V2, "*e18*", "*", "armo_obj_cosine.json")))
    if not hits:
        print("  skip armo-objective-cosine (no data)"); return
    m = np.array(load_json(hits[0])["cosine_matrix"])
    n = m.shape[0]
    off = m[~np.eye(n, dtype=bool)]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    norm = TwoSlopeNorm(vmin=min(-0.2, float(off.min())), vcenter=0, vmax=1.0)
    im = ax.imshow(m, cmap=div_cmap(theme), norm=norm)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("cosine between objective directions", color=c["ink2"])
    cb.ax.tick_params(colors=c["muted"])
    cb.outline.set_edgecolor(c["line"])
    ax.set_xticks(range(0, n, 2)); ax.set_yticks(range(0, n, 2))
    ax.set_xlabel("objective index"); ax.set_ylabel("objective index")
    ax.set_title(f"ArmoRM's {n} objectives are mostly aligned, but not one direction",
                 fontsize=11, loc="left")
    save(fig, "armo-objective-cosine", theme)


if __name__ == "__main__":
    print(f"writing dual-theme figures to {OUT}")
    for theme in THEMES:
        print(f"  theme: {theme}")
        c = apply_style(theme)
        fig_lens_curve(theme, c)
        fig_crystallization(theme, c)
        fig_attr_vs_patch(theme, c)
        fig_bars(theme, c)
        fig_hacking(theme, c)
        fig_concept(theme, c)
        fig_cross_model(theme, c)
        fig_armo_cosine(theme, c)
    print("done.")
