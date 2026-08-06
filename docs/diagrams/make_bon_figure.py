#!/usr/bin/env python3
"""Build the best-of-n ladder figure (best-of-n-ladder-{light,dark}.svg).

The claim the figure makes: you can price optimization pressure in nats before you
spend it. The left panel is the exact best-of-n KL identity, computed from the
library itself (``reward_lens.loops.bon_kl``): ``KL(bo_n || base) = log n - (n-1)/n``.
It is a function of n alone, so the whole ladder is known before any policy is run.
The right panel is the payoff side, the (KL, expected-reward) frontier, computed for
real by ``reward_lens.loops.bon_ladder`` on a synthetic bank of base-policy scores
drawn from N(0, 1). Nothing here is fabricated: every KL is the exact identity and
every reward is the library's plug-in expected-maximum estimator on the same seed.

Standalone, CPU-only, no model download. From an environment with the package installed:
    python docs/diagrams/make_bon_figure.py

Outputs land in docs/content/assets/figures/best-of-n-ladder-{light,dark}.svg.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import rl_palette` works when run standalone
from rl_palette import apply_style, palette  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "content", "assets", "figures"))
os.makedirs(OUT, exist_ok=True)

from reward_lens.loops import DEFAULT_NS, bon_kl, bon_ladder  # noqa: E402

# --- compute once, theme-independent ----------------------------------------
# The ladder: the doubling sequence out to 1024 (a subset of DEFAULT_NS). The KL
# identity holds unchanged across the whole DEFAULT_NS to 10^4; 1024 keeps the log
# axis legible while showing the curve's shape.
NS = np.array([n for n in DEFAULT_NS if n <= 1024], dtype=np.int64)

# The price: exact, model-free, a function of n alone.
KL = bon_kl(NS)

# The payoff: a real library estimate of the frontier on synthetic base-policy
# scores. A bank of 32 prompts x 16384 draws from N(0, 1); best-of-n keeps the
# max, so the expected reward is the plug-in expected maximum of n draws. Seeded,
# so the figure is reproducible.
_rng = np.random.default_rng(0)
_banks = _rng.standard_normal((32, 16384))
_ladder = bon_ladder(_banks, ns=NS).value
assert np.allclose(_ladder.kl, KL), "ladder KL must equal the exact identity"
GAIN = _ladder.expected_reward - _ladder.baseline_reward  # reward gained over base
SEM = _ladder.reward_sem
IDX = {int(n): i for i, n in enumerate(NS)}  # n -> array index, for annotation


def _fmt_pow2(n: int) -> str:
    return str(int(n))


def make(theme: str) -> None:
    c = apply_style(theme)
    price = c["accent"]      # the KL cost, warm
    reward = c["chosen"]     # the reward gained, the house "toward what reward wants" blue
    ink, ink2, muted = c["ink"], c["ink2"], c["muted"]
    base, surface = c["baseline"], c["surface"]

    plt.rcParams.update({"font.size": 11, "axes.titlecolor": ink, "figure.dpi": 120})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.9, 4.5))

    # ---- Panel A: the price, KL(n), exact and model-free -------------------
    axL.plot(NS, KL, color=price, lw=2.4, zorder=3, solid_capstyle="round")
    axL.scatter(NS, KL, s=26, color=price, zorder=4, edgecolor=surface, linewidth=0.9)
    axL.set_xscale("log", base=2)
    axL.set_xticks(NS)
    axL.set_xticklabels([_fmt_pow2(n) for n in NS], fontsize=8.5)
    axL.set_xlim(NS[0] * 0.85, NS[-1] * 1.18)
    axL.set_ylim(-0.28, KL[-1] * 1.16)
    axL.set_xlabel("best-of-n draws  (n)", color=ink2)
    axL.set_ylabel("KL of best-of-n from base  (nats)", color=ink2)
    axL.set_title("1  ·  The price is fixed by n alone", loc="left", fontsize=12, color=ink)
    axL.grid(axis="y", alpha=0.55)

    # the exact identity, stated in the corner
    axL.text(0.035, 0.93, r"$\mathrm{KL} = \ln n - \dfrac{n-1}{n}$",
             transform=axL.transAxes, fontsize=12.5, color=ink, va="top", ha="left")
    axL.text(0.035, 0.79, "no model, no scores, no fit",
             transform=axL.transAxes, fontsize=9, color=muted, va="top", ha="left")

    # mark two ladder points with their real cost; the dotted guide ties n to the axis
    for n in (8, 64):
        i = IDX[n]
        axL.plot([n, n], [0, KL[i]], color=price, lw=0.9, ls=":", alpha=0.55, zorder=2)
        axL.annotate(f"n = {n}\n{KL[i]:.2f} nats", xy=(n, KL[i]),
                     xytext=(-7, 12), textcoords="offset points",
                     color=ink2, fontsize=9.5, ha="right", va="bottom")
    # best-of-1 is the base policy: free
    axL.annotate("best-of-1 = base policy, 0 nats", xy=(1, 0),
                 xytext=(0.085, 0.115), textcoords=axL.transAxes,
                 color=muted, fontsize=8.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=base, lw=1))

    # ---- Panel B: the payoff, expected reward vs KL ------------------------
    axR.fill_between(KL, GAIN - SEM, GAIN + SEM, color=reward, alpha=0.16, lw=0, zorder=2)
    axR.plot(KL, GAIN, color=reward, lw=2.4, zorder=3, solid_capstyle="round")
    axR.scatter(KL, GAIN, s=26, color=reward, zorder=4, edgecolor=surface, linewidth=0.9)
    axR.set_xlim(-0.25, KL[-1] * 1.06)
    axR.set_ylim(-0.15, GAIN[-1] * 1.20)
    axR.set_xlabel("KL of best-of-n from base  (nats)", color=ink2)
    axR.set_ylabel("expected reward gained over base", color=ink2)
    axR.set_title("2  ·  and it prices what you get for it", loc="left", fontsize=12, color=ink)
    axR.grid(alpha=0.5)

    # the diminishing-returns reading, in the open upper-left
    axR.text(0.055, 0.90, "the first nat buys most of the gain;\nlater nats buy little",
             transform=axR.transAxes, fontsize=9, color=muted, va="top", ha="left")
    # two points along the frontier
    i = IDX[8]
    axR.annotate(f"n = 8\n{GAIN[i]:.2f} reward for {KL[i]:.2f} nats", xy=(KL[i], GAIN[i]),
                 xytext=(12, -4), textcoords="offset points",
                 color=ink2, fontsize=9, ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=muted, lw=1))
    i = IDX[512]
    axR.annotate(f"n = 512\n{GAIN[i]:.2f} reward for {KL[i]:.2f} nats", xy=(KL[i], GAIN[i]),
                 xytext=(-10, 22), textcoords="offset points",
                 color=ink2, fontsize=9, ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=muted, lw=1))
    # honest source label, bottom-right corner
    axR.text(0.975, 0.05, "synthetic base policy ~ N(0, 1)\nlibrary plug-in estimator",
             transform=axR.transAxes, fontsize=8.5, color=muted, va="bottom", ha="right")

    for ax in (axL, axR):
        ax.tick_params(length=3)

    fig.tight_layout(pad=1.1, w_pad=2.8)
    for ext_theme in (theme,):
        p = os.path.join(OUT, f"best-of-n-ladder-{ext_theme}.svg")
        fig.savefig(p, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote best-of-n-ladder-{theme}.svg")


if __name__ == "__main__":
    print(f"writing best-of-n ladder figure to {OUT}")
    print("exact KL(n) ladder (nats):")
    for n in (1, 2, 4, 8, 16, 64, 512, 1024):
        print(f"  n={n:5d}  KL={float(bon_kl(n)):.4f}")
    make("light")
    make("dark")
    print("done.")
