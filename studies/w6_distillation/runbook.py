"""The runbook: what the maintainer types, in order, if the compute gets bought.

Rendered from code rather than written out, so the prices, the study id and the power numbers in it
came from the same functions the study uses and cannot drift from them. `RUNBOOK.md` in this
directory is the rendered output and a test asserts the two agree.

    python -m studies.w6_distillation.runbook > studies/w6_distillation/RUNBOOK.md
"""

from __future__ import annotations

from studies.w6_distillation.analysis import TARGET_CONTRAST_PP, build_spec, frozen_study
from studies.w6_distillation.price import (
    NO_PUBLIC_TRIPLE,
    Assumptions,
    price,
    reference_multi_seed_gpu_hours,
)


def _header() -> list[str]:
    frozen = frozen_study(frozen_at="1970-01-01T00:00:00+00:00")
    bill = price()
    a = bill.assumptions
    return [
        "# W6.3 / K1 runbook: measuring the distillation gap",
        "",
        f"Study `{frozen.study_id}`, frozen before the subject exists. The id depends only on the",
        "spec, so if you edit a prediction it changes and the change is visible in every row the",
        "study stamps.",
        "",
        "## The decision this runbook is for",
        "",
        "Nothing in `studies/w6_distillation/` has ever been run against a real pair of",
        "checkpoints. The code, the frozen predictions, the acceptance test and this runbook exist",
        "so that buying the compute is a decision with a bill in front of it. The bill:",
        "",
        "```",
        bill.render(),
        "```",
        "",
        "The multi-seed reference study the build spec prices at ten seeds by three conditions is",
        f"{reference_multi_seed_gpu_hours():,.0f} GPU-hours, so K1 at {a.n_seeds} seeds is "
        f"{bill.gpu_hours / reference_multi_seed_gpu_hours():.0%} of that.",
        "",
        "**What you cannot buy at any price.**",
        "",
        NO_PUBLIC_TRIPLE,
        "",
        "## What has to be true before arm A1",
        "",
        "1. **One base checkpoint**, and every later artifact descends from it. If the expert and",
        "   the student have different bases there is no shared feature scale and no installed",
        "   shift to divide by, and the reading answers a different question under K1's name.",
        "2. **One prompt file**, held out of every training set used in A1 and A2. Every arm answers",
        "   the same prompts, keyed by prompt id and not by position.",
        "3. **One decoding configuration** for every draw: temperature, top_p, max_new_tokens, the",
        "   stop set and the completions per prompt. The instrument refuses when the declared",
        "   settings differ between arms, and it can only see what you declare, so declare them.",
        f"4. **{a.completions_per_prompt} completions per prompt; two is the hard minimum.** The",
        "   survival slope is corrected for sampling error in its own regressor and the correction",
        "   needs a within-prompt variance. With one completion there is none, the correction",
        "   cannot run, and the uncorrected slope depends on how many completions you drew.",
        "",
    ]


def _arms() -> list[str]:
    a = price().assumptions
    return [
        "## The arms, in order, and what to check after each",
        "",
        "### A0 - base rollouts, and the blanks (inference only, minutes)",
        "",
        "Draw the base checkpoint's completions on the prompt file, then draw them again at two",
        f"more sampling seeds. That gives the base arm and {a.n_blank_arms} blank arms. Run this",
        "**first**, before any training, because it is the only arm that can be checked without",
        "anything to compare against and because a problem here invalidates everything after it.",
        "",
        "```python",
        "from reward_lens.policy.hf import HFPolicy  # or your own sampler",
        "from reward_lens.policy.base import SampleSpec",
        "",
        "policy = HFPolicy.load(BASE_CHECKPOINT)",
        f"spec = SampleSpec(max_new_tokens={a.max_new_tokens}, temperature=1.0, top_p=1.0, "
        f"group_size={a.completions_per_prompt}, seed=0)",
        "rollouts = policy.sample(prompts, spec)  # then seed=1 and seed=2 for the blanks",
        "```",
        "",
        "**Check:** every prompt came back with the full group and non-empty assistant text. Then",
        "run the detection floor on the blanks alone and read `blank_mean`. It should sit near",
        "zero. A blank mean far from zero means the two draws differ systematically, which means",
        "the decoding settings moved between them, and every later number would inherit it.",
        "",
        "**A failed A0 looks like:** a `RECORD_INCOMPLETE` refusal naming how many prompts had no",
        "readable rollout, or a blank mean of the same order as the shift you are trying to",
        "measure. Fix the sampler before spending anything on A1.",
        "",
        "### A1 - the RL expert (the expensive arm)",
        "",
        f"{a.n_seeds} group-relative RL runs from the base against a real grader, one per seed,",
        "differing in nothing but the seed. This is the arm this library exists to instrument, so",
        "tap it: `reward_lens.tap.adapters.trl` records the run while it happens and F1, F2 and the",
        "threshold instruments then have something to read besides K1.",
        "",
        "```bash",
        "accelerate launch train_grpo.py --seed 0 --output_dir experts/seed0   # then 1, 2",
        "```",
        "",
        "**Check after each run, before starting the next:** the reward trace rose and the entropy",
        "trace did not collapse. That is void condition 1, `ARM_COLLAPSE`, and reading a study off",
        "a collapsed arm is the failure `studies.void` exists to prevent. Also draw the expert's",
        "rollouts now and confirm the installed shift clears the detection limit from A0: if RL",
        "installed nothing this basis can see, K1 has no denominator and there is no point paying",
        "for A2.",
        "",
        "**A failed A1 looks like:** a `BELOW_LOD` refusal saying every feature's installed shift is",
        "under the limit. That is not a failed measurement, it is the finding that this run left no",
        "behavioural trace the basis can see. Widen the basis with `RecordedFeatures` over your own",
        "features, or train longer, and re-check before A2.",
        "",
        "### A2 - the distilled student",
        "",
        "On-policy distillation from each expert back into **the same base**, one student per seed.",
        "Per-token log-ratio teacher reward is the cheap version every lab but DeepSeek ships; if",
        "you use full-vocabulary logit distillation instead, record which, because that is the one",
        "named technical disagreement between labs on this step and nobody publishes the ablation.",
        "",
        "**Check:** the student's aggregate score on the grader is within noise of the expert's.",
        "That is what the labs report and it is also H4's premise. If the student is visibly worse",
        "on the grader, the distillation run underconverged and K1 would be measuring that instead.",
        "",
        "**A failed A2 looks like:** a student whose score is well below its teacher's, or a",
        "survival fraction and a reliability that both sit at the very top of their range, which",
        "usually means the student was initialised from the expert rather than from the base and",
        "there was never a gap to measure.",
        "",
        "### A3 - the audit draw",
        "",
        "Draw expert and student rollouts on the same prompt file, same decoding, same group size.",
        "Nothing else changes between them.",
        "",
        "### A4 - localisation (optional, and it is nearly free)",
        "",
        "Per-token gradients on the expert and the student over the audit rollouts, at",
        "`POLICY: BACKWARD`. Compose `reward_lens.policy.credit` rather than writing a second",
        "gradient disintegration; its conservation identity closes at 2.704e-16 and a second one",
        "would not. This is the arm that answers the published audit's actual question, *where* the",
        "teacher signal acts, and at under one GPU-hour it costs less than the electricity of",
        "deciding whether to run it.",
        "",
    ]


def _analysis() -> list[str]:
    spec = build_spec()
    return (
        [
            "## Running the study",
            "",
            "```python",
            "from reward_lens.core.store import EvidenceStore",
            "from reward_lens.studies import run_study, render_report",
            "from studies.w6_distillation.analysis import build_spec",
            "from studies.w6_distillation.survival import Arm",
            "",
            "arms = {",
            '    "base": Arm("base", rollouts_by_prompt(BASE_ROLLOUTS)),',
            '    "expert": Arm("expert", rollouts_by_prompt(EXPERT_ROLLOUTS)),',
            '    "student": Arm("student", rollouts_by_prompt(STUDENT_ROLLOUTS)),',
            '    "blank0": Arm("blank0", rollouts_by_prompt(BASE_ROLLOUTS_SEED1)),',
            '    "blank1": Arm("blank1", rollouts_by_prompt(BASE_ROLLOUTS_SEED2)),',
            '    "hack_features": ("...",),  # the features you call reward-hacking-relevant',
            '    "markers": ("...",),  # literals a hack is known to contain, for the string match',
            '    "sampling": {"base": DECODE, "expert": DECODE, "student": DECODE},',
            "}",
            "frozen, result = run_study(build_spec(), subjects=arms, store=EvidenceStore(PATH))",
            "print(render_report(frozen, result, EvidenceStore(PATH)))",
            "```",
            "",
            "`hack_features` is the one judgement call in the whole procedure and it is registered by",
            "being written into the study's subjects before the run. Name them from the grader's own",
            "structure, not from what the reading turns out to say.",
            "",
            "## Reading the result",
            "",
        ]
        + [
            f"- **{h.id}**: {h.statement}. Registered as `{h.prediction.metric} "
            f"{h.prediction.comparator} {h.prediction.threshold:g}`."
            for h in spec.hypotheses
        ]
        + [
            "",
            "and three kill criteria:",
            "",
        ]
        + [
            f"- **{k.id}** fires on `{k.metric} {k.comparator} {k.threshold:g}`: {k.description}"
            for k in spec.kill_criteria
        ]
    )


def _power() -> list[str]:
    """The power statement, recomputed on the planted subject so the numbers came from a run."""
    import tempfile

    from reward_lens.core.store import EvidenceStore
    from reward_lens.studies import run_study

    with tempfile.TemporaryDirectory() as d:
        _, result = run_study(build_spec(), store=EvidenceStore(d))
    m = result.metrics
    a = Assumptions()
    return [
        "",
        "## Power, at the n this runbook buys",
        "",
        "Measured on the planted subject, which fixes the noise level at the plant's and not at a",
        "real subject's. Read it as the scaling rule rather than as a promise, and recompute it",
        "from your own A0 draw before committing to a prompt count.",
        "",
        f"- On {m['n_prompts']:.0f} prompts at 4 completions each, the minimum detectable survival",
        f"  contrast is {m['contrast_mde_pp']:.1f} pp at 80% power and alpha 0.05.",
        f"- Resolving a {TARGET_CONTRAST_PP:.0f} pp contrast therefore needs about "
        f"{m['contrast_prompts_for_target']:.0f} prompts at that",
        f"  noise level. The priced design is {a.n_prompts:,} prompts at "
        f"{a.completions_per_prompt} completions, which is "
        f"{a.n_prompts / m['contrast_prompts_for_target']:.1f}x that in",
        "  prompts and halves the per-prompt sampling variance again, so the margin is real.",
        f"- The expert-versus-student detection arm reaches power {m['detector_power']:.2f} at the "
        f"realised",
        f"  {m['n_detector_items']:.0f} rollouts, with N* = {m['detector_n_star']:.0f} for 80%.",
        "- The minimum detectable effect scales as one over the square root of the prompt count, so",
        "  a pilot of 50 prompts after A1 tells you what the full draw will resolve.",
        "",
        "## What this cannot tell you, however much you spend",
        "",
        "K1 reads rollouts. A distillation step that preserves every measured behaviour by a",
        "different mechanism reads here as full survival, and only A4 could tell those apart. The",
        "survival fraction is conditional on the feature basis, so a basis that misses the axis",
        "distillation moved reports high survival: the fitted feature names are on every reading",
        "for exactly that reason. And the blank arm bounds sampling noise at fixed weights, not the",
        "seed-to-seed variability of the RL run, which is why the price buys three seeds and not",
        "one.",
    ]


def render_runbook() -> str:
    """The whole runbook, as markdown."""
    return "\n".join(_header() + _arms() + _analysis() + _power()) + "\n"


if __name__ == "__main__":  # pragma: no cover - the regeneration entry point
    print(render_runbook(), end="")


__all__ = ["render_runbook"]
