<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a grader, its source, and a corpus</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantity</span> grader.card</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">access</span> GRADER: QUERY|REPLICATE, more buys more fields</span>
</div>

# The grader card

**You are about to spend a month of compute optimising against a verifier. What do you know about it?**

Today, almost nothing that bears on the question. A model card states architecture, training mixture and a leaderboard number. A marketplace listing states less. Nothing published states, for any verifier or reward model you can download, what fraction of its branches your corpus has ever exercised, how many edits to it that corpus would fail to notice, or how much of the gap between two rollouts is the grader disagreeing with itself.

`reward_lens.measure.card` is D7, and it assembles thirteen such quantities from the instruments that already produce them onto one page.

```python
from reward_lens.core.types import Access, Component, Phase
from reward_lens.measure.card import CardInputs, card_plan, grader_card, render_card

inputs = CardInputs(verifier=source, corpus=rollouts, exploit_log=log)
access = {Component.GRADER: Access.QUERY | Access.REPLICATE | Access.SOURCE}

print(card_plan(inputs, access=access, phase=Phase.PRE_RUN).render())   # no grader call, no GPU
print(render_card(grader_card(inputs, access=access, phase=Phase.PRE_RUN)))
```

`card_plan` answers "what would this card contain and what would it cost" with no grader call and no GPU. For a lot of readers that report is the product.

Here is the head of a real one, the card for the answer checker most open RLVR math pipelines call:

```text
GRADER CARD  hendrycks/math is_equiv
  what a buyer gets before spending money on a grader, and what an auditor gets after

  subject        verifier:95be1ad8dbf1474a1a7d56058e5f2b25
  substrate      PROGRAM    phase  PRE_RUN
  access         GOLD: QUERY, GRADER: RECORD|QUERY|REPLICATE|SOURCE, RECORD: RECORD,
                 TASK: QUERY|REPLICATE|SOURCE
                 meets D7's stated minimum (GRADER: QUERY|REPLICATE)
  trust          exploratory, computed by the gates
  lowest reading exploratory
  envelope       STATIONARY_GRADER was not measured, so this check did not run. A card
                 assembled across a grader edit describes two programs rather than one.
  cost           2,400 grader calls, which is a floor: 4 of the 5 fields that read do not
                 model their own cost
  not checked    envelope (regime not measured) on 4 field(s) and the card itself; limit
                 of detection on 5 field(s) and the card itself

  5 of 13 fields read and 8 refused. The lowest trust among them is exploratory.

  coverage                   Across 240 rollouts is_equiv has never taken 32% of its
                             branches (12 of 38). Those are behaviours it cannot
                             distinguish.  [exploratory]
                             a random sample of the same size reaches 0.653 against the
                             corpus's 0.684
  surviving mutants          Of 60 mutants of is_equiv, 60 survive: 60 ways it could be
                             wrong that no rollout in your corpus would reveal.
```

Read the `not checked` line before anything else on a card. A check that did not run is not a check that passed, and the card says which is which.

## A card is mostly refusals, and a card with a refusals section is the more informative one

This is the design decision worth arguing for, because the instinct runs the other way.

Every field on a card is a `Reading`, which is `Evidence` or a `Refusal`. A grader nobody instrumented has no replicated scoring design, no exploit log and no recorded abstention channel, so those fields refuse. Each refusal names the instrument that would have filled the field, the input it did not have, and the thing somebody would have to go and record.

A rendered blank names nothing, and a blank is indistinguishable from a measured zero. That is the confusion the artifact exists to remove: "we did not measure the silent-zero rate" and "the silent-zero rate is zero" are opposite findings and a blank cell reports both.

So the refusals are not the card failing to be a card. On the four real subjects it was rendered against, they are most of what the reader learns.

## What it said about four real scoring programs

The card was run on four programs people train against, chosen so the second of each pair differs in kind from the first. Two graders: `is_equiv` from the MATH benchmark, which most open RLVR math pipelines still call directly or through a fork, and the report builder inside SWE-bench's `grading.py`, which decides whether a patch resolved an issue. Two environment reward functions: the GSM8K and Search-R1 scorers that verl ships.

Across the four cards, `52` fields were rendered. `22` read and `30` refused.

**None of the thirty refused for want of access.** That is the result. The access on these subjects is about as good as access ever gets, up to and including the source text, and thirty fields still could not be filled. They divide by reason, and the division is the useful part:

| Reason | Count | What it means here |
|---|---|---|
| `RECORD_INCOMPLETE` | `21` | the access is sufficient, the instrument applies, and the field was never written down |
| `SUBSTRATE_MISMATCH` | `8` | a category error rather than a gap: the curl mass and the Afriat index are about a grader that expresses preferences between items, and all four subjects are programs |
| `ABOVE_LOD_BELOW_LOQ` | `1` | the exploit-family estimator: no family in the log appears exactly twice, which is the quantity Chao1 divides by, so it attached the bias-corrected floor instead of a number it could not support |

Twenty-one of thirty being `RECORD_INCOMPLETE` is the single most useful number on the page, because it says where the work is. Nothing about these graders is unmeasurable. The measurements were never taken and never recorded, and the fix is upstream in whatever produced the run.

The release publishes a refusal summary that groups all thirty by field with the remedy in full, because the question a reader actually has is not "what did this card fail to say" but "what would it take to say it about any grader", and that is answered one field at a time rather than one grader at a time.

## Two findings from the fields that did read

**The corpus is doing less work than it looks like it is.** The coverage instrument does not report a number on its own; it reports it against a mandatory baseline, a random subsample of the same size drawn from the same corpus. On all four subjects the two are nearly the same. `is_equiv`'s `240` real MATH-500 answer pairs reach `0.6534` where the whole corpus reaches `0.6842`, and on the other three the random subsample matches the corpus to four decimal places. Nothing in these corpora is buying branch coverage that a smaller random draw would not have bought.

**Not one of `60` mutants of `is_equiv` is killed by those `240` pairs.** The mutation instrument edits the entrypoint and re-grades; a mutant is killed when a score changes. On `is_equiv` the mutation score is exactly zero: no edit to the canonical math answer checker changes its verdict on any of the gold answer pairs from the benchmark it ships with. The other three subjects kill `2`, `24` and `26` of `60`.

The reason is in the source rather than on the card. `is_equiv` ends in a bare `except:` that returns a raw string comparison, so any failure inside its normaliser, up to and including a `KeyboardInterrupt`, is converted into a verdict. There are three such handlers in its 152 lines, one in `is_equiv` itself and one each in two helpers that return their input unchanged. No caller can tell a normalised match from a fallback comparison, because both return `True`.

The mutation budget is capped at `60` per subject and the cap is printed on every card, because a survival rate measured under a cap is a survival rate over the mutants generated first rather than over all of them.

## Three properties worth knowing before reading one

**The card cannot be asked to trust itself.** Trust is computed by the gates and this package exposes no way to set it. The card additionally prints the lowest trust among the readings it composes, and says in words when its own level exceeds that floor.

**Exploit content is withheld by default.** Two of the thirteen fields carry it: the surviving-mutant list is a reproducible set of edits that make the grader wrong without the corpus noticing, and the false-positive catalogue is the same thing found by search. Both arrive with a sensitive flag on the payload rather than on the row, so a renderer cannot forget it. The published cards are the redacted form: the counts survive, the reproducers do not. Nothing in the script that produced them can ask for the unredacted form, because releasing that content is a decision with a name attached rather than a command-line flag.

**Taking access away only ever turns a reading into a refusal.** Rendered again for a reader holding a log and nothing else, all four cards read zero fields and refuse all thirteen. The artifact does not disappear below the access minimum: what a reader gets there is thirteen remedies, which is more than any marketplace listing carries today.

## What the four cards do not claim

None of the four corpora is a rollout produced by a policy under training. Two are real on both sides, and two are half assembled: the MATH-500 pairs are real gold answers paired by a rule, and the SWE-bench status maps are built from the real gold test lists to span the four outcomes the function distinguishes, because no per-test status dump from a model run is a cheap download.

So nothing here is a statement about what a grader does to an optimisation run. It is a statement about what a grader does to inputs, which is the question a buyer asks before the run rather than the one an auditor asks after it.

The release publishes the full cards, one file per subject, with the capability report that preceded each one. Every number in the write-up is bound to a row in the evidence store the run produced, and `reward-lens-claims` verifies the write-up against that store.
