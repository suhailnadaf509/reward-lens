<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a grader or a policy, plus a certified reference</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> instrument.recovery_auc, instrument.erasure_cost, judge.commitment_position</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">access</span> MUTATE on the subject, for the controls</span>
</div>

# Reference materials, and instruments that publish their losses

**Your method found the planted rule. So did difference-in-means. Which of those two facts did you report?**

`reward_lens.measure.selection` is series C3 through C8, and it has one idea behind all six instruments: a claim about a direction is worth what its controls are worth. Two of the six are about making the comparison honest, three are controls, and one is a reference material, which is the piece nobody in this field currently pays for.

| | Instrument | What it does |
|---|---|---|
| C3 | `InstrumentRecoveryTable` | every localisation method against one planted key, losses included |
| C4 | `ErasureCost` | what an erasure removes, what it costs, and the dose window |
| C5 | `AcuteChronic` | the effect that survives continued training |
| C6 | `RescueFraction` | put the ablated component back and check the behaviour returns |
| C7 | `DoubleDissociation` | necessity is not sufficiency, and one dissociation is not two |
| C8 | `VerdictDirection` | when a judge's verdict direction stopped moving |

The estimators underneath live in `reward_lens.policy.selection` and take arrays rather than subjects, so the same code reads a grader and a policy without knowing which it has.

## A table that cannot be a table of winners

C3's contract is not "report your method's recovery". It is: run every localisation method you can against one planted key and publish the ranking, including the rows where the method that read the model's internals came out below one that read none.

The contract is written so that a table of four winners does not discharge it. It requires four or more methods **and** at least one white-box row placed below the best method that read no internals. The requirement is about the losses being visible, not about the count.

```python
from reward_lens.measure.selection import InstrumentRecoveryTable

table = InstrumentRecoveryTable(panel, planted, ours="diffmean").table()
print(table.n_methods, [r.name for r in table.losers()])
```

## The refusal carries the table

Ask C3 for a `Reading` without a certified reference and it refuses with `REFERENCE_UNCERTIFIED`. That is the right reason: a recovery number calibrated against a reference with no uncertainty of its own inherits an error nobody has measured.

What makes this one worth studying is that the refusal is **bounded**. `Refusal.partial` carries the whole recovery table, losers included, because the table is the deliverable and the certification governs what you may claim about the number rather than whether you may see the ranking. A reader holding that refusal has the ordering, has the losses, and knows exactly what the missing certificate would buy.

## The library's first certified reference material

Section L1 asks for a reference material with an uncertainty of its own, and this is it. Twelve micro-organisms were trained at four planted doses across three seeds, plus three stability checkpoints, and all three ISO Guide 35 terms were measured at model level rather than at data level.

| Term | Value | What it is |
|---|---|---|
| `u_characterisation` | `0.04266` | the standard error of the fitted dose-response line |
| `u_homogeneity` | `0.1034` | spread between seeds trained on identical data |
| `u_stability` | `0.07766` | drift after further training |
| `u_CRM` | `0.13617` | the three composed in quadrature |

The composed value is `None` rather than a partial sum when any term is missing, deliberately. Adding up the terms that happen to exist and calling the result `u_CRM` makes an uncharacterised reference look better than a characterised one with a large homogeneity term, which is the exact understatement the type exists to stop.

Producing it costs about eleven minutes of CPU, which is why the acceptance run restates the certificate rather than re-measuring it; `certified_micro_reference` in `reward_lens.organisms.dose` is the function that produced it and `tests/acceptance/test_w3_6_labels.py` owns the machinery.

**The measurement is model level, and that is the whole point.** The existing plant helper measures the realised dose in the *data* and says in as many words that this is a floor rather than the term the model-organism lottery is about: two trunks trained on identical data with different seeds can express the planted rule at different strengths, and that difference is invisible without training. Here every plant is a trunk that was actually trained, the response is the trained model's own behaviour, and the stability arm re-measures after further training.

## L1's kill condition does not fire, and that is the finding

The catalogue entry for L1 says: *if homogeneity is negligible across seeds, single-seed plants are fine and this is one measurement, once.* That would have retired most of the work.

Measured, homogeneity is the **largest** of the three terms, at `2.4` times the characterisation term. So it does not fire, and the finding is the reason: single-seed plants are not fine. A planted organism trained once and used as ground truth carries an uncertainty larger than the fit uncertainty everyone does report, and nobody has been reporting it because nobody has been training the same organism twice.

That is the sort of result a kill condition exists to produce. It was written down before the measurement, it named the outcome that would have ended the line of work, and the measurement came back the other way with a number attached.

## Reconciling an erasure against somebody else's

C4's clause asks for the erasure result to be reconciled with the published alternative, or for the discrepancy to be documented. It does both, and the reconciliation carries three things rather than one.

A **verdict**, so the comparison lands somewhere. A **computed random-scoring floor**, because an accuracy delta is uninterpretable without one: a forty-point drop from a high accuracy means one thing when the floor is zero and something completely different when the floor is a quarter, and best-of-N preference benchmarks have a floor of one over N that almost nothing reports beside the headline. And a list of **named differences** between the two experiments, drawn from what each side actually reports, so that a disagreement is attributed rather than shrugged at.

The reconciliation runs against the campaign's own stored result and the benchmark's own row structure. Because that store lives outside the repository, the arithmetic is covered unconditionally by a second test, so the clause is never silently skipped on a fresh checkout.

## Controls that refuse rather than remind

C8 is the clearest statement of the series' idea. It refuses without its four controls, and refuses again when they fire.

A direction carrying a claim has to be decodable, has to be used, has to be unmatched by a dumb baseline, and has to be unmatched by a coherent but irrelevant semantic direction. A condition nobody measured is not a condition that passed, so the admission protocol is a gate that returns a `Refusal` rather than a checklist in a docstring. "The controls run before any claim" is discharged by there being no code path that returns `Evidence` without them, which is asserted rather than described.

C6 and C7 are the other half of the same discipline, and each of them says in its own docstring what it cannot do.

Ablating a component and watching the behaviour drop shows necessity. Putting it back and watching it return is what separates an ablation from a lesion that broke something else, and that is `RescueFraction`, the cheapest methodological upgrade in the catalogue: one extra forward pass over an ablation you already ran. It reports a fraction rather than a verdict on purpose, because a within-pass re-injection at a later layer puts the coordinate into a residual stream the intervening layers have already written to under the ablated condition, so a fraction below one confounds "the direction was not sufficient" with "the computation above had already gone elsewhere".

And one dissociation is not two. `DoubleDissociation` refuses on a 2x2 that is not complete, because three of the four cells is a single dissociation wearing a 2x2's clothes, and the missing cell is always the one that could have shown the effect was a difficulty difference. It is also careful about what it licenses: ablation tests necessity, and a double dissociation says two components are differently necessary for two behaviours. It does not say either is sufficient. Steering is the experiment for that, and papers routinely report one and claim the other.

Everything on this page is asserted in `tests/acceptance/test_w5_5_selection.py`. C5, the chronic arm, is registered and written and has not been run, because it needs continued training on a real subject rather than a fixture.
