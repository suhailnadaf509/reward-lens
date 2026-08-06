<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a training record, or a checkpoint at rung 2</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> selection.heritability_h2, selection.evolvability, selection.autonomy</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">gauge</span> invariant</span>
</div>

# Feature heritability

**Your susceptibility index says the reward favours short answers. Can a gradient step actually make the policy shorter?**

Not necessarily, and the two questions are different enough that answering the first and assuming the second is a real way to waste a run.

A behavioural feature varies across rollouts for two reasons. Part of that variance is reachable: some parameter move would produce it. The rest is sampling, and no parameter move produces it at all. Selection acts on the observed variance; the policy responds only to the reachable part. A feature sitting mostly in the unreachable part looks susceptible and is inert.

`reward_lens.measure.indices.heritability` is C2, and it is the one-number test for that.

## Where the quantity comes from

Let \(J = \partial z/\partial\theta\) and let \(F\) be the Fisher information. A natural-gradient step is \(\Delta\theta = \eta F^{-1} g\), and decomposing the advantage in the feature basis gives

\[ \Delta z = \eta\, G\, \beta + \eta\, J F^{-1} e, \qquad G := J F^{-1} J^{\top} \]

which is Lande's equation with its residual named. For any unit vector \(v\), Cauchy-Schwarz gives \(v^{\top} G v \le v^{\top} C v\), so \(C \succeq G\) and \(N := C - G \succeq 0\). **\(N\) is exactly the part of feature variance no parameter perturbation can produce.**

Per feature,

\[ h^2_i = G_{ii} / C_{ii} \in [0, 1] \]

the fraction of a feature's observed variance a parameter move can reach. Three companion indices ship beside it, all on a unit-normalised selection gradient: evolvability \(e(\beta) = \beta^{\top} G \beta\), conditional evolvability \(c(\beta) = 1/(\beta^{\top} G^{-1} \beta)\), and autonomy \(a = c/e\), the share of the response to a given pressure that is not collateral.

They are not interchangeable with \(h^2\) and the distinction has its own literature. \(h^2\) is a **ratio**, dimensionless and bounded, about the proportion of variance that is reachable; \(e\) is a **scale**, in units of a feature squared, about the amount of achievable response. A feature can have a high \(h^2\) and a tiny \(e\), or a low \(h^2\) on a feature so variable that the reachable sliver is still the largest absolute response in the basis. Both ship on every reading.

One field on the evolvability reading is worth knowing about before you read a zero. `beta_outside_G` is the fraction of the selection gradient's squared length lying outside the range of \(G\). When it is non-zero, \(\beta^{\top} G^{-1} \beta\) is infinite and \(c\) is exactly zero, which means the direction has a component no parameter move can produce at all. That is a real answer rather than a numerical failure and is reported as one.

This is not a new idea, and the module cites rather than rediscovers. De Carlo, Ferrante, Zeeuwe, Ellers and Eiben compute \(h^2\) of morphological and behavioural traits of evolved robots and track it across evolution, explicitly to ask which traits a representation can transmit. Same question, genetic machinery rather than gradient machinery. The companion indices are Hansen and Houle's, with eighteen years of estimator theory attached.

## It refuses on both subjects, and the reason is the finding

This is the outcome worth reading the page for, because a reader who takes \(h^2\) to a real run will hit the same two walls.

**On the campaign evidence store, the refusal is `ACCESS_INSUFFICIENT`.** That store holds reward models scored on fixed banks and no reinforcement-learning training run at all, so there is no policy to take a Jacobian or a Fisher against, and \(G = J F^{-1} J^{\top}\) does not exist there. The tempting substitution is `C` itself, and the refusal states what that gives: \(h^2 = 1\) on all seven features, which filters nothing. A ratio that is one by construction is not a measurement, it is the statement that a fixed bank has no unreachable variance because it has no parameters.

**On the 200-step GRPO record the rung-0 realised estimator returns a number out of bounds, and the instrument refuses with `ENVELOPE_VIOLATED`.** The estimate comes back above `1e3` on all three varying features against an upper bound of one. That is not a large heritability and it is not an outlier: \(C \succeq G\) bounds the ratio as a theorem, so a value above one says the numerator and the denominator are not related by the breeder's equation on this record. The number that says so is the selection share, below `1e-5` on `response_chars`, and it is in the refusal because it is what a reader acts on.

Two of the five features carry no estimate at all rather than a bad one. `type_token_ratio` and `n_turns` have no within-group spread, so they never entered the ledger's selection term. Undefined is not out of bounds, and they are named in a separate field.

The default is to refuse. `RealisedHeritability.from_ledgers(ledgers).compute()` returns the refusal; producing the raw number at all takes an explicit `enforce_bounds=False`, and the remedy on the refusal names the subject a real claim would need rather than restating the failure: consecutive steps that share their prompts, and a rung-2 `G`.

## The rung-2 wall, which is the same wall

Going to rung 2 to get a `G` that is not `C` does not rescue this, and the reason is worth knowing before spending the Fisher solves.

The empirical Fisher has rank at most the rollout count. With fewer rollouts than parameters, which is every real policy, every feature lies exactly inside the span of the score vectors, the projection is the identity on it, and the *undamped* plug-in gives \(\hat G = \hat C\) exactly. So \(h^2 = 1\) for every feature no matter what is true. Damping does not fix it either: \(N\) collapses onto the regulariser, and the reported heritability comes out as \(1 - \lambda\).

Both halves of that were measured independently, by two packages that did not coordinate. On the 200-step fixture with the damping taken toward zero the three moving features come back at `0.99999`. On a 24-step window with 192 rollouts against 2,453,368 parameters, the damped solve gives heritability between `0.99135` and `0.99143` on all three features against a damping of `0.0087442`, which is the same number.

That is why every `MetricG` carries its damping and a `damping_stable` flag, and why a reading whose \(h^2\) moves materially across a decade of \(\lambda\) is reporting the regulariser rather than the policy. The damping is not a numerical convenience here, it is the entire content of the estimate.

## What it would take

A usable \(h^2\) at rung 2 needs a policy with more rollouts than the Fisher needs to have rank against, which in practice means sketching the score matrix or doing the \(k\) Fisher solves directly rather than through the kernel form. A usable \(h^2\) at rung 0 needs consecutive steps drawing overlapping prompt sets, so that \(\Delta z\) is a difference between two policies rather than between two task samples: on the reference record the median task overlap between consecutive steps is zero.

Neither is a defect in the instrument, and the instrument says so by refusing with those two sentences attached rather than by returning a number.

The two refusals you can be handed here are [`ACCESS_INSUFFICIENT`](../refusals.md#access-insufficient) on a subject with no policy to differentiate, and [`ENVELOPE_VIOLATED`](../refusals.md#envelope-violated) on a record whose response is not driven by the selection term. The envelope is a single condition, `LINEAR_RESPONSE`, measured by the fraction of the observed motion the selection term explains, and the instrument sets no threshold of its own: every line of the derivation above is a first-order expansion, so when that fraction is near zero every rung of this instrument is reading noise.

The module computes no `G` of its own. `G` arrives as an argument in a named basis, and `measure/indices/` stays free of torch by contract; the computation lives in [the cost book](cost-book.md). Everything on this page is asserted in `tests/acceptance/test_w5_3_selection.py` and `tests/test_measure_heritability.py`.
