# Findings

What this library measured while it was being built, on real subjects, with the predictions frozen
first.

Every number below has a package and an acceptance test behind it, and is written
in code style with the package that produced it named beside the section. Nothing here is
illustrative and nothing here was typed from memory. **Six numbers in the first version of this
document were wrong and are corrected**; how they were found, and why the automated check could not
have found three of them, is the last section, because it is a result about this project's own
methods rather than an apology.

---

## Why there are so many refutations in here

Ten predictions were registered before the instruments that resolve them existed, and **four of them
resolved against the prediction**. Several instruments fired their own kill conditions. One spike
returned a flat no-go. The library's own worst-looking published result turned out to be the number
the field does not have.

That reads as failure and it is not, for a reason worth stating once at the top: **nobody else can
produce a result that could have refuted itself.** A ledger of preregistered refutations is what a
measuring instrument looks like when it is working. A ledger with no refutations in it is either a
lucky project or an unfalsifiable one, and the second is far more common.

So the ordering below is deliberate. Results about the field come first, because they are what a
reader came for. Results about our own instruments come second, because they are what makes the
first set believable. And the caveat that governs the whole document has its own section rather than
a footnote, because it applies to more results than any single finding does.

---

## Results about the field

### A number the field cites constantly, and nobody measures

A widely used measure of how "intransitive" a set of preference comparisons is reports a mass of
`0.214` on a well-known reward-model corpus. **That number contains no intransitivity at all.**

The comparison graph has zero cyclic triples in `186,378` triangles, and `10,000` of `10,000`
sampled tournaments are acyclic. The mass being measured is an artifact of how the comparisons were
encoded, not a property of anyone's preferences.

This started as an argument and became a proof. Curl mass on a complete graph depends on a tournament
only through its score sequence, so enumerating Landau score sequences gives the exact minimum over
all `2^C(n,2)` tournaments at every size to nine: always `(n-2)/(3n)`, with the minimisers being
exactly the total orders. Tournament energies add, so the corpus's own design floor is exactly
computable. It comes to `0.21397613137732557` against an observed `0.2139761313773256`, which is
`2.78e-17` apart. On a second corpus the observed value and the floor are identical to the last digit.

**The registered threshold for calling this result interesting was `0.03`, which is `0.140` of the
floor. The prediction could not have failed.** That is the finding, and it is a finding about how the
prediction was written as much as about the corpus.

The write-up caught its own first-pass error, which is worth recording because it changes the claim.
Minimising curl alone let a cyclic orientation tie, because `338` of the tournaments are chordless
four-cycles with no filled triangles. Minimising the registered curl-plus-harmonic quantity, every
minimiser on every design is acyclic, and those `338` independently cross-check the pooled first
Betti number.

*Packages: `measure/composition/` (B1, B5).*

### Published leaderboard orderings do not survive inference, and six groups found it separately

Six independent groups working in five different statistical traditions across six benchmarks all
concluded, in 2026, that published leaderboard orderings do not survive inference. As far as five
search sweeps could establish, nobody had assembled them.

**Four of the six report both a numerator and a denominator, so the pool is k = 4 and not k = 6.**
That belongs here rather than in a footnote, because it is the number that governs how much the
pooled estimate is worth. Pooled by random-effects meta-analysis over those four, the fraction of
adjacent leaderboard ranks that are not resolved is `0.487`, with a confidence interval of
[`0.344`, `0.632`] and a prediction interval of [`0.205`, `0.777`].

**The confidence interval excludes 25%. The prediction interval does not.** Reporting both is the
entire point: the first says the mean effect is real, the second says what to expect from the next
benchmark, and a field that reports only the first will be surprised by the next benchmark.

**And the limit travels with it.** At k = 4 the between-study variance comes back at exactly zero,
which a careless write-up would report as "the six studies agree" and which at this k mostly means
the estimator cannot see heterogeneity: Cochran's Q has well under a third of the power it would
need. So the honest reading is that four independent measurements of the same quantity land close
together, and that four is not enough to say whether the field is homogeneous.

*Packages: `stats/meta.py`, `reward_lens/experiments/x8_leaderboard_meta.py`.*

### The canonical maths verifier has no test suite that can fail it

Pointed at the equivalence checker a large part of the reinforcement-learning-with-verifiable-rewards
literature scores against, mutation testing found that **not one of sixty mutants is killed by `240`
real gold answer pairs.** The other three scoring programs tested kill `2`, `24` and `26`.

A mutant that no rollout kills is a way the verifier could be wrong that no corpus in use would
reveal. Sixty of sixty is not a rate, it is an absence.

Two things found alongside it. The checker carries three bare `except:` handlers in `152` lines, so
an exception anywhere inside it becomes a score rather than an error. And on three of the four scoring
programs a random subsample of the corpus reaches **exactly** the same branch coverage as the whole
corpus, with the fourth three points short (`0.653` against `0.684`). That is a statement about corpus
redundancy rather than about coverage: on one subject a sample of 5 rollouts out of 240 reaches
everything the full corpus reaches.

Separately, a widely used software-engineering benchmark **decides whether a missing test result
counts against a patch by which repository the instance came from**, and its two evaluation types
disagree on `120` of `240` rollouts.

*Packages: `verifier/` (D1, D2, D9), `measure/card/`.*

### The one complete object in the ecosystem is opt-in, and nobody enables it

One framework's trajectory-step stream is the only place in the ecosystem where token ids, per-token
logprobs, routing, reward and advantage already coexist in one serialisable object. That makes it
the ecosystem's one complete object.

**It is opt-in on disk and nobody turns it on**: `0` of `37` published run manifests, `0` of `697`
published rollouts.

*Package: `tap/adapters/verifiers.py`.*

### Labs publish rollouts with rewards, and reward code with gates, and never both

Established by searching rather than by assuming: twelve dataset queries across `44` datasets and
four code searches. No public per-rollout record pairs rewards with the reward code's own hard gates.

This matters because it is what blocks a whole class of measurement. A hard reward threshold is a
sharp regression discontinuity with a deterministic, perfectly measured assignment rule, which is the
ideal case for a literature that usually has to work much harder. The instruments exist and the
subject does not.

*Package: `measure/threshold/`.*

### Reading a panel of reward models as a panel, rather than as a vote

Twelve reward models read as a majority vote give `15` violations of the generalised axiom of
revealed preference. Read as a panel, the same twelve give `19,198`.

**The vote hid the disagreement it was a vote over.**

*Package: `measure/composition/` (B2).*

---

## Results about our own instruments

These are here because a library that only published its wins would be asking to be trusted rather
than checked.

### A reliability-growth interval that under-covers, with a measurable repair

The Crow-AMSAA interval is the standard tool for asking whether a defect-discovery process is
converging. On a homogeneous Poisson process with the growth exponent equal to one **by
construction**, so that the true answer is known:

- Nominal-95% coverage is `0.264` to `0.402`.
- It calls the process converging in `0.469` to `0.552` of runs.
- Least squares is biased low, at worst `0.914` against a truth of 1, while the Crow estimator is
  unbiased at `0.992` to `0.999`.

**And the repair is measurable**: widening by 5.73x to 7.44x restores nominal coverage. At that
width, one real framework's confident [`0.191`, `0.670`] becomes [-`1.122`, `1.982`] and its converging
verdict evaporates. On a second real log the two estimators land on opposite sides of one.

*Package: `verifier/growth.py` (D6).*

### A density test that is wrong in centre and in scale on a real subject

Pointed at a real length density with no gate present, the McCrary-style discontinuity test returns
absolute z between `50.4` and `76.3`. Its own smooth-density null comes back centred at `-23.2` with
spread `2.2`, where the asymptotics imply `0` and `1`.

**So the estimator is wrong in centre and in scale, not merely noisy.** Standardising against the
measured band leaves the statistic `24.3` standard deviations out, which makes the null a detector
rather than a correction. Restricted to a range where the asymptotics hold, the same test gives
absolute z at most `1.18`.

The first write-up of this said the baseline rescues the reading. That was wrong and was corrected
in all four places. The instrument now refuses and names the restriction.

*Package: `measure/threshold/` (I1, I2).*

### The rung-2 Fisher is not usable at this scale, found twice from opposite directions

Two packages converged on this independently, which is why it is reported as a property of the
estimator rather than as one package's note.

With far fewer rollouts than parameters, the empirical Fisher has rank at most the rollout count,
every measured feature lies in the span of the scores, the unreachable-variance term collapses onto
the regulariser, and **reported heritability is one minus the damping rather than a property of
anything.** One package measured heritability at `0.99135` to `0.99143` against a damping of `0.0087442`,
and one minus that damping is `0.99126`, so the two agree to three decimals rather than exactly. The
other proved the undamped plug-in is **exactly** the observed covariance, and measured heritability
at `0.99999` as the damping went to zero.

Both added a degeneracy guard. Without one, the rung-2 fit would have been claimed as an independent
test that it is not.

*Packages: `measure/indices/heritability.py`, `measure/efficiency/`, `measure/reconcile/`.*

### A recorded no-go, which is a successful spike

The score-function estimator for the selection covector was the largest technical risk in the design,
so it was spiked before anything depended on it, with thresholds registered in advance: relative
standard error below `1.0` **and** split-half cosine above `0.5`.

Measured: `1.075` and `0.145`. Both fail on every arm. The best relative standard error over all five
arms is the `1.075` already quoted, and the arm with the best cosine reaches only `0.4515` at a relative
standard error of `1.097`, so the verdict does not depend on which arm was called registered. The differentiable
surrogate, the spike's own must-beat comparator, wins by `51x`. Top-eigenvector overlap is `0.357`
against a chance level of `0.290`, so the leading direction is near-indistinguishable from random.

**One variance-reduction technique actively hurt**: antithetic sampling took the cosine from `0.451`
to `0.145`, structurally rather than by tuning, because inverse-CDF pairing is negatively correlated
only at the first token. That is not carried forward.

The instrument ships as the differentiable-surrogate case. **A spike that returns no-go, changes the
plan, and is recorded is the spike working.**

*Package: `measure/frontier/covector.py`.*

### The first certified reference material says single-seed plants are not fine

Twelve micro-organisms over four doses and three seeds, plus three stability checkpoints, `649`
seconds on CPU. Characterisation uncertainty `0.04266`, homogeneity `0.1034`, stability `0.07766`,
combined `0.13617`.

The kill condition for this instrument was that homogeneity would turn out to be negligible, in which
case planting one seed would be fine and the instrument would be decoration. **It does not fire:
homogeneity is the largest of the three terms at `2.4x` characterisation.** Planting a single seed
and quoting a nominal dose understates the uncertainty by the largest term in the budget.

*Package: `measure/selection/`, `organisms/dose.py`.*

### A detector that used the future to define normal

The selection-strength summary was registered with a prediction that it moves **before** the labelled
hack rate does. On a real reward-hacking run of `25,664` rollouts over `401` steps, the registered
metric returned a clean lead of `+3.97` transition widths.

**The frozen analysis found, on its own numbers, that the number is an artifact.** The registered path
standardises the series against a mean that the post-transition regime itself raised, so early points
sit below a threshold defined partly by the future and the accumulator crosses almost immediately.
**A detector that uses the future to define normal cannot measure a lead time, and it will always
report one.**

The detector-free comparison, the same fit applied to both series, gives a **lag** of `0.93`
transition widths. Both numbers are published, the frozen metric is left as registered, and the
acceptance test asserts the disagreement rather than quietly adopting the one that reads better.

The underlying summary is real on this series: `0.0371` with an interval of `[0.0128, 0.0789]`
against a permuted-step null at `p = 0.001`.

*Package: `measure/ledger/` (F1, F2).*

### E-values do not multiply, and a conjunction detector that was rigged by its own threshold

The original design justified building a conjunction detector out of e-values on the grounds that
e-values multiply legally under arbitrary dependence. **They do not.** Demonstrated with three
perfectly dependent valid e-values whose product has expectation above ten. The arithmetic mean is
the merging function valid under arbitrary dependence.

The instrument needed no merging rule at all, and that is the more useful half: a conjunction fires
only when every channel fires, so the joint false-alarm rate is bounded by the smallest channel's by
containment, with no dependence assumption anywhere. The original design reached for a theorem to
justify something that follows from the definition of an intersection.

**And the conjunction's kill condition fires on five of six designs** once the comparison is matched
on realised false-alarm rate, because a common threshold rigs it. Without a clean reference scale the
honest recommendation is to watch the best single channel.

One number from the same package that makes the case for anytime-valid inference in one line: **a
fixed interval read every step is wrong on `38.5%` of streams against its advertised `5%`.**

*Package: `monitor/` (J1-J5).*

### Two wrong numbers on default paths, and a detector nothing obliged anyone to run

A fresh-context review of the record layer confirmed the schema is structurally right, with the
tiling invariant matching an independent predicate over `68,344` cases with zero mismatches. It also
found two wrong numbers on default paths that had been shipping since the record landed.

The policy-ratio clip was applied as a bound on the advantage, which **pinned `400` of `400` groups
to exactly `0.2`**, so every counterfactual read as "nothing moved". And the record divided by the
population standard deviation where every framework in scope applies Bessel's correction.

**The sharpest part is not either defect. The estimator module had already documented both of them
correctly, with the numbers, and shipped a detector for them, and nothing obliged a caller to run the
detector before reading the value.** A defect recorded only in a docstring is a defect nobody
scheduled.

The record now replays a real trainer at `2.97e-06` over `400` groups, which is the float32
round-trip and nothing else.

*Package: `record/`.*

### A quantity built from round-off

One optimiser telemetry channel spans **six float32 units in the last place across `200` steps.** A
naive constancy check passes it, and standardising it yields z-scores built entirely from round-off.

*Package: `monitor/`.*

### Correcting the wedge's flagship number, and checking what it moved

The effective group size was multiplying a grader property by a reward-distribution property. Both
factors are computed from the same contaminated observations, so measurement noise entered twice.

The correction moved the number across eleven real reward models: rung 0 from `2.9859` to `4.0000`,
rung 3 from `1.9097` to `2.5582`. Kish's shape factor now travels beside the reading rather than
inside it, because it is a statement about the reward distribution's shape and not a property of the
grader.

**The question that was frozen in advance was not whether the number would change, but whether
anything published on the strength of it would.** A kill condition that starts firing because a
quantity was corrected is a much larger result than a corrected quantity. It does not: still zero of
eleven graders overlap, and the margin **widened** from `0.3847` to `0.5484`.

*Package: `measure/metrology/` (A1).*

### The library's worst-looking published result was mislabelled, and the real number had never been measured

2.0.1 published a maximum absolute recovery gap of `0.419` between planted organisms and real models,
against a registered threshold of `0.15`, and treated it as an embarrassment to be caveated.

**It is not an embarrassment and it is also not what it was called.** Recomputed from its own parent
row to nine decimal places, the `0.419` is real. But both of its arms are planted: the count of natural
corpora in it is zero. So it is a simulation-to-real-**model** transfer coefficient, and the
planted-to-real coefficient the catalogue says it is, the one that says whether an instrument
calibrated on organisms means anything on a naturally arising corpus, **had never been measured at
all.**

Measured against 25,664 real labelled rollouts, it is `0.4732` with an interval of [`0.4543`, `0.4933`].

**And the point is not that number, it is that the number moves with a design choice nobody records.**
Under a planting design that appends the planted behaviour, the coefficient is `0.4732`. Under one that
substitutes it, which is what the policy in the real corpus actually did, it is an order of magnitude
smaller. **A transfer coefficient quoted without its organism design is not yet a measurement**, and
no published transfer coefficient in this field quotes one.

This is the result the introduction to this document is about. The library's worst-looking number was
worth publishing as a quantity with a method rather than hiding as a caveat, and doing so turned up
the fact that the field's version of it does not exist.

*Packages: `measure/labels/`, `core/reference.py`.*

---

## The predictions

Ten of fourteen registered predictions have resolved. **Four resolved against the prediction**, one
was a spike that returned no-go, and one was refuted because the person who wrote it had the premise
wrong by an order of magnitude in the wrong direction.

Rows still open: one needs three held-out runs this build does not have; two resolve against studies
that are deliberately written, priced and never run; and one resolves against work not yet released.

Two of the refutations are worth reading for the same reason, which is that **a prediction can be
unfalsifiable for arithmetic reasons and nobody notices until it resolves.**

The selection-gradient prediction registered a required improvement of `0.10`. The baseline it had to
beat already correlated with the outcome at `0.9643`, so **the maximum achievable improvement was
`0.0357`**, and on a second model the ceiling was exactly zero. The threshold was unreachable when it
was written. The gradient was also worse, which is an independent second refutation, and the reason
is structural: on a fixed bank every direction is reachable, so the two operators coincide identically
and **the forecaster is the thing it was being compared against.**

The intransitivity prediction registered a threshold of `0.03` against a design floor of
`0.21397613137732557`, which is `0.140` of it. (`0.238` is the floor of one slice of that corpus, a
complete graph on seven items, and not the corpus's own.)

Both are facts about how the predictions were written rather than about the instruments, and both were
found by resolving them rather than by reviewing them.

---

## The caveat that governs this whole document

**Several results here closed on a subject that could not have failed them.** This is not a
list of excuses at the end; it is the single most important thing in the document, and each result
above carries its own version of it in the same paragraph as its number.

The sharpest instance found itself. The four-book reconciliation closes on the 200-step run at
`1.082`, `1.097` and `1.042`, which reads as a result. It is not one yet. Monte Carlo uncertainty is
**`99.9999999999%` of the composed variance** on every one of the six feature-run pairs, and the
closure test's own detection floor is **between five and six orders of magnitude above** the
first-order prediction it is arbitrating. On the worst pair the floor is `3.760` against a predicted
5.647e-06, a ratio of 665,862; the median ratio over the six pairs is 854,374. Five of nine
budget terms were computed and four are named absent, so **no verdict from this subject can
distinguish an unmodelled term from an unmeasured one.** Both verdicts were reported, the flattering
one was declined, and the reason for declining it is stronger than either verdict.

The same shape appears three more times. The selection-gradient prediction was refuted on a bank where
the two covariance operators coincide identically. One exploit log refuses outright because the
statistic it needs has a zero denominator. And four instruments in the retrofitted battery have a real
quantity and a real label and **no grader**, so their only offline signal is a randomly initialised
classifier at `0.5568` against a TF-IDF baseline's `0.6892`.

**None of these is a defect in an instrument. Each is the instrument correctly reporting that the
subject cannot answer the question, which is what the architecture is for. What would be a defect is
reading any of them as a measurement.**

### Subjects this build does not have, named rather than absorbed

- An agentic record with tens of turns and per-token logprobs. The fixtures carry two turns and
  **zero tool calls across `1,600` trajectories**, asserted.
- A live judge, so one shipped result is neither confirmed nor refuted.
- A trained reward model on a preference set with declared lineages.
- A run scored by a genuinely **composed** grader: no fixture has more than one score leaf, so one
  registered quantity has never been exercised on real data.
- A public per-rollout record whose hard gate has a **continuous** running variable, which was
  established not to exist by searching rather than by assuming.

The two reinforcement-learning records this build does have are real optimisation traces and **not**
reward-hacking transitions: the model is `2.45M` parameters over two layers, trained against a length
grader. Anything in this document about a transition is measured on a separate labelled series of
`25,664` rollouts over `401` steps, and says so.

---

## How to check any of this

Every number above came from code that ran, and most of them carry an evidence id in the store the
run wrote. The library has a gate in continuous integration that scans every tracked page for numbers
not bound to a stored measurement, and this document runs against it with no baseline at all, where
every other page has a backlog a ratchet holds.

**That gate passes on this document and you should not take much comfort from it.** It masks inline
code spans before it looks for numbers, on the reasonable ground that a number inside one is usually
a literal rather than a claim. Measurements here are written in code style, which is the convention
every page in this project uses. So the gate reports zero unbound numbers and zero claims checked,
and the second half of that sentence is the honest one.

The obvious repair is to bind each number to its evidence id, and it does not work today for a
structural reason worth naming rather than leaving as an omission. The check runs in continuous
integration against a fresh checkout, where the default evidence store is empty, so **every `ev:`
reference would fail to resolve and the job would be red on correctly-bound numbers.** The evidence
exists, in the per-experiment stores published with the runs, and pointing the checker at them resolves every
claim; the job does not point at them. Until it does, binding a number here would make the document
fail for being right. That is a change to the job in a later release, not a change to this page.

What actually checked this document was a person re-deriving its numbers from the stores and the code
that produced them, and **it found six wrong ones**, three of which were sitting inside code spans
where the gate is structurally unable to look. A calibrated interval was transcribed with both
endpoints wrong. A combined uncertainty was off in the fifth decimal. A quoted best-case arm was a
value that appears nowhere in the data it was attributed to. A ratio of two numbers was reported as
their quotient when it was the median of six such quotients. A pooled estimate over four studies was
described as pooling six. And a design floor belonging to one slice of a corpus was quoted against
the whole of it, in a document that gives the correct floor forty lines earlier.

Those are corrected above. The reason they are described here rather than quietly fixed is that they
are the argument: **a gate that works exactly as written and enforces nothing is the failure mode
this project has now found five times**, and this document's own check is the fifth. Every one was
found by somebody using the gate rather than testing it, and the only thing that has ever caught this
class of error is a second person recomputing the number from the thing that produced it.
