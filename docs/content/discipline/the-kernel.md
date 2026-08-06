# The kernel

Most of this library is instruments. Underneath them is a small set of types that every instrument
shares, and those types are the part that is hard to see from the outside. They are also the part
that decides what the library will and will not tell you, so they are worth twenty minutes.

## An assay

The word this project organises itself around is borrowed rather than invented. **An assay is a
quantitative test of what is actually in a material, run against a certified reference, reported
with an uncertainty and a stated limit of detection.** Every clause of that does work.

*Quantitative test of what is actually in it*, not of what the label says. An assay is what you run
when the thing in front of you is a sample and the question is composition, and the reason to run
one is that you do not already know the answer.

*Against a certified reference*: a material whose own value has been established, and whose own
uncertainty has been measured and published alongside it. Calibrating against an uncertified
reference is the mistake that makes a whole laboratory's output uninterpretable, because the
reference's error goes into every reading and nobody can say how big it was.

*With an uncertainty*: not a point value. A number with no uncertainty attached is not a
measurement, it is a reading off a dial.

*And a stated limit of detection*: below some effect size the instrument cannot tell the sample
apart from a blank, and the honest thing to do with a difference below that limit is to decline to
call it a difference. Analytical chemistry has had this discipline for decades. Machine learning
reports effect sizes below its own substrate noise routinely, and has no vocabulary for saying so.

That is what these types are for. The rest of this page is the vocabulary, in the order it stacks.

## A quantity is not an estimator

A **quantity** is what you want to know. An **estimator** is one way to get it, at a stated access
level, with a stated bias, at a stated cost. Conflating them is what forces a library to have one
architecture per access profile.

Separating them buys three things. A closed lab gets a real answer at the cheapest rung with an
honest bias direction, rather than a refusal. Two labs' claims become comparable, because a reading
at rung 0 and a reading at rung 3 are two claims about one quantity and the ladder says which is
which. And when two rungs disagree on the same data, that disagreement is not an embarrassment: it
is the cheap method's transfer uncertainty, measured, and it composes into the calibration chain.

A quantity registered with no estimator is an open research target rather than a bug, and this site
[names every one of them](../catalogue/open.md) instead of quietly omitting them.

### `Unit` is three axes, and it refuses

A unit here is not a string label. It has three axes, because they fail independently:

- `dimension`, what is being counted: `nats`, `count`, `probability`, `correlation`, `reward`.
- `per`, what it is counted over: `token`, `sequence`, `step`, `group`, or nothing at all for an
  extensive quantity.
- `scale`, the convention it is expressed in: `nats`, `bits`, `raw`, `normalised`.

A per-token KL and a per-sequence KL are different quantities, not one quantity in two outfits.
`Unit.compatible_with` returns `False` between them, and the comparison raises `UNIT_MISMATCH`
rather than converting. The conversion is arithmetically available, and it is still not done
silently, because the factor is a property of your data (how many tokens?) rather than of the unit.
Doing it for you would require information the comparison does not have.

One axis can read `OPEN`, meaning the decomposition has not been settled. A unit with an undecided
axis is incomparable with everything, **including another undecided one**. Unknown is not a value.
Before that rule existed, every quantity whose decomposition nobody had settled compared cleanly
against every other, so a score tree compared cleanly against a confidence interval, and it looked
fine.

## Four typing dimensions

Whether an instrument can run at all is four separate questions, and they fail in different ways.

**Access: what can I touch?** Eight components (`TASK`, `GRADER`, `POLICY`, `ESTIMATOR`,
`OPTIMIZER`, `ARTIFACT`, `GOLD`, `RECORD`) crossed with eight flags. The flags are `RECORD` (read
what was logged), `QUERY` (call it again on inputs of my choosing), `REPLICATE` (call it again
under controlled facet variation), `FORWARD` (run it and read activations), `BACKWARD`
(differentiate through it), `SOURCE` (read its code, control-flow graph and tests), `MUTATE`
(patch, ablate, edit, recompile) and `CONTROL` (stand up a counterfactual arm of the whole loop).

`REPLICATE` does not follow from `QUERY`, and that is the member that earns its place. A hosted
judge with a fixed internal seed is callable and not facet-varyable, and without facet variation
there is no variance decomposition, no effective group size and no attenuation factor. Collapsing
the two would silently delete half of the metrology series for anybody behind an API that will not
take a seed.

**Phase: when am I?** `PRE_RUN`, `IN_RUN`, `POST_RUN`, `DEPLOYED`. Asking an in-run question of a
finished artifact is not a hard case, it is a category error, and a phase you have already passed
cannot be revisited.

**Substrate: what kind of thing is the grader?** `NEURAL_SCALAR`, `NEURAL_GEN`, `PROGRAM`,
`PROCEDURAL`, `HUMAN`, `COMPOSITE`. Asking a program for its activations is the same kind of error.
The interesting consequence runs the other way: a program has source, and a network does not, so
the verifier instruments can ask questions of a grader that no white-box instrument can ask of a
model.

**Regime: are the estimator's assumptions live?** This is the one that matters, and it gets its own
section, because the first three fail loudly and this one fails quietly.

## The validity envelope

Access, phase and substrate tell you whether an instrument is *available*. The envelope tells you
whether its answer would *mean* anything. The gap between those two is where confident wrong
numbers come from.

Take the selection term in the Price ledger. It needs a record and a featuriser and nothing else,
so it will compute happily on any run ever logged. But it is a first-order expansion, so it means
nothing if the step is large. It assumes the group has spread, so it means nothing on all-fail
groups. It assumes the advantage transform is the one you think it is, so it means nothing if the
estimator z-scores. And it assumes one generating policy per trajectory, so it means nothing under
partial rollouts. Four ways to get a confident wrong number, and access can see none of them.

An `EnvelopeSpec` names the conditions an estimator depends on, drawn from twelve that are each
measurable from a record: `QUASI_STATIC`, `LINEAR_RESPONSE`, `GROUP_NONDEGENERATE`, `NEAR_POLICY`,
`STATIONARY_GRADER`, `EXOGENOUS_CURRICULUM`, `NO_COMPACTION`, `ABOVE_LOD`, `ESS_ADEQUATE`,
`LIGHT_TAILED`, `SCALAR_REPRESENTABLE`, `MASK_STABLE`.

Three rules are enforced at construction rather than by review, so an envelope that cannot be
enforced cannot be built:

- An empty `requires` fails unless the author passes `unconditional=True` **with a justification**.
  Almost nothing is unconditional, and an empty envelope is far more often an author who has not
  looked than one who has.
- Every condition in `requires` must appear in `measured_by`. A declared precondition that nobody
  can check is worse than no precondition, because it reads as rigour and enforces nothing.
- An id in `measured_by` that resolves to no registered quantity fails too. Appearing in
  `measured_by` was the whole check that a precondition is measurable, so a name that resolves to
  nothing passes the check and measures nothing.

And unknown is not a pass. `EnvelopeSpec.admits` returns `False` for a condition that could not be
determined, because a check that did not happen reading as a check that succeeded is the entire
failure this module exists to prevent.

### Three behaviours, and they are not interchangeable

When a condition fails, `on_violation` says what happens. There are three, and choosing between
them is a real decision about the instrument rather than a style preference.

`refuse` returns a `Refusal`. The quantity is not estimable here and no number comes back. This is
the default and it should be, because most estimators outside their envelope are not weakly wrong,
they are meaningless.

`bound` falls back to a named weaker estimator that survives outside the envelope, and the reading
comes back labelled a bound rather than a point estimate. The type will not let you declare `bound`
without naming the fallback: a promise with nothing behind it is not a policy.

`downgrade` keeps the quantity defined and caps its trust. The worked case is a before-and-after
comparison outside `STATIONARY_GRADER`: the comparison is still computable and it is now
exploratory rather than registered, with the violated condition recorded on the reading. Use this
only when the quantity really does survive, in a weakened form, outside its envelope. Reaching for
it because a refusal felt unhelpful is how the envelope becomes decorative.

## Every reading is `Evidence` or a `Refusal`

`Reading = Evidence[Any] | Refusal`. That is a real union under the type checker, so annotating an
instrument `-> Reading` and returning a float is an error the type checker catches.

A `Refusal` is a value, not an exception. It is never a `None`, never a zero, and never a silent
fall back to a worse estimator. It carries the reason, the statistics that produced it, and a
remedy, and it cannot be constructed without the remedy, because a refusal with no remedy is a tool
that looks broken instead of a tool that looks careful.

Refusals and exceptions stay different by use. A refusal is for a condition the instrument
anticipated: insufficient access, a violated envelope, an effect below the limit of detection, an
uncertified reference. An exception is for a condition it did not: a corrupt file, a shape
mismatch, a bug. Catching a broad exception and returning it as a refusal is exactly the
`except Exception: score = 0.0` pattern that one of the instruments in the catalogue exists to
count.

There are sixteen reasons, and [each one has a page section saying what to do about
it](../refusals.md).

## Invariance groups

Causal abstraction, the framework underneath activation patching, causal scrubbing, distributed
alignment search, concept erasure and sparse dictionaries alike, says nothing at all unless the
class of admissible reparameterisations is stated. Without one you can map a randomly initialised
model onto a named circuit with perfect accuracy and the framework has no way to object.

So every instrument declares a relation to a group, and the declaration generates a property test
the author did not write. Seven groups, plus the trivial group that a deliberate answer of "none"
resolves to:

| Group | Acts on | What the assertion is |
|---|---|---|
| `reward.affine` | scores | invariant, or covariant with a stated power |
| `reward.null` | scores | advantages unchanged, and everything downstream of advantages |
| `repr.basis` | activations | invariant, covariant given a shared frame, or raw-only |
| `policy.reparam` | parameters | Fisher-metric quantities unchanged; parameter norms are not |
| `tokenization` | tokens | declare a normalisation and be invariant under it, or refuse |
| `group.permutation` | groups | any group statistic unchanged |
| `units` | any | a cross-unit comparison raises rather than converting |
| `trivial` | nothing | no group acts on this, declared rather than omitted |

Three things about this are easy to get wrong and are typed rather than documented.

A group does not have a status; an instrument's *relation* to a group does. `repr.basis` admits all
three of invariant, covariant and raw-only, and which one applies depends on the instrument.

A covariant instrument does not return the same value. It scales by a stated power of the group
parameter, so the check is that the reading scales correctly, not that it is unchanged. A checker
that only knows how to assert equality silently forces every covariant instrument to declare itself
invariant, and then passes it.

A failure is not always a defect. An instrument that fails `group.permutation` is sensitive to
rollout order, and for a judge that is position bias, measured. The report says so rather than
reading as a bug, because the test detecting position bias is the test working.

## Reference materials, and the chain

Every earlier version of this project assumed the answer key was right. It is not.

A laboratory does not calibrate against "a sample somebody prepared". It calibrates against a
**certified reference material**, which ships with an assigned value and an uncertainty on that
value, decomposed three ways: `u_char`, how well the assigned value was established;
`u_bb`, whether two independent preparations agree; and `u_stab`, whether the value drifts.
The three compose as `u_CRM² = u_char² + u_bb² + u_stab²`.

Each has an exact analogue here and none of the three is routinely measured anywhere in this field.
`u_char` is how well a planted rule's strength is known, and a plant at a nominal dose is a nominal
dose rather than a measured one. `u_bb` is whether two plants with different seeds give the same
answer. `u_stab` is whether the plant drifts as the host is finetuned further.

The rule that earns `ReferenceMaterial` its place is one line, and it is enforced in the trust
computation rather than written on a caveats page: a missing homogeneity term is not a missing
field. It renders in every downstream reading as "reference uncertainty not characterised" and it
caps the trust ladder.

This bites hardest on labelled corpora, which are reference materials whether or not anyone calls
them that. A corpus whose label error rate nobody has published is an uncertified reference, and
scoring against it measures the labels. That is what `LABEL_QUALITY_UNKNOWN` is for.

Calibration chains compose, and the composition is checked. Reading a production number calibrated
through an intermediate against a reference means the uncertainties add in quadrature along the
chain, and a chain with a missing link is a `REFERENCE_UNCERTIFIED` refusal rather than a silent
pass.

## The budget, and the limits

The last two types answer different questions. The budget answers "how wrong is this number, and
which part of the apparatus is responsible?" The limits of detection answer "is this number
distinguishable from the substrate's disagreement with itself at all?"

**The budget is a table, not an interval.** A confidence interval is one number that has already
thrown away the thing worth knowing, which is which term dominates. The GUM, the Guide to the
Expression of Uncertainty in Measurement, formalises the alternative: enumerate every contribution,
state for each whether it was evaluated statistically (Type A) or by judgement (Type B), give each
a sensitivity coefficient, and compose in quadrature.

Keeping the Type A and Type B split matters even though both are treated identically once they are
standard uncertainties, because the split records *how you know*. "Type B, rectangular, half-width
from the vLLM-versus-HuggingFace residual comparison" is auditable. A pooled interval is not.

The payload is the last line of the table. The observation this whole apparatus exists to make is
that **the largest term is almost never sampling noise**, and a budget that cannot say so is not
doing its job.

**The limits of detection turn a noise floor into a decision rule.** Analytical chemistry has used
`LOD = 3.3 * sigma_blank / S` for decades, where `S` is the slope of the calibration curve of
reading against dose. Three outcomes rather than two:

- Below the limit of detection: refuse. This is not a negative result and must not be written up as
  one.
- Between the limit of detection and the limit of quantification: return a bound. Detected, not
  quantifiable, and a point estimate here would be false precision.
- Above the limit of quantification: report the value with its budget.

`S` is not assumed. It comes from a fitted calibration curve over a planted dose sweep, and the
seam is deliberate, so that an instrument with no dose sweep cannot quietly invent a sensitivity of
one and report a limit it has not earned.

## Where to go next

[The anatomy of evidence](anatomy-of-evidence.md) is the other half of this: what comes back when
an instrument does not refuse. [The trust ladder](trust-ladder.md) is how the trust level on a
reading gets computed rather than asserted. And [every refusal](../refusals.md) is the page to open
when one of these types has just told you no.
