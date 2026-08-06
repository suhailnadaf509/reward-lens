# The measurement discipline

What does it take to trust a number about a reward model? Not a disclaimer, and not good intentions. It takes machinery that refuses to hand you a number dressed as more than it is. That machinery is the most original part of this library, and this is where it is written down in full.

Everything up to here has been at working altitude: load a model, run a tool, read a plain-words result. This section drops the floor out. If the [concepts airlock](../concepts/measurement-you-can-trust.md) told you a measurement arrives with a receipt, this is where you read every line on the receipt and learn what had to be true for each one.

## One kernel, three gates

The library is a small kernel of subsystems, a layer of studies that consume it, and three gates that hold the whole thing honest. The kernel is what every measurement stands on. You rarely touch all of it. You reach for the part a question needs, and the layers stay lazy, so the epistemics core pulls only numpy and nothing loads torch until you open a model.

![The kernel: signals and runtime at the base, the instruments above, studies and artifacts on top, and the three gates in the foundation.](../assets/figures/kernel-map-light.svg#only-light){ .rl-fig .rl-fig--wide }
![The kernel: signals and runtime at the base, the instruments above, studies and artifacts on top, and the three gates in the foundation.](../assets/figures/kernel-map-dark.svg#only-dark){ .rl-fig .rl-fig--wide }

/// caption
**One kernel underneath, so every study on top is thin.** Signals and runtime at the base, the instruments (measure, interventions, geometry, concepts) above them, studies and artifacts on top. In the foundation sit core, stats, and the evidence atom, and the three gates that compute trust.
///

The three gates are the spine of the whole section:

- The **calibration gate**. A tool with no scorecard, meaning no measured performance against a case whose answer is known, cannot claim more than exploratory trust. It earns calibration on model organisms with structure planted by construction.
- The **gauge gate**. A quantity that only means something in a fixed basis, a direction or an angle or a subspace, cannot be compared across models without a shared frame. Ask for that comparison without fixing the frame and the library raises, rather than hand back a coordinate change dressed as a real one.
- The **registration gate**. A confirmatory claim requires a frozen preregistration. A study is a spec plus a thin analysis function, and freezing it stamps the git commit and locks the predictions before the run.

## The section

- [The anatomy of evidence](anatomy-of-evidence.md). Every field on the receipt, and why a bare float was the bug that made the old library unsafe.
- [The trust ladder](trust-ladder.md). The four levels, what computes each, why the caller cannot set trust, and what each level licenses you to say.
- [Calibration and organisms](calibration-and-organisms.md). Answer keys by construction: plant a rule, verify it governs behavior, score every method against it, publish the scorecard. The floor the calibration gate stands on.
- [Gauge and frames](gauge-and-frames.md). Why two reward directions can look orthogonal in raw coordinates when they are nearly the same function, and what a shared frame does about it.
- [Generalizability theory](generalizability.md). How much of a score is the response and how much is which judge you drew on which day, answered by a decomposition that has existed since 1972 and has never been pointed at a reward model.
- [Studies and preregistration](studies-and-preregistration.md). A spec, checkable predictions, kill criteria, a freeze, and a runner that adjudicates against the prediction instead of the author.
- [The evidence store](evidence-store.md). The append-only record everything lands in, and the artifacts that are views over it, a card or a leaderboard or a safety case that can only say what the store can back.

Read it in order or drop into the one you came for. None of it holds back.
