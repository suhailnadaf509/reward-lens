# How-to guides

**You know the job and you want the exact calls.** Each recipe is one page, one task, and code you can paste. The runnable ones use the tiny CPU signal, so they finish in under a minute with no download; anything that needs an 8B model is marked and names the call rather than pretending to run.

If you want to understand why the calls are shaped the way they are, the [instruments](../instruments/index.md) and [concepts](../concepts/index.md) are the longer read.

## Point reward-lens at your grader

<div class="grid cards" markdown>

-   __[Load or wrap a reward model](load-a-reward-model.md)__

    Wrap a model already in memory, or build a tiny one on CPU. The hub loader is gated and says so.

-   __[Use a DPO checkpoint as a signal](dpo-implicit-reward.md)__

    A DPO policy is a reward model with no head. Read its implicit log-ratio reward.

-   __[Use an LLM judge as a signal](llm-judge-signal.md)__

    Read a verdict off the unembedding, and validate that the judgment lands where you think.

-   __[Score with a process reward model](process-reward-model.md)__

    One reward per reasoning step. The delimiter split is exact; the learned detector is a stub.

-   __[Write an adapter](write-an-adapter.md)__

    Make a model family reward-lens does not ship support for speak the signal protocol.

</div>

## Read an instrument

<div class="grid cards" markdown>

-   __[Detect length bias](detect-length-bias.md)__

    Run the bias battery across axes and read a standardized effect size with an honest sample size.

-   __[Attribute a reward score](attribute-a-score.md)__

    Split a margin into per-component contributions, and know why that is only the first look.

-   __[Patch without running out of memory](patching-memory.md)__

    Triage with the cheap tools, then patch only the components that matter.

-   __[Compare two reward models](compare-two-models.md)__

    See where two models form the same preference, and why raw numbers need a shared frame first.

</div>

## Trust the number

<div class="grid cards" markdown>

-   __[Effective sample size of an eval set](effective-sample-size.md)__

    Thirty rows from six seeds are not thirty data points. Count what the data actually earned.

-   __[Effective group size of a GRPO group](effective-group-size.md)__

    Sixteen rollouts is not sixteen independent observations once the grader disagrees with itself.

-   __[Calibrate a detector on an organism](calibrate-on-an-organism.md)__

    No instrument earns more than exploratory trust without an answer key to check it against.

-   __[Freeze and run a study](freeze-and-run-a-study.md)__

    Write the prediction down first, freeze it, then let the spec adjudicate the result.

-   __[Build a card, check a manuscript](cards-and-claims.md)__

    A card can only say what the evidence store can back. A claim check fails on an unbacked number.

</div>

## Wire a training loop

<div class="grid cards" markdown>

-   __[Hook into a training loop](training-loop-hooks.md)__

    Attach a reward function and log the geometry as optimization pressure builds.

</div>
