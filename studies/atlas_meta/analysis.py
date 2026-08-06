"""Atlas meta on the kernel: the universality study retyped, and what a record cannot answer.

`atlas_meta` is the one directory under `studies/` that ships two frozen studies rather than one:
`universality` (T13, the value convergence excess) and `performative` (T11, the audit half-life).
This module is the retype layer for the directory. It carries the `ScienceRetype` for the
universality study, which is the half whose destination is live: `grader.value_convergence_excess`
ships as `measure/indices/vce.py` and the study already calls its pure function, so the retype adds
the binding and the record path rather than a second index.

`analyze` and `build_spec` are re-exported from `universality` unchanged. Neither study's arithmetic
is touched by anything here, and both keep running under `reward_lens.studies.runner` exactly where
they did.

**The performative half is not retyped, and the reason is structural rather than a judgement about
the science.** One `ScienceRetype` carries one `StudySpec`, and the retype registry is keyed on the
directory, so a second study in the same directory has nowhere to declare a second retype that the
acceptance walk would accept. What T11 needs before it can bind is written into
`PERFORMATIVE_PENDING` below, as data rather than as a comment, so it can be read without grepping.
"""

from __future__ import annotations

from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component
from reward_lens.record.schema import Run
from studies._retype import MetricSpec, ScienceRetype, count_trajectories
from studies.atlas_meta.universality import analyze, build_spec

#: What the T11 half of this directory still needs. `half_life_obs` and `half_life_causal` are
#: audit half-lives in developer-response generations, which is `monitor.half_life`'s unit
#: (count/step) on a generation axis rather than an optimiser-step axis; `half_life_gap` is their
#: difference, the same unit. `monitor.half_life` is registered against catalogue letter I4 and no
#: shipped class declares it, so binding to it today would record a destination that nothing
#: measures. The third metric, `real_half_life_gap`, is gated on a real developer-response loop.
PERFORMATIVE_PENDING = {
    "study": "atlas-performative-halflife",
    "metrics": ("half_life_gap", "half_life_obs", "real_half_life_gap"),
    "candidate_quantity": "monitor.half_life",
    "blocked_by": (
        "monitor.half_life is registered with no instrument declaring it, and the generation axis "
        "an audit half-life is counted on is not the optimiser-step axis the unit names"
    ),
    "structural_blocker": (
        "one ScienceRetype carries one StudySpec and retypes() is keyed on the directory name, so "
        "the second study in this directory has no key to declare itself under"
    ),
}


# ---------------------------------------------------------------------------
# The retype: the universality study on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="atlas_meta",
    spec=build_spec(),
    headline="grader.value_convergence_excess",
    destination=(
        "grader.value_convergence_excess, which ships as measure/indices/vce.py and which the "
        "study already runs. All four frozen metrics resolve to that one id under four subjects, "
        "because three of them are the same excess on a different model pair and the fourth is the "
        "null tail probability the excess is read against. The T11 half of this directory, the "
        "audit half-life, is not bound: see PERFORMATIVE_PENDING."
    ),
    needs={Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="vce_convergent",
            quantity="grader.value_convergence_excess",
            arc="vce-convergent-pair",
            frame="frame-canonicalised",
            source="organism",
            note=(
                "the excess itself on the pair with a planted shared value subspace: the reward "
                "subspace alignment minus the matched capability alignment, both canonicalised in "
                "one frame. That is the registered definition line for line. It is source='organism' "
                "because the sign is only knowable against a planted pair, and because the pair is "
                "the thing a record does not contain."
            ),
        ),
        MetricSpec(
            metric="vce_sign_gap",
            quantity="grader.value_convergence_excess",
            arc="vce-null-pair",
            frame="convergent-minus-independent",
            source="organism",
            note=(
                "the convergent pair's excess minus the independent pair's, which is a difference "
                "of two readings of the same quantity and carries the same unit. Its own arc rather "
                "than its own frame on the first: the independent pair is a second construction "
                "with its own draw, and folding it into the convergent arc would hide the second "
                "alignment behind the first."
            ),
        ),
        MetricSpec(
            metric="reward_convergent_p_value",
            quantity="grader.value_convergence_excess",
            arc="rum-null",
            frame="identifiability-null-tail",
            source="organism",
            note=(
                "where the convergent pair's reward alignment falls in the RUM-identifiability "
                "null, the overlap two independently estimated utilities share for free because a "
                "random utility model pins a reward subspace's dimension and not its orientation. "
                "The registered definition names that null as part of the quantity, so the arc that "
                "samples it is this quantity's arc. The number is a tail probability rather than an "
                "alignment, which is why it is source='organism' and never stamped: the registry "
                "has no id for a p-value and inventing one is not a builder's call."
            ),
        ),
        MetricSpec(
            metric="real_vce",
            quantity="grader.value_convergence_excess",
            arc="production-pair",
            dataset="production-rm-pair",
            source="gated",
            note=(
                "the same excess on a production reward-model pair with matched capability probes. "
                "The index is unchanged and what is missing is two real populations in a shared "
                "frame, which is why this is a dataset on the same id rather than a second one."
            ),
        ),
    ),
    arc_requires={
        "vce-null-pair": ("vce-convergent-pair",),
        "rum-null": ("vce-convergent-pair",),
        "production-pair": ("vce-convergent-pair",),
    },
)


def read(run: Run) -> Reading:
    """Universality against a real training record: the excess is defined on a pair, a run has one model.

    VCE is align(reward subspaces) minus align(capability subspaces) for two reward models in a
    shared frame. Every term in it needs two of them. One training record carries one grader, so
    there is no second subspace to align against, no capability match to subtract, and no
    identifiability null to read the difference against.

    Scope limit, three lines in: this refuses on the record rather than on access deliberately.
    Granting FORWARD here would let a frame be fitted and one model's subspaces estimated, and would
    still leave a difference of two alignments with one alignment in it. The remedy has to name the
    second model, not the next rung, because the next rung does not help.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    grader = run.component(Component.GRADER)
    named = getattr(grader, "name", None) or "unnamed"
    n_traj = count_trajectories(run)
    return RETYPE.incomplete(
        field="second reward model to align against",
        subject=f"run {run.id}, whose only grader is {named!r}",
        remedy=(
            f"open a second record scored by the other model of the pair, or hold both score heads "
            f"and fit one frame across them; `measure.indices.vce.value_convergence_excess` then "
            f"takes the two alignment scalars and the (d, k) the null is sampled at. Matched "
            f"capability probes have to come with it, because the excess is a difference and a "
            f"reward alignment reported without the capability alignment beside it is the number "
            f"T13 exists to say is not enough. Adding access to {named!r} does not help: the "
            f"missing term is a model, not a rung."
        ),
        trajectories=n_traj,
        graders=[named],
        n_steps=int(run.n_steps),
    )


_ACCESS_REMEDY = (
    "open a run written by the recorder so the grader and its scores are readable. That is only the "
    "first half of what universality needs: the excess is defined on a pair of reward models, and a "
    "record of one training run carries one of them."
)


__all__ = ["PERFORMATIVE_PENDING", "RETYPE", "analyze", "build_spec", "read"]
