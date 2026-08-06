"""D7, the grader card: the composite artifact and the wedge product.

Nothing like it exists. A marketplace listing for a reward model states its architecture, its
training mixture and a leaderboard number, and states none of the thirteen quantities a buyer
would need to know whether the thing measures what it claims to. This package assembles those
thirteen from the instruments that already produce them and puts them on one page.

    from reward_lens.measure.card import CardInputs, grader_card, render_card

    reading = grader_card(
        CardInputs(verifier=source, corpus=rollouts, exploit_log=log),
        access={Component.GRADER: Access.SOURCE | Access.QUERY},
        phase=Phase.PRE_RUN,
    )
    print(render_card(reading))

Three properties are worth knowing before reading one.

**A card is mostly refusals and that is correct.** Each field is a `Reading`, which is
`Evidence | Refusal`, and a grader nobody instrumented has no replicated scoring design, no
exploit log and no recorded abstention channel. Those fields refuse, each naming what would let
them read. A rendered blank would be indistinguishable from a measured null, which is the exact
confusion this artifact exists to remove.

**The card cannot be asked to trust itself.** `Evidence.trust` is computed by the gates and this
package exposes no way to set it. The card additionally prints the lowest trust among the readings
it composes, and says in words when its own level exceeds that floor.

**Exploit content is withheld by default.** The surviving-mutant list and the false-positive
catalogue are reproducible ways to make the grader wrong. They carry `sensitive=True` on the
payload, the rendered card shows the redacted form, and the unredacted form needs both an explicit
request and the payload's own recorded disclosure decision.

`card_plan` answers "what would this card contain and what would it cost" with no grader call and
no GPU. For most readers that report is the product.
"""

from __future__ import annotations

from reward_lens.measure.card.card import (
    CARD_BASELINES,
    CARD_PURPOSE,
    D7_ACCESS_MIN,
    D7_ENVELOPE,
    CardField,
    CardPlan,
    CardReading,
    FieldPlan,
    GraderCard,
    card_context,
    card_plan,
    grader_card,
    refusal_reasons,
    render_card,
)
from reward_lens.measure.card.fields import (
    CARD_FIELDS,
    CATALOGUE_FIELDS,
    CardInputs,
    FieldSpec,
)

__all__ = [
    "CARD_BASELINES",
    "CARD_FIELDS",
    "CARD_PURPOSE",
    "CATALOGUE_FIELDS",
    "D7_ACCESS_MIN",
    "D7_ENVELOPE",
    "CardField",
    "CardInputs",
    "CardPlan",
    "CardReading",
    "FieldPlan",
    "FieldSpec",
    "GraderCard",
    "card_context",
    "card_plan",
    "grader_card",
    "refusal_reasons",
    "render_card",
]
