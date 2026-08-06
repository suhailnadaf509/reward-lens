"""``reward_lens.attribution`` — reward decomposition (Direct Linear Attribution).

:mod:`reward_lens.attribution.dla` is the canonical, substrate-free implementation of the head- and
component-level reward decomposition that the battery calls. It used to sit beside the v1
``ComponentAttribution`` primitive, which was kept as the E-parity reference; that primitive and
the parity suite that compared them both retired with the rest of the v1 corpus, once E-parity had
passed across two releases as its deprecation condition required. The head math had already been
collapsed into this one implementation, so nothing was lost with it.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.attribution")

from reward_lens.attribution.dla import (
    component_reward_contributions,
    head_reward_contributions,
    project_onto_reward,
)

__all__ = [
    "project_onto_reward",
    "head_reward_contributions",
    "component_reward_contributions",
]
