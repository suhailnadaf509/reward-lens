"""W6: the rate package. Two frozen studies, a price, a power calculation and a runbook.

Nothing here has been run and nothing here spends anything. `spec.freeze_all()` hashes both study
specifications through `studies/freeze.py` before any arm exists, which is the only ordering that
makes their predictions predictions. `price.render_all()` says what the arms cost, with every
assumption in code. `power` simulates the operating characteristic at exactly those arm sizes.
`RUNBOOK.md` is what a maintainer types if the price is worth paying.
"""

from studies.w6_rate.analysis import METRIC_ARCS, analyze_w6_1, analyze_w6_2
from studies.w6_rate.spec import freeze_all, w6_1_spec, w6_2_spec

__all__ = [
    "METRIC_ARCS",
    "analyze_w6_1",
    "analyze_w6_2",
    "freeze_all",
    "w6_1_spec",
    "w6_2_spec",
]
