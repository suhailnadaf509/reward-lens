"""STUB: HF release tooling for RewardBench-GT.

`zoo.py` will package trained organisms plus their answer keys into the public RewardBench-GT release
(the organisms, the answer keys, and the MIB-RM track), the citation-magnet artifact of the ground
truth program. That needs the Hugging Face Hub upload path, a dataset card schema, and a stable
serialization of trained trunks, none of which is in this package's scope. This module is a marked
placeholder so the subsystem surface is complete; the functions raise with a clear pointer rather
than pretending to publish.

Everything needed to build the release already exists in the other modules: `foundry` generates the
organisms and answer keys, `train` produces the trunks, `verify` gates acceptance, and `scorecard`
produces the leaderboard rows. `zoo.py` is only the packaging-and-upload layer over those.
"""

from __future__ import annotations

from typing import Any


def export_rewardbench_gt(*_args: Any, **_kwargs: Any) -> None:
    """STUB: package accepted organisms + answer keys as a RewardBench-GT release.

    Deferred: needs the Hugging Face Hub upload path and a trunk serialization format this package
    does not have. Assemble the release from `foundry` (organisms + keys), `train`/`verify` (accepted trunks),
    and `scorecard` (leaderboard rows) when the Hub tooling lands.
    """
    raise NotImplementedError(
        "export_rewardbench_gt is a STUB: HF Hub release tooling for RewardBench-GT "
        "is deferred. The organisms, answer keys, trained trunks, and scorecards it would package are "
        "already produced by foundry/train/verify/scorecard."
    )


def load_rewardbench_gt(*_args: Any, **_kwargs: Any) -> None:
    """STUB: load a published RewardBench-GT release. Deferred with `export_...`."""
    raise NotImplementedError(
        "load_rewardbench_gt is a STUB: the RewardBench-GT release format and HF Hub "
        "download path are deferred."
    )


__all__ = ["export_rewardbench_gt", "load_rewardbench_gt"]
