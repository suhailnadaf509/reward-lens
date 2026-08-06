"""Freezing a study: preregistration made mechanical.

Freezing hashes the spec (hypotheses, predictions, analysis plan, kill criteria) and records the
git commit that was current when it happened. The resulting StudyID, ``study:name@vN#hash``, is
what stamps every Evidence the study produces as REGISTERED. The point is temporal: a prediction
that is hashed before the run cannot be edited after seeing the data without producing a new,
visibly different study version. This is the corpus's preregistration discipline turned into
a content hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from reward_lens.core.provenance import git_sha
from reward_lens.core.types import StudyID, content_hash
from reward_lens.studies.spec import StudySpec


@dataclass(frozen=True)
class FrozenStudy:
    """A study spec frozen at a point in time.

    Carries the spec, the derived StudyID, the git sha of the code that froze it, the freeze
    timestamp, and the spec hash. A dirty working tree is visible in ``git_sha`` (it carries a
    ``+dirty`` suffix), so a study frozen against uncommitted code is not mistaken for a
    reproducible one.
    """

    spec: StudySpec
    study_id: StudyID
    git_sha: str
    frozen_at: str
    spec_hash: str

    @property
    def predates(self) -> str:
        """The freeze timestamp; every REGISTERED Evidence under this study is created after it."""
        return self.frozen_at


def freeze(
    spec: StudySpec, repo_dir: str | None = None, frozen_at: str | None = None
) -> FrozenStudy:
    """Freeze a study spec, computing its StudyID and recording the git sha.

    The StudyID is ``study:{spec.id}@v{spec.version}#{hash8}`` where the hash is the first 8 hex
    characters of the spec's content hash. Two specs that differ in any registered field (a
    prediction threshold, a kill criterion, the analysis path) get different ids, which is exactly
    the property that makes post-hoc editing visible as a new version.
    """
    spec_hash = content_hash(spec.__canonical__(), "spec")
    short = spec_hash.split(":")[1][:8]
    study_id = StudyID(f"study:{spec.id}@v{spec.version}#{short}")
    ts = frozen_at or datetime.now(timezone.utc).isoformat()
    return FrozenStudy(
        spec=spec,
        study_id=study_id,
        git_sha=git_sha(repo_dir),
        frozen_at=ts,
        spec_hash=spec_hash,
    )


__all__ = ["FrozenStudy", "freeze"]
