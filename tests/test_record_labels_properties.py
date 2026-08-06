"""Property tests for `record/labels.py`, in their own module because of one dependency.

`hypothesis` is in the `[dev]` and `[verifier]` extras and not in the base install, and the
base-install job in CI runs `pytest tests/` against the bare wheel to prove the torch-free subset
passes there. A module-level `from hypothesis import given` in `test_record_labels.py` would turn
that job's collection into an error, so the three property tests live here behind an
`importorskip` and the forty-odd deterministic ones stay runnable on a base install. They are the
tests most worth running there: the disjoint field sets, the name blocklist and the detector
introspection need nothing but the standard library.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis", reason="hypothesis is in the [dev] and [verifier] extras")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from reward_lens.core.types import canonical_bytes  # noqa: E402
from reward_lens.record.labels import (  # noqa: E402
    BLOCKED_NAMES,
    Blind,
    LabelLeak,
    LabelQuality,
    RolloutFrame,
    blind,
)
from reward_lens.record.schema import FeatureID, decode_foreign, encode_foreign  # noqa: E402

MEASURED = LabelQuality(
    error_rate=0.03,
    n_audited=200,
    method="two raters on a stratified sample, third adjudicating disagreements",
    measured_by="pool-a",
)

# Payloads distinctive enough that finding one inside a rendered string means it leaked, rather
# than meaning a short value collided with a digit of the fingerprint.
_PAYLOADS = st.one_of(
    st.text(alphabet="ZQXJKVW", min_size=8, max_size=24),
    st.integers(min_value=10**12, max_value=10**15),
)


@given(payload=_PAYLOADS, key=st.text(alphabet="abcdefg_", min_size=1, max_size=8))
def test_no_rendering_of_a_blind_ever_contains_its_payload(payload: object, key: str) -> None:
    """Over every label value, not just the one a hand-written test happened to pick."""
    label = blind(payload, key=key)
    for rendered in (repr(label), str(label), canonical_bytes(label).decode("utf-8")):
        assert str(payload) not in rendered


@given(payload=_PAYLOADS)
def test_a_blind_round_trips_through_the_record_codec_as_a_blind(payload: object) -> None:
    """The property `record/schema.py` depends on: the codec returns a `Blind`, never a dict.

    A `Blind` decoded to a mapping would hand over the label with the wrapper removed, which is why
    the kernel's codec raises on an unregistered payload type instead of degrading.
    """
    label = blind(payload, key="hacked", quality=MEASURED)
    restored = decode_foreign(encode_foreign(label))
    assert isinstance(restored, Blind)
    assert restored._value == payload
    assert restored.fingerprint == label.fingerprint
    assert restored.quality == MEASURED


@given(
    names=st.lists(st.sampled_from(sorted(BLOCKED_NAMES)), min_size=1, max_size=4, unique=True),
    clean=st.lists(
        st.sampled_from(["len_tokens", "n_turns", "advantage", "reward_pass"]),
        max_size=4,
        unique=True,
    ),
)
def test_every_blocked_name_is_caught_in_any_mixture(names: list[str], clean: list[str]) -> None:
    """Every blocked name fires, in any order, mixed with any number of legitimate features."""
    features = {FeatureID(n): 1.0 for n in names + clean}
    with pytest.raises(LabelLeak) as exc:
        RolloutFrame(
            trajectory_id="t1",
            task_id="task-1",
            turns=(),
            n_tokens=0,
            advantage=None,
            features=features,
        )
    for name in names:
        assert repr(name) in str(exc.value)
