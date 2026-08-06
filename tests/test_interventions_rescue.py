"""Unit tests for `interventions/rescue.py`, including against the real tiny policy.

The mechanics that make a rescue a rescue: the recorder captures what the ablation is about to
remove, the pair runs in one forward pass, re-injecting the exact coordinate restores the
activation, and the norm-matched control is matched by construction rather than by arithmetic.

`TestAgainstTheRealPolicy` runs the whole thing through `trl-internal-testing/tiny-Qwen3ForCausalLM`
so the hook path, the site resolution and the composition order are exercised against a real decoder
stack rather than against a tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reward_lens.core.types import Site  # noqa: E402
from reward_lens.interventions.base import compose  # noqa: E402
from reward_lens.interventions.rescue import (  # noqa: E402
    RecordRemoved,
    Reinject,
    RemovedCoordinate,
    RescueError,
    knockout_and_rescue,
    norm_matched_random,
)

SITE = Site(0, "resid_post")


def _hidden(seed: int = 0, d: int = 8, b: int = 2, t: int = 5) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g, dtype=torch.float64)


def _unit(d: int = 8, seed: int = 1) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(d)
    return v / np.linalg.norm(v)


class TestTheRecorder:
    def test_it_passes_the_activation_through_untouched(self) -> None:
        h = _hidden()
        box = RemovedCoordinate()
        hook = RecordRemoved(SITE, _unit(), into=box).compile().mounts[SITE]
        assert torch.allclose(hook(h, {}), h)

    def test_it_captures_the_coordinate_the_ablation_will_remove(self) -> None:
        h = _hidden()
        u = _unit()
        box = RemovedCoordinate()
        RecordRemoved(SITE, u, into=box).compile().mounts[SITE](h, {})
        expected = (h * torch.as_tensor(u, dtype=h.dtype)).sum(dim=-1, keepdim=True)
        assert torch.allclose(box.value, expected)
        assert box.site == SITE and box.n_calls == 1

    def test_compiling_clears_a_stale_coordinate(self) -> None:
        """Replaying a previous pass's coordinates would be a silent wrong answer."""
        box = RemovedCoordinate()
        recorder = RecordRemoved(SITE, _unit(), into=box)
        recorder.compile().mounts[SITE](_hidden(), {})
        assert box.recorded
        recorder.compile()
        assert not box.recorded and box.n_calls == 0


#: `interventions.steer.unit_direction` casts to fp32 by design, so a directional ablation applied
#: to a float64 activation leaves a residual coordinate at fp32 epsilon rather than at float64's.
#: Measured at about 7e-8 on the fixture below. Asserting 1e-12 would be asserting a property of a
#: float32 direction that it does not have.
FP32_RESIDUAL = 1e-6


def _run(arm, hidden):
    """Apply an arm (a list of mountable single-site objects) at one site, in order."""
    out = hidden
    for m in arm:
        if m.site == SITE:
            out = m.apply(out)
    return out


class TestReinjection:
    def test_record_ablate_reinject_is_the_identity_at_one_site(self) -> None:
        """The sanity check: restoring at the ablated site returns the original activation."""
        h = _hidden()
        u = _unit()
        _ablated, rescued, spec = knockout_and_rescue(ablate_at=SITE, direction=u)
        assert spec.is_same_site and not spec.is_control
        assert torch.allclose(_run(rescued, h.clone()), h, atol=FP32_RESIDUAL)

    def test_the_ablation_alone_removes_the_component(self) -> None:
        h = _hidden()
        u = _unit()
        ablated, _rescued, _spec = knockout_and_rescue(ablate_at=SITE, direction=u)
        coord = (_run(ablated, h.clone()) * torch.as_tensor(u, dtype=h.dtype)).sum(dim=-1)
        assert torch.allclose(coord, torch.zeros_like(coord), atol=FP32_RESIDUAL)

    def test_a_random_re_injection_is_norm_matched_by_construction(self) -> None:
        """`c*v` and `c*u` have the same norm for any two unit vectors: no arithmetic needed."""
        h = _hidden()
        u = _unit()
        v = norm_matched_random(u, seed=7)
        assert np.isclose(np.linalg.norm(v), 1.0)
        base_arm, real, _ = knockout_and_rescue(ablate_at=SITE, direction=u)
        _abl2, control, spec = knockout_and_rescue(
            ablate_at=SITE, direction=u, substitute=v, substitute_id="random"
        )
        assert spec.is_control
        real_out = _run(real, h.clone())
        ctrl_out = _run(control, h.clone())
        base = _run(base_arm, h.clone())
        assert torch.allclose(
            (real_out - base).norm(dim=-1), (ctrl_out - base).norm(dim=-1), atol=FP32_RESIDUAL
        )
        assert not torch.allclose(real_out, ctrl_out)

    def test_scaling_gives_a_partial_rescue(self) -> None:
        h = _hidden()
        u = _unit()
        _abl, half, _ = knockout_and_rescue(ablate_at=SITE, direction=u, scale=0.5)
        coord = (_run(half, h.clone()) * torch.as_tensor(u, dtype=h.dtype)).sum(
            dim=-1, keepdim=True
        )
        original = (h * torch.as_tensor(u, dtype=h.dtype)).sum(dim=-1, keepdim=True)
        assert torch.allclose(coord, 0.5 * original, atol=FP32_RESIDUAL)

    def test_the_arms_are_mountable_by_the_shipped_runtime(self) -> None:
        """`runtime/hooks.py` mounts `site` + `apply(hidden)`, not a `CompiledIntervention`."""
        ablated, rescued, _ = knockout_and_rescue(ablate_at=SITE, direction=_unit())
        for arm in (ablated, rescued):
            assert arm and all(hasattr(m, "site") and hasattr(m, "apply") for m in arm)
            assert [m.site.layer for m in arm] == sorted(m.site.layer for m in arm)

    def test_re_injecting_with_nothing_recorded_raises(self) -> None:
        """A wiring bug, not an anticipated condition, so it raises rather than refusing."""
        box = RemovedCoordinate()
        hook = Reinject(SITE, _unit(), source=box).compile().mounts[SITE]
        with pytest.raises(RescueError, match="nothing was recorded"):
            hook(_hidden(), {})

    def test_a_shape_mismatch_between_the_two_sites_raises(self) -> None:
        box = RemovedCoordinate()
        RecordRemoved(SITE, _unit(), into=box).compile().mounts[SITE](_hidden(b=2, t=5), {})
        hook = (
            Reinject(Site(1, "resid_post"), _unit(), source=box)
            .compile()
            .mounts[Site(1, "resid_post")]
        )
        with pytest.raises(RescueError, match="row-for-row"):
            hook(_hidden(b=3, t=7), {})


class TestComposition:
    def test_the_recorder_runs_before_the_ablation(self) -> None:
        """`ComposedIntervention` chains hooks at one site in declaration order, which is what makes
        the recorded coordinate the one the ablation is about to remove rather than zero."""
        h = _hidden()
        u = _unit()
        box = RemovedCoordinate()
        from reward_lens.interventions.ablate import AblationIntervention

        pair = compose(
            [
                RecordRemoved(SITE, u, into=box),
                AblationIntervention(site=SITE, direction=u, mode="directional"),
            ]
        )
        pair.compile(None).mounts[SITE](h.clone(), {})
        expected = (h * torch.as_tensor(u, dtype=h.dtype)).sum(dim=-1, keepdim=True)
        assert torch.allclose(box.value, expected)
        assert not torch.allclose(box.value, torch.zeros_like(box.value))

    def test_the_two_arms_differ(self) -> None:
        """The rescued arm does something the ablated one does not, at the same site."""
        h = _hidden()
        ablated, rescued, _ = knockout_and_rescue(ablate_at=SITE, direction=_unit())
        assert not torch.allclose(_run(ablated, h.clone()), _run(rescued, h.clone()))


@pytest.mark.whitebox
class TestAgainstTheRealPolicy:
    """The hook path, site resolution and composition order against a real decoder stack."""

    @pytest.fixture(scope="class")
    def policy(self):
        pytest.importorskip("transformers")
        from reward_lens.policy.hf import from_pretrained

        try:
            return from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM", contrast=("Yes", "No")
            )
        except Exception as exc:  # pragma: no cover - network/cache dependent
            pytest.skip(f"tiny-Qwen3 unavailable: {exc}")

    def test_the_rescue_restores_the_score_it_ablated(self, policy) -> None:
        from reward_lens.policy.selection import behaviour_under, capture_at

        items = ["The capital of France is", "Water boils at"]
        site = Site(max(int(policy.meta.n_layers) - 2, 0), "resid_post")
        acts = capture_at(policy, items, site)
        direction = acts.mean(axis=0)
        direction = direction / np.linalg.norm(direction)

        ablated, rescued, spec = knockout_and_rescue(ablate_at=site, direction=direction)
        clean = behaviour_under(policy, items, readout="decision", condition="clean")
        knocked = behaviour_under(
            policy, items, intervention=ablated, readout="decision", condition="ablated"
        )
        back = behaviour_under(
            policy, items, intervention=rescued, readout="decision", condition="rescued"
        )
        assert spec.is_same_site
        # Restoring at the ablated site is a no-op on the forward pass, so the score returns.
        assert np.allclose(back.scores, clean.scores, atol=1e-4), (
            f"clean {clean.scores}, ablated {knocked.scores}, rescued {back.scores}"
        )
