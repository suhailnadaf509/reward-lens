"""Debt three acceptance: two defects fixed, each asserted with the number it changed.

1. **The causal algebra could not be mounted by the shipped runtime.** `mounted_interventions`
   expected a single-site object exposing `site` and `apply(hidden)`. Nothing in `interventions/`
   has that shape: every intervention there carries `site` and a `compile(signal) -> {site: hook}`
   mapping, and a `ComposedIntervention` carries neither. Measured before the fix, all three of
   `SteeringIntervention`, `compose([a, b])` and a compiled `CompiledIntervention` raised
   `AttributeError` inside the runtime, so the only mountable thing in the library was the adapter
   `interventions/rescue.py` had to write for itself. Two hook surfaces were wrong as well: a
   `head_out` mount is the `o_proj` input and was being installed as a forward hook on the
   `o_proj` output, and head sites resolve through a map keyed with `head=None`.

2. **Two of the three instruments in `measure/threshold/` had a baseline nobody could reach.**
   `BunchingElasticity` declares `baseline.smooth_density_null` and `baseline.window_sensitivity`
   as mandatory, and its constructor accepted neither the null's draw count nor the sweep's widths,
   so both were pinned at the free function's defaults for anyone going through the instrument.

3. **`DensityDiscontinuity` reported a gate where its own baseline said it could not.** On 25,664
   recorded completion lengths with no gate anywhere in them, the full-range test returned |z| from
   50.4 to 76.3 at three cutoffs with p below any printable floor, while its own smooth-density
   null came back centred at -23.1 with a spread of 2.26 where the asymptotics imply 0 and 1.
   Standardising against the band left the statistic 23.5 to 31.3 spreads out, so the baseline
   detected the failure and did not repair it. It now refuses with `ENVELOPE_VIOLATED` and carries
   both numbers.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Site
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.threshold import (
    BunchingElasticity,
    DeadZoneFraction,
    DensityDiscontinuity,
    Gate,
    RunningVariable,
    VarianceDerivative,
    density_discontinuity,
)
from reward_lens.measure.threshold.density import (
    MAX_NULL_CENTRE_SPREADS,
    MAX_NULL_SPREAD_RATIO,
    NullBand,
    null_band_failure,
)

torch = pytest.importorskip("torch", reason="the mount path needs the white-box extra")

import torch.nn as nn  # noqa: E402

from reward_lens.interventions.ablate import AblationIntervention  # noqa: E402
from reward_lens.interventions.base import compose  # noqa: E402
from reward_lens.interventions.rescue import knockout_and_rescue  # noqa: E402
from reward_lens.interventions.steer import SteeringIntervention  # noqa: E402
from reward_lens.runtime.hooks import mount_points, mounted_interventions  # noqa: E402

#: X5's cached completion lengths from `ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2`,
#: the subject the numbers in point 3 above were measured on. The cache is not in this
#: repository: point ``REWARD_LENS_X5_LENGTHS`` at an ``.npz`` holding a ``length`` array to
#: reproduce them, or the tests that need it skip.
_LENGTHS_ENV = os.environ.get("REWARD_LENS_X5_LENGTHS")
LENGTHS = pathlib.Path(_LENGTHS_ENV) if _LENGTHS_ENV else None

D_MODEL = 8
N_HEADS = 2


# ---------------------------------------------------------------------------
# A toy network with the two hook surfaces the runtime addresses
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.received: "torch.Tensor | None" = None
        real = self.o_proj.forward

        def spy(x: "torch.Tensor") -> "torch.Tensor":
            self.received = x.detach().clone()
            return real(x)

        self.o_proj.forward = spy  # type: ignore[method-assign]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.o_proj(x)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.lin = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor"]:
        return (self.lin(x) + self.self_attn(x),)


class _Net(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _SiteMap:
    """Resolves the two surfaces, and asserts that head sites arrive keyed with `head=None`."""

    n_heads = N_HEADS

    def resolve(self, site: Site) -> str:
        if site.point == "head_out":
            assert site.head is None, (
                f"a head mount resolved through {site}; the site map keys the one head-agnostic "
                f"o_proj module with head=None and the hook edits the requested head's slice"
            )
            return f"layers.{site.layer}.self_attn.o_proj"
        return f"layers.{site.layer}"


class _Adapter:
    def extract_layer_output(self, out):  # noqa: ANN001, ANN201
        return out[0] if isinstance(out, tuple) else out

    extract_attn_output = extract_layer_output
    extract_mlp_output = extract_layer_output


@pytest.fixture
def net():
    torch.manual_seed(0)
    return _Net(), _SiteMap(), _Adapter(), torch.randn(2, 3, D_MODEL)


def _unit(index: int) -> np.ndarray:
    v = np.zeros(D_MODEL)
    v[index] = 1.0
    return v


# ===========================================================================
# 1. The causal algebra mounts
# ===========================================================================


def test_every_shape_the_algebra_produces_is_mountable(net):
    """All three raised `AttributeError` inside the runtime before this. The list is the fix."""
    model, site_map, adapter, x = net
    a = SteeringIntervention(site=Site(0, "resid_post"), direction=_unit(0), strength=1.5)
    b = SteeringIntervention(site=Site(1, "resid_post"), direction=_unit(1), strength=-0.5)

    for label, obj in (
        ("a bare Intervention", a),
        ("a ComposedIntervention", compose([a, b])),
        ("an already-compiled CompiledIntervention", a.compile(None)),
    ):
        with mounted_interventions(model, adapter, site_map, [obj]):
            out = model(x)
        assert torch.isfinite(out).all(), label


def test_two_interventions_at_one_site_both_land(net):
    """Composition rather than last-writer-wins, which is what makes `compose` an algebra.

    The network is affine, so a steer at layer 0 propagates additively: the composed run has to
    equal the sum of the two single runs minus the clean one, to floating point.
    """
    model, site_map, adapter, x = net
    a = SteeringIntervention(site=Site(0, "resid_post"), direction=_unit(0), strength=1.5)
    b = SteeringIntervention(site=Site(0, "resid_post"), direction=_unit(1), strength=-0.5)

    clean = model(x).detach()
    with mounted_interventions(model, adapter, site_map, [compose([a, b])]):
        both = model(x).detach()
    with mounted_interventions(model, adapter, site_map, [a]):
        only_a = model(x).detach()
    with mounted_interventions(model, adapter, site_map, [b]):
        only_b = model(x).detach()

    assert float((both - clean).abs().max()) > 0.5
    assert float((both - only_a).abs().max()) > 0.05, "the second member did not land"
    assert float((both - (only_a + only_b - clean)).abs().max()) < 1e-5


def test_a_head_mount_edits_the_o_proj_input_and_not_its_output(net):
    """The surface, not just the site. A head's contribution is only separable before `o_proj`.

    Before the fix this mounted as a forward hook on `o_proj`, so the head-slicing hook was handed
    the projected output: it ran without error and zeroed the wrong tensor.
    """
    model, site_map, adapter, x = net
    ablation = AblationIntervention(
        site=Site(0, "head_out", 1), direction=None, mode="head", n_heads=N_HEADS
    )
    with mounted_interventions(model, adapter, site_map, [ablation]):
        model(x)

    received = model.layers[0].self_attn.received
    assert received is not None
    per_head = received.view(2, 3, N_HEADS, D_MODEL // N_HEADS)
    assert float(per_head[:, :, 1, :].abs().max()) == 0.0, "head 1 was not ablated at the input"
    assert float(per_head[:, :, 0, :].abs().max()) > 0.0, "head 0 was ablated as well"


def test_the_single_site_shape_the_rescue_adapter_uses_still_mounts(net):
    """`rescue.Mounted` predates the fix and must keep working, so both contracts are accepted."""
    model, site_map, adapter, x = net
    ablated, rescued, spec = knockout_and_rescue(
        ablate_at=Site(0, "resid_post"), direction=_unit(0), direction_id="d0"
    )
    clean = model(x).detach()
    with mounted_interventions(model, adapter, site_map, ablated):
        knocked = model(x).detach()
    with mounted_interventions(model, adapter, site_map, rescued):
        restored = model(x).detach()

    assert spec.is_same_site
    assert float((knocked - clean).abs().max()) > 0.1, "the ablation did nothing"
    assert float((restored - clean).abs().max()) < 1e-5, "a same-site rescue is a near no-op"


def test_mount_points_orders_a_multi_site_intervention_by_forward_order(net):
    """A recorder at layer 0 and a re-injection at layer 1 mount in the order the pass reaches."""
    a = SteeringIntervention(site=Site(1, "resid_post"), direction=_unit(0), strength=1.0)
    b = SteeringIntervention(site=Site(0, "resid_post"), direction=_unit(1), strength=1.0)
    pairs = mount_points(compose([a, b]))
    assert [site.layer for site, _ in pairs] == [0, 1]


def test_something_with_neither_contract_says_so(net):
    with pytest.raises(TypeError, match="not mountable"):
        mount_points(object())


# ===========================================================================
# 2. The bunching instrument can reach its own baselines
# ===========================================================================


@pytest.fixture
def lengths() -> np.ndarray:
    if LENGTHS is None or not LENGTHS.exists():
        pytest.skip("no cached AISI completion lengths; set REWARD_LENS_X5_LENGTHS")
    return np.asarray(np.load(LENGTHS)["length"], dtype=np.float64)


def _gate(cutoff: float) -> Gate:
    return Gate(
        name=f"budget_gate_{int(cutoff)}",
        cutoff=float(cutoff),
        unit="characters",
        penalised_side="above",
        kind="notch",
        penalty_fraction=2.0,
        constant=-1.0,
        installed=True,
        provenance="installed counterfactually onto recorded completion lengths",
    )


def _running(values: np.ndarray, name: str) -> RunningVariable:
    return RunningVariable(name=name, values=values, unit="characters", source="x5 cache")


def test_the_bunching_null_and_sweep_are_reachable_through_the_constructor(lengths):
    """Both were pinned at 200 draws and nine widths for anyone using the instrument."""
    window = lengths[(lengths >= 200.0) & (lengths <= 2000.0)]
    running = _running(window, "completion length in [200, 2000]")

    default = BunchingElasticity(running, _gate(900.0), n_boot=20, n_placebos=5).compute()
    assert not isinstance(default, Refusal), default
    assert default.smooth_null.n_draws == 200
    assert tuple(default.sweep.window_bins) == (1, 2, 3, 4, 6, 8, 12, 16, 24)

    asked = BunchingElasticity(
        running, _gate(900.0), n_boot=20, n_placebos=5, n_null=25, sweep_bins=(1, 3, 9)
    ).compute()
    assert not isinstance(asked, Refusal), asked
    assert asked.smooth_null.n_draws == 25
    assert tuple(asked.sweep.window_bins) == (1, 3, 9)


# ===========================================================================
# 3. The density test refuses when its own null says the asymptotics do not hold
# ===========================================================================


def test_the_null_band_check_fires_on_a_displaced_centre_and_on_a_wrong_spread():
    """The two faults separately, because they are different faults and both were measured."""
    assert null_band_failure(NullBand("null", 300, 0.05, 1.02, 2.0, 0.5)) is None

    centre = null_band_failure(NullBand("null", 300, -23.15, 2.26, 27.0, 0.0))
    assert centre is not None
    assert centre[1]["null_centre_spreads"] > MAX_NULL_CENTRE_SPREADS
    assert "centre" in centre[0]

    spread = null_band_failure(NullBand("null", 300, -0.05, 1.69, 3.4, 0.4))
    assert spread is not None
    assert spread[1]["null_sd"] > MAX_NULL_SPREAD_RATIO
    assert "spread" in spread[0]

    unmeasured = null_band_failure(NullBand("null", 0, float("nan"), float("nan"), 0.0, 0.0))
    assert unmeasured is None, "a baseline that could not run is not a violated premise"


def test_the_full_range_length_density_is_refused_at_every_cutoff(lengths):
    """The three cutoffs X5 read. Before this they returned p below any printable floor."""
    running = _running(lengths, "completion length")
    for cutoff in (600.0, 900.0, 1200.0):
        out = density_discontinuity(
            running, _gate(cutoff), rung=0, n_null=300, n_placebos=10, n_boot=0, seed=1
        )
        assert isinstance(out, Refusal), f"cutoff {cutoff:g} returned a reading"
        assert out.reason is RefusalReason.ENVELOPE_VIOLATED
        stats = out.statistics
        assert abs(stats["z"]) > 50.0, "the statistic it withheld is still on the refusal"
        assert "[200, 2000]" in out.remedy, "the remedy names the worked example"
        assert "0.000 +/- 1.000" in out.detail, "the detail names what the asymptotics imply"


def test_the_same_test_on_the_smooth_window_still_reads(lengths):
    """The refusal has to be about the density and not about the instrument refusing everything."""
    window = lengths[(lengths >= 200.0) & (lengths <= 2000.0)]
    running = _running(window, "completion length in [200, 2000]")
    for cutoff in (600.0, 900.0, 1200.0):
        out = density_discontinuity(
            running, _gate(cutoff), rung=0, n_null=300, n_placebos=10, n_boot=0, seed=1
        )
        assert not isinstance(out, Refusal), out
        assert abs(out.z) < 2.0
        assert abs(out.smooth_null.mean) < 0.5
        assert abs(out.smooth_null.sd - 1.0) < 0.1


def test_the_refusal_does_not_depend_on_which_rung_produced_the_statistic(lengths):
    """The premise is a property of the density, so the answer cannot depend on the estimator.

    At rung 1 the same three cutoffs return |z| below 7 rather than above 50, because the robust
    estimator's bootstrap variance is larger. The band is unchanged and the reading is still
    invalid, which is why the check is on the band rather than on the statistic.
    """
    running = _running(lengths, "completion length")
    out = density_discontinuity(
        running, _gate(900.0), rung=1, n_null=300, n_placebos=10, n_boot=200, seed=1
    )
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert abs(out.statistics["z"]) < 20.0, "rung 1 shrinks the statistic and not the violation"


# ===========================================================================
# Lint, on everything this package ships
# ===========================================================================


def test_every_threshold_instrument_lints_clean():
    """Rule 3 of the standing four: an instrument that cannot pass lint does not exist."""
    for instrument in (
        DensityDiscontinuity(),
        BunchingElasticity(),
        DeadZoneFraction(),
        VarianceDerivative(),
    ):
        findings = lint_instrument(instrument)
        assert findings == [], f"{instrument.name}: {[f.problem for f in findings]}"
