"""M1 acceptance: the runtime's ``hvp`` computes the reward Hessian correctly (M1).

The reward is a scalar, so its Hessian at any site is ``d x d`` and reachable by Hessian-vector
products without ever materializing the network Jacobian. This test builds the tiny ``ClassifierRM``,
picks the final decoder layer's residual as the differentiation site (reward is a genuinely nonlinear
function of it, through the final RMSNorm and the head, so the Hessian is non-trivial), and checks
``HFRuntime.hvp`` two ways:

  1. The dense ``d x d`` Hessian recovered by stacking ``hvp`` over the standard basis matches an
     independent dense reference (``torch.autograd.functional.hessian`` on the reward as a function
     of the final-position residual, plus a finite-difference cross-check), and is symmetric.
  2. The top eigenvalue found by power iteration driven entirely by ``hvp`` matches the top
     eigenvalue of the dense reference.

This is an M1 acceptance criterion: a runner correct on the tiny model, with the second-order path
proven against a reference, is the deliverable.
"""

from __future__ import annotations

import numpy as np
import torch

from reward_lens.core.types import Site
from reward_lens.runtime.hooks import resolve_module
from reward_lens.signals.loaders import from_tiny


def _build():
    signal = from_tiny(seed=7, conformance_quickcheck=False)
    runtime = signal.runtime
    d_model = signal.meta.d_model
    site = Site(signal.meta.n_layers - 1, "resid_post")
    scalar_fn = signal.reward_scalar_fn("reward")
    tokenized = [signal.tokenize(("Question?", "alpha beta gamma delta epsilon"))]
    batch = runtime.collate(tokenized)
    return signal, runtime, d_model, site, scalar_fn, batch


def _dense_hvp_hessian(runtime, scalar_fn, site, d_model, batch) -> np.ndarray:
    """Materialize the dense Hessian by stacking hvp over the standard basis (the tested path)."""
    basis = torch.eye(d_model)
    stacked = runtime.hvp(batch, scalar_fn, site, basis)  # (B, d, d)
    return stacked[0].detach().numpy()


def _reference_hessian(signal, runtime, site, d_model, batch) -> np.ndarray:
    """An independent dense Hessian: autograd.functional.hessian on reward(final-position residual).

    A forward hook substitutes a differentiable vector ``x`` at the final position of the site's
    residual and the reward is read as ``x``'s function; ``torch.autograd.functional.hessian`` then
    differentiates it. This is a different code path from the runtime's manual double-backprop, so
    agreement validates the runtime's ``hvp`` rather than restating it.
    """
    weight = signal.readouts()[0].vector
    module = resolve_module(runtime.model, runtime.site_map.resolve(site))
    final_pos = int(runtime._final_positions(batch.attention_mask)[0])

    def reward_of_x(x: "torch.Tensor") -> "torch.Tensor":
        captured: dict = {}
        head_handle = runtime.head_module.register_forward_pre_hook(
            lambda _m, a: captured.__setitem__("h", a[0])
        )

        def hook(_m, _i, out):
            hidden = out[0] if isinstance(out, tuple) else out
            new = hidden.clone()
            new[0, final_pos] = x
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        edit_handle = module.register_forward_hook(hook)
        try:
            runtime.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            )
        finally:
            head_handle.remove()
            edit_handle.remove()
        head_input = captured["h"]
        return head_input[0, final_pos].to(torch.float32) @ weight

    # baseline value of the residual at the final position
    store: dict = {}

    def capture(_m, _i, out):
        hidden = out[0] if isinstance(out, tuple) else out
        store["x0"] = hidden[0, final_pos].detach().clone()

    handle = module.register_forward_hook(capture)
    with torch.no_grad():
        runtime.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        )
    handle.remove()
    x0 = store["x0"].requires_grad_(True)
    return torch.autograd.functional.hessian(reward_of_x, x0).detach().numpy()


def test_hvp_matches_dense_hessian():
    signal, runtime, d_model, site, scalar_fn, batch = _build()
    hessian_hvp = _dense_hvp_hessian(runtime, scalar_fn, site, d_model, batch)
    reference = _reference_hessian(signal, runtime, site, d_model, batch)

    assert hessian_hvp.shape == (d_model, d_model)
    # A Hessian is symmetric; the hvp-stacked one must be too, up to autograd noise.
    asym = float(np.abs(hessian_hvp - hessian_hvp.T).max())
    assert asym < 1e-4, f"hvp Hessian not symmetric: max asymmetry {asym}"
    # And it must match the independent autograd reference to autograd precision.
    diff = float(np.abs(hessian_hvp - reference).max())
    assert diff < 1e-4, f"hvp Hessian disagrees with autograd reference: max|diff|={diff}"


def test_hvp_finite_difference_cross_check():
    """A finite-difference Hessian cross-checks the hvp Hessian (loose tol for O(eps^2) error)."""
    signal, runtime, d_model, site, scalar_fn, batch = _build()
    hessian_hvp = _dense_hvp_hessian(runtime, scalar_fn, site, d_model, batch)

    weight = signal.readouts()[0].vector.numpy()
    module = resolve_module(runtime.model, runtime.site_map.resolve(site))
    final_pos = int(runtime._final_positions(batch.attention_mask)[0])

    def reward_of_x(x: np.ndarray) -> float:
        captured: dict = {}
        head_handle = runtime.head_module.register_forward_pre_hook(
            lambda _m, a: captured.__setitem__("h", a[0])
        )

        def hook(_m, _i, out):
            hidden = out[0] if isinstance(out, tuple) else out
            new = hidden.clone()
            new[0, final_pos] = torch.tensor(x, dtype=new.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        edit_handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                runtime.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )
        finally:
            head_handle.remove()
            edit_handle.remove()
        return float(captured["h"][0, final_pos].to(torch.float32).numpy() @ weight)

    store: dict = {}

    def capture(_m, _i, out):
        hidden = out[0] if isinstance(out, tuple) else out
        store["x0"] = hidden[0, final_pos].detach().numpy().copy()

    handle = module.register_forward_hook(capture)
    with torch.no_grad():
        runtime.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        )
    handle.remove()
    x0 = store["x0"]

    # Central-difference the top-left sub-block only: it is O(d^2) forwards, and the exact autograd
    # reference in ``test_hvp_matches_dense_hessian`` already validates the whole matrix. A block is
    # enough to cross-check the two second-order code paths without a multi-minute finite-diff sweep.
    block = min(d_model, 6)
    eps = 1e-3
    hessian_fd = np.zeros((block, block))
    for i in range(block):
        ei = np.zeros(d_model)
        ei[i] = eps
        for j in range(i, block):
            ej = np.zeros(d_model)
            ej[j] = eps
            value = (
                reward_of_x(x0 + ei + ej)
                - reward_of_x(x0 + ei - ej)
                - reward_of_x(x0 - ei + ej)
                + reward_of_x(x0 - ei - ej)
            ) / (4 * eps * eps)
            hessian_fd[i, j] = value
            hessian_fd[j, i] = value

    scale = max(float(np.abs(hessian_hvp).max()), 1.0)
    diff = float(np.abs(hessian_fd - hessian_hvp[:block, :block]).max())
    assert diff / scale < 5e-3, f"finite-diff Hessian disagrees: rel diff {diff / scale}"


def test_power_iteration_top_eigenvalue():
    """The top eigenvalue from power iteration via hvp matches the dense reference (M1 criterion)."""
    signal, runtime, d_model, site, scalar_fn, batch = _build()
    reference = _reference_hessian(signal, runtime, site, d_model, batch)
    dense_top = float(np.max(np.abs(np.linalg.eigvalsh(reference))))

    torch.manual_seed(0)
    vec = torch.randn(d_model)
    vec = vec / vec.norm()
    eigenvalue = 0.0
    for _ in range(300):
        hv = runtime.hvp(batch, scalar_fn, site, vec.reshape(1, d_model))[0, 0]
        eigenvalue = float(vec @ hv)  # Rayleigh quotient
        norm = hv.norm()
        if float(norm) < 1e-12:
            break
        vec = hv / norm

    assert dense_top > 1e-6, "degenerate Hessian; pick a site with curvature"
    rel = abs(abs(eigenvalue) - dense_top) / dense_top
    assert rel < 1e-3, (
        f"power-iteration top |eig| {abs(eigenvalue)} vs dense {dense_top} (rel {rel})"
    )


def test_grad_matches_readout_direction_shapes():
    """A sanity check on ``grad``: gradient shapes are right for both site and embedding targets."""
    signal, runtime, d_model, site, scalar_fn, batch = _build()
    g_site = runtime.grad(batch, scalar_fn, site)
    assert g_site.shape[-1] == d_model and g_site.shape[0] == 1
    g_embed = runtime.grad(batch, scalar_fn, "embeddings")
    assert g_embed.shape[-1] == d_model and g_embed.shape[0] == 1
    assert torch.isfinite(g_site).all() and torch.isfinite(g_embed).all()
