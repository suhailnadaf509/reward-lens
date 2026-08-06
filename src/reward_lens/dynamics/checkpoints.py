"""Checkpoint sequences over training time, with a verifiable fingerprint chain.

The developmental science (RM-Pythia / D1) asks how a reward model's internals form across training,
so its subject is not one model but an ordered chain of checkpoints. This module is the substrate:
a `CheckpointSequence` is a sequence of `(step, ModelFP, loader)` triples that any battery or index
can sweep over with full provenance (`sweep.py`). The provenance is what distinguishes a
developmental result from a pile of unrelated snapshots, so the sequence is not a bare list: the
checkpoints are linked into a hash chain over their fingerprints, and the chain is verifiable.

The chain is the ordinary tamper-evidence construction. Each checkpoint carries a ``link`` that hashes
its ``(step, model_fp, prev_link)``; the genesis checkpoint chains from a fixed constant. Editing any
recorded fingerprint, reordering the steps, or dropping a checkpoint breaks the recomputed link at
that point and every link after it, so `verify_chain` localizes the first tampered checkpoint. The
chain alone only protects the *recorded* metadata; `verify_fingerprints` closes the loop by loading
each checkpoint and recomputing `runtime.fingerprint`, which catches a swapped weight file whose
recorded fingerprint was never updated. Together they answer "is this the training run it claims to
be", which is the precondition for reading a trajectory as development rather than noise (RK9).

The real RM-Pythia run (Qwen2.5 0.5B/1.5B/7B and Llama-3.1-8B, fifty to a hundred log-spaced
checkpoints carried deliberately through a second epoch) is a few-hundred-GPU-hour build; it is wired
here as a GPU-gated function that refuses without CUDA and never fabricates a checkpoint number.
The CPU-provable vehicle is `synthetic_planted_sequence`, a handful of tiny `ClassifierRM`s that
differ only in a planted, growing reward loading, so the sweep, the bias-entry
curve, and the stabilization detector are all provable on this hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from reward_lens.core.errors import ProvenanceError
from reward_lens.core.evidence import register_payload
from reward_lens.core.types import ModelFP, content_hash

if TYPE_CHECKING:  # torch and the signal surface are referenced only in annotations
    import numpy as np

    from reward_lens.signals.base import RewardSignal

# The chain's genesis: a fixed constant the first checkpoint links from, so a one-checkpoint chain is
# still a chain and the first link depends on nothing outside the sequence.
GENESIS_LINK = "clink:genesis"

# A loader materializes the signal for a checkpoint. It is a thunk so a hundred 8B checkpoints are
# never all resident at once; the sweep calls it only on a cache miss.
CheckpointLoader = Callable[[], "RewardSignal"]


def _link(step: int, model_fp: ModelFP, prev_link: str) -> str:
    """Hash a checkpoint's ``(step, model_fp, prev_link)`` into its chain link.

    The link derives only from structural content (an integer step, the content-derived fingerprint
    string, and the previous link), so two builds of the same chain land on the same links and a
    changed fingerprint or step changes this link and, through ``prev_link``, every link downstream.
    """
    return content_hash(
        {"step": int(step), "model_fp": str(model_fp), "prev_link": prev_link}, "clink"
    )


@dataclass
class Checkpoint:
    """One point on the training trajectory: a step, a fingerprint, and a way to load it.

    ``step`` is the training step (or epoch fraction) the checkpoint was taken at and the covariate
    every developmental curve is read against. ``model_fp`` is the `runtime.fingerprint` of the
    weights at that step, recorded once and never recomputed cheaply. ``loader`` materializes the
    `RewardSignal` on demand and is deliberately excluded from equality and hashing (two checkpoints
    are the same iff their recorded identity is). ``link``/``prev_link`` are the hash-chain fields.
    """

    step: int
    model_fp: ModelFP
    loader: CheckpointLoader = field(compare=False, repr=False)
    link: str = ""
    prev_link: str = GENESIS_LINK
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def load(self) -> "RewardSignal":
        """Materialize the signal for this checkpoint (calls the loader thunk)."""
        return self.loader()


@register_payload
@dataclass
class ChainVerification:
    """The result of verifying a checkpoint chain.

    ``ok`` is the single bit; ``first_bad_step`` and ``reason`` localize the first tampered or
    inconsistent checkpoint so a failure is actionable rather than a bare False. ``mode`` records
    whether the shallow link check or the deep fingerprint recomputation produced this result, and
    ``n_checkpoints`` is how many were examined.
    """

    ok: bool
    n_checkpoints: int
    mode: str = "chain"
    first_bad_step: int | None = None
    reason: str = ""


class CheckpointSequence:
    """An ordered, fingerprint-chained sequence of checkpoints over training time.

    Build one with `CheckpointSequence.build` from `(step, ModelFP, loader)` triples; the constructor
    computes the hash chain so the sequence is self-verifying from the moment it exists. Iterate it to
    sweep an Observable or index across training time (`sweep.sweep_over_checkpoints`); the sequence's
    `signature` is what keys a resumable sweep to this exact chain.

    Verification comes in two depths. `verify_chain` recomputes the links and is pure and instant: it
    catches an edited fingerprint, a reordered or dropped checkpoint, or a broken link. `verify_fingerprints`
    additionally loads each model and recomputes `runtime.fingerprint`, catching weights swapped under
    an unchanged record. `verify` runs the chain check and, when ``deep=True``, the fingerprint check,
    raising `ProvenanceError` on the first failure so a study cannot sweep a chain it has not trusted.
    """

    def __init__(self, checkpoints: list[Checkpoint]):
        self._checkpoints = list(checkpoints)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        triples: list[tuple[int, ModelFP, CheckpointLoader]],
        *,
        meta: dict[int, dict[str, Any]] | None = None,
    ) -> "CheckpointSequence":
        """Build a chained sequence from ``(step, model_fp, loader)`` triples.

        The triples are sorted by step and linked in order, so the caller may pass them in any order.
        ``meta`` optionally attaches per-step metadata (a local path, an HF revision id, an epoch
        fraction) that rides along on each `Checkpoint`. Steps must be distinct; a duplicate step is a
        `ProvenanceError` because it makes the trajectory covariate ambiguous.
        """
        ordered = sorted(triples, key=lambda t: t[0])
        steps = [s for s, _, _ in ordered]
        if len(set(steps)) != len(steps):
            raise ProvenanceError(
                f"checkpoint steps must be distinct; got duplicates in {steps}. A repeated step makes "
                f"the developmental covariate ambiguous."
            )
        meta = meta or {}
        checkpoints: list[Checkpoint] = []
        prev = GENESIS_LINK
        for step, model_fp, loader in ordered:
            link = _link(step, model_fp, prev)
            checkpoints.append(
                Checkpoint(
                    step=step,
                    model_fp=model_fp,
                    loader=loader,
                    link=link,
                    prev_link=prev,
                    meta=dict(meta.get(step, {})),
                )
            )
            prev = link
        return cls(checkpoints)

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._checkpoints)

    def __iter__(self):
        return iter(self._checkpoints)

    def __getitem__(self, idx: int) -> Checkpoint:
        return self._checkpoints[idx]

    @property
    def steps(self) -> list[int]:
        """The training steps in order (the developmental covariate)."""
        return [cp.step for cp in self._checkpoints]

    @property
    def head_link(self) -> str:
        """The final chain link, which transitively commits to every checkpoint's identity."""
        return self._checkpoints[-1].link if self._checkpoints else GENESIS_LINK

    def signature(self) -> str:
        """A content id for the whole chain (the head link), used to key a resumable sweep to it."""
        return self.head_link

    # -- verification -------------------------------------------------------

    def verify_chain(self) -> ChainVerification:
        """Recompute the hash chain and report the first inconsistency.

        Pure and instant: for each checkpoint it recomputes ``link`` from ``(step, model_fp, prev)``
        and checks it matches the recorded link, that the recorded ``prev_link`` matches the running
        previous link, and that the steps strictly increase. Any edit to a recorded fingerprint or
        step, any reordering, or any dropped checkpoint surfaces here as the first bad step.
        """
        prev = GENESIS_LINK
        last_step: int | None = None
        for cp in self._checkpoints:
            if last_step is not None and cp.step <= last_step:
                return ChainVerification(
                    ok=False,
                    n_checkpoints=len(self._checkpoints),
                    first_bad_step=cp.step,
                    reason=f"steps not strictly increasing: {cp.step} follows {last_step}",
                )
            if cp.prev_link != prev:
                return ChainVerification(
                    ok=False,
                    n_checkpoints=len(self._checkpoints),
                    first_bad_step=cp.step,
                    reason=f"prev_link mismatch at step {cp.step}: broken or reordered chain",
                )
            expected = _link(cp.step, cp.model_fp, prev)
            if expected != cp.link:
                return ChainVerification(
                    ok=False,
                    n_checkpoints=len(self._checkpoints),
                    first_bad_step=cp.step,
                    reason=(
                        f"link mismatch at step {cp.step}: recorded fingerprint or step was "
                        f"tampered after the chain was sealed"
                    ),
                )
            prev = cp.link
            last_step = cp.step
        return ChainVerification(ok=True, n_checkpoints=len(self._checkpoints))

    def verify_fingerprints(self) -> ChainVerification:
        """Load each checkpoint and recompute its fingerprint against the record.

        This is the deep check the chain cannot do on its own: it materializes each signal through its
        loader and recomputes `runtime.fingerprint`, so a weight file swapped under an unchanged record
        is caught even though the recorded metadata is internally consistent. It loads models, so it is
        torch-gated in effect (the loaders build models) and is the expensive verification; a study
        runs it once before trusting a chain, not on every sweep.
        """
        from reward_lens.runtime import fingerprint as compute_fingerprint

        for cp in self._checkpoints:
            signal = cp.load()
            tokenizer = getattr(signal, "tokenizer", None)
            adapter = getattr(getattr(signal, "runtime", None), "adapter", None)
            adapter_name = type(adapter).__name__ if adapter is not None else ""
            model = getattr(getattr(signal, "runtime", None), "model", None)
            if model is None:
                return ChainVerification(
                    ok=False,
                    n_checkpoints=len(self._checkpoints),
                    mode="fingerprint",
                    first_bad_step=cp.step,
                    reason=f"checkpoint at step {cp.step} loaded no inspectable model",
                )
            recomputed = compute_fingerprint(model, tokenizer, adapter_name)
            if recomputed != cp.model_fp:
                return ChainVerification(
                    ok=False,
                    n_checkpoints=len(self._checkpoints),
                    mode="fingerprint",
                    first_bad_step=cp.step,
                    reason=(
                        f"fingerprint mismatch at step {cp.step}: loaded weights "
                        f"{recomputed} do not match recorded {cp.model_fp} (weights swapped)"
                    ),
                )
        return ChainVerification(ok=True, n_checkpoints=len(self._checkpoints), mode="fingerprint")

    def verify(self, *, deep: bool = False) -> ChainVerification:
        """Verify the chain (and, when ``deep``, the fingerprints), raising on the first failure.

        Returns the passing `ChainVerification` so a caller may log it. Raises `ProvenanceError` with
        the localized reason when either check fails, which is the guard the sweep relies on: a chain
        that does not verify is never swept, so a developmental result is never read off an untrusted
        run (RK9).
        """
        chain = self.verify_chain()
        if not chain.ok:
            raise ProvenanceError(
                f"checkpoint chain failed verification at step {chain.first_bad_step}: {chain.reason}"
            )
        if deep:
            fp = self.verify_fingerprints()
            if not fp.ok:
                raise ProvenanceError(
                    f"checkpoint fingerprints failed verification at step {fp.first_bad_step}: "
                    f"{fp.reason}"
                )
            return fp
        return chain

    # -- tamper helpers (for tests and audits) ------------------------------

    def tampered(
        self,
        index: int,
        *,
        model_fp: ModelFP | None = None,
        loader: CheckpointLoader | None = None,
        reseal: bool = False,
    ) -> "CheckpointSequence":
        """Return a copy with one checkpoint altered, modelling an attacker's edit.

        With ``reseal=False`` (the default) the links are left untouched, so replacing a recorded
        ``model_fp`` models an edited manifest and `verify_chain` will reject it; replacing only the
        ``loader`` (keeping ``model_fp``) models swapped weights that pass the chain but fail
        `verify_fingerprints`. With ``reseal=True`` the chain is rebuilt around the change, which is
        what an honest re-release would do (and then verification passes). This exists for tests and
        audit drills; production sequences are built by `build`.
        """
        cps = list(self._checkpoints)
        old = cps[index]
        cps[index] = replace(
            old,
            model_fp=model_fp if model_fp is not None else old.model_fp,
            loader=loader if loader is not None else old.loader,
        )
        if reseal:
            triples = [(c.step, c.model_fp, c.loader) for c in cps]
            metas = {c.step: c.meta for c in cps}
            return CheckpointSequence.build(triples, meta=metas)
        return CheckpointSequence(cps)


# ---------------------------------------------------------------------------
# The CPU-provable test vehicle: a planted, growing reward loading
# ---------------------------------------------------------------------------


@dataclass
class SyntheticSequence:
    """A synthetic developmental run with a known planted feature (the CPU vehicle).

    Bundles everything the sweep, the bias-entry curve, and the stabilization detector need to be
    provable without a GPU: the `CheckpointSequence` itself, the fixed evaluation ``view``, the planted
    per-item ``feature`` values (the ground-truth covariate the bias enters along), a ready-made
    ``probe`` naming that feature, and the schedule metadata (``alphas`` the planted loading, and the
    step at which the reward direction was designed to stop rotating). Because the feature is planted
    by construction, "does the bias-entry curve rise" and "does the direction stabilize" have known
    answers, which is exactly the organism-style calibration required before trusting an instrument.
    """

    sequence: CheckpointSequence
    view: list[Any]
    feature: "np.ndarray"
    probe: Any  # dynamics.curves.Probe, typed loosely to avoid an import cycle
    planted_direction: "np.ndarray"
    alphas: list[float]
    scales: list[float]
    expected_stabilization_step: int


def synthetic_planted_sequence(
    *,
    n_checkpoints: int = 8,
    n_items: int = 40,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seed: int = 0,
    planted_coord: int = 7,
    alpha_max: float = 4.0,
    alpha_midpoint: float = 2.0,
    alpha_width: float = 0.7,
    scale_growth: float = 0.6,
    stabilization_eps: float = 1e-3,
) -> SyntheticSequence:
    """Construct the tiny planted-feature checkpoint sequence the tests run on.

    Every checkpoint shares one frozen trunk (a real `LlamaForSequenceClassification`, built from the
    same seed each time) and differs only in its reward head, which is set to ``s_t * normalize(w_base
    + alpha_t * e)``. The planted direction ``e`` is a single trunk coordinate orthogonalized against
    the base head, so ``alpha_t`` is literally how strongly the reward loads onto that feature at step
    ``t``. ``alpha_t`` follows a saturating logistic: the direction rotates fast early and then settles,
    while the scale ``s_t`` keeps growing linearly. That separation is deliberate. The bias-entry curve
    reads ``alpha_t`` through the reward's response to the feature and rises monotonically; the
    stabilization detector, working on the scale-invariant canonical direction, sees the rotation stop
    at the logistic knee even though the raw head keeps growing, which is the "stops rotating versus
    merely rescaling" distinction made concrete.

    Returns a `SyntheticSequence` whose ``sequence`` verifies (chain and fingerprints) and whose
    planted feature is exposed so a test can assert the curve rises and the direction stabilizes.
    Nothing here is a real checkpoint number; it is a controlled instrument calibration.
    """
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification

    from reward_lens.dynamics.curves import Probe
    from reward_lens.signals.loaders import _build_tokenizer, wrap_hf_model

    tokenizer = _build_tokenizer("gpt2")
    vocab_size = int(getattr(tokenizer, "vocab_size", 1000) or 1000)

    def _build_signal(head_vec: "np.ndarray") -> Any:
        # Same seed each call rebuilds the identical frozen trunk; only the head is overwritten, so
        # the checkpoints share internals and differ solely in the planted reward loading.
        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            intermediate_size=2 * d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
            num_labels=1,
            attn_implementation="eager",
        )
        model = LlamaForSequenceClassification(config).eval()
        with torch.no_grad():
            weight = torch.tensor(np.asarray(head_vec), dtype=model.score.weight.dtype)
            model.score.weight.copy_(weight.reshape(1, d_model))
        return wrap_hf_model(
            model,
            tokenizer,
            device="cpu",
            architecture="LlamaForSequenceClassification",
            conformance_quickcheck=False,
        )

    # A fixed, deterministic evaluation set. The text content is incidental; the frozen trunk turns it
    # into fixed activations, and the planted head decides how reward reads them.
    words = [
        "the cat sat",
        "a dog ran",
        "blue sky",
        "red car",
        "green tree",
        "loud noise",
        "soft rain",
        "fast river",
    ]
    view = [(f"q{i}", f"{words[i % len(words)]} {i}") for i in range(n_items)]

    # Base head (the pre-existing reward direction) and the planted feature direction e, orthogonalized
    # so alpha_t is a clean measure of the feature's loading.
    base_signal = _build_signal(np.zeros(d_model, dtype=np.float64) + 1e-3)
    w_base = base_signal.readout("reward").vector.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed + 3)
    w_base = rng.standard_normal(d_model)
    w_base /= np.linalg.norm(w_base)
    e = np.zeros(d_model, dtype=np.float64)
    e[planted_coord % d_model] = 1.0
    e = e - (e @ w_base) * w_base
    e /= np.linalg.norm(e)

    # The planted feature value f_i is the reward the model would assign along e alone, read once
    # through the real score path; it is the ground-truth covariate the bias enters along.
    probe_signal = _build_signal(e)
    feature = probe_signal.score(view).value.values.astype(np.float64)

    def _logistic(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-x)))

    alphas = [
        alpha_max * _logistic((t - alpha_midpoint) / alpha_width) for t in range(n_checkpoints)
    ]
    scales = [1.0 + scale_growth * t for t in range(n_checkpoints)]

    # A ground-truth expectation for the stabilization step, computed from the schedule's own unit
    # directions: the earliest checkpoint from which every subsequent raw-direction rotation is below
    # tolerance. This is scale-free (it normalizes each direction) and so mirrors what the canonical
    # detector sees, giving the test a sensible expectation rather than a brittle alpha-delta count.
    unit_dirs = []
    for a in alphas:
        u = w_base + a * e
        unit_dirs.append(u / np.linalg.norm(u))
    dir_cos = [float(unit_dirs[k] @ unit_dirs[k + 1]) for k in range(len(unit_dirs) - 1)]
    expected_stab = n_checkpoints - 1
    for k0 in range(len(dir_cos)):
        if all((1.0 - c) < stabilization_eps for c in dir_cos[k0:]):
            expected_stab = k0 + 1
            break

    triples: list[tuple[int, ModelFP, CheckpointLoader]] = []
    meta: dict[int, dict[str, Any]] = {}
    for t in range(n_checkpoints):
        direction = w_base + alphas[t] * e
        direction /= np.linalg.norm(direction)
        w_r = (scales[t] * direction).astype(np.float64)
        signal = _build_signal(w_r)
        model_fp = signal.meta.fingerprint

        def _loader(_signal: Any = signal) -> Any:
            return _signal

        triples.append((t, model_fp, _loader))
        meta[t] = {"alpha": float(alphas[t]), "scale": float(scales[t]), "synthetic": True}

    sequence = CheckpointSequence.build(triples, meta=meta)
    probe = Probe(name="planted-feature", feature=feature)
    return SyntheticSequence(
        sequence=sequence,
        view=view,
        feature=feature,
        probe=probe,
        planted_direction=e,
        alphas=alphas,
        scales=scales,
        expected_stabilization_step=expected_stab,
    )


# ---------------------------------------------------------------------------
# The GPU-scale builds: marked, gated, never fabricated
# ---------------------------------------------------------------------------

# The checkpoint-suite-release intent: once trained, the RM-Pythia chain is published
# to the HF hub with its fingerprint chain, so the few-hundred-GPU-hour run happens once and every
# downstream sweep replays cached checkpoints and cached activations rather than retraining.
RM_PYTHIA_RELEASE_INTENT = (
    "The RM-Pythia checkpoint suite (Qwen2.5 0.5B/1.5B/7B and Llama-3.1-8B, 50 to 100 log-spaced "
    "checkpoints carried deliberately through a second epoch) is released to the HF hub with its "
    "fingerprint chain so the training run happens once and all developmental sweeps replay cached "
    "checkpoints and activations. Not yet trained; requires flagship GPUs."
)


def train_rm_pythia(
    *,
    bases: tuple[str, ...] = (
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-3.1-8B",
    ),
    n_checkpoints: int = 64,
    epochs: int = 2,
    log_spaced: bool = True,
    output_dir: str | None = None,
    allow_gpu: bool = True,
    **_: Any,
) -> "CheckpointSequence":
    """Train the RM-Pythia checkpoint suite. GPU-gated; refuses without CUDA.

    This is the real developmental substrate: a preference-model training run over ``bases``, saving
    ``n_checkpoints`` log-spaced snapshots and continuing deliberately into a second epoch so the
    second-epoch-collapse autopsy (`curves.second_epoch_collapse_autopsy`) has data. Each saved
    checkpoint is fingerprinted and linked into a `CheckpointSequence`, then the suite is released
    per `RM_PYTHIA_RELEASE_INTENT`. The training loop is a few-hundred-GPU-hour build on flagship
    hardware and is not run on this machine; the function raises rather than fabricate a checkpoint,
    which is the standing rule that a GPU-scale number is never faked.
    """
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch absent is just another "no CUDA" here
        has_cuda = False
    if not (allow_gpu and has_cuda):
        raise RuntimeError(
            "train_rm_pythia is GPU-gated and needs flagship CUDA hardware; no usable CUDA device is "
            "available here. It is a few-hundred-GPU-hour build and is never simulated "
            "on CPU. Use synthetic_planted_sequence for the CPU-provable developmental vehicle, or run "
            "this on adequate GPUs. " + RM_PYTHIA_RELEASE_INTENT
        )
    raise NotImplementedError(  # pragma: no cover - only reachable on flagship GPUs
        "The RM-Pythia training loop runs only on flagship GPUs and its outputs are released as the "
        "cached checkpoint suite; wire it to the training pipeline at run time. " + str(bases)
    )


def from_hf_revisions(
    repo_id: str,
    revisions: list[tuple[int, str]],
    *,
    allow_download: bool = False,
    **load_kwargs: Any,
) -> "CheckpointSequence":
    """Build a `CheckpointSequence` from a chain of HF hub revisions. Download-gated.

    ``revisions`` pairs each training ``step`` with an HF revision (a branch, tag, or commit) of
    ``repo_id``, which is how the released RM-Pythia suite is consumed: one repository, many revisions,
    each a checkpoint. The loader for each revision defers to `signals.load_signal`, whose large-model
    load is itself gated on this hardware, so this function refuses to fan out downloads unless
    ``allow_download=True`` on a machine that can hold the models. The fingerprint for each revision is
    computed at first load; until a revision is loaded its recorded fingerprint is the revision string,
    so the chain over revisions is still well-formed and inspectable.
    """
    if not allow_download:
        raise NotImplementedError(
            f"from_hf_revisions('{repo_id}', {len(revisions)} revisions) would download a chain of "
            f"checkpoints from the HF hub; that is gated on this hardware. Set allow_download=True "
            f"on a machine that can hold the models, or use "
            f"synthetic_planted_sequence for the CPU vehicle."
        )

    from reward_lens.signals.loaders import SignalSpec, load_signal  # pragma: no cover - GPU path

    triples: list[tuple[int, ModelFP, CheckpointLoader]] = []  # pragma: no cover
    meta: dict[int, dict[str, Any]] = {}
    for step, revision in revisions:  # pragma: no cover - only runs with downloads enabled
        spec = SignalSpec(source=repo_id, allow_download=True, **load_kwargs)

        def _loader(_spec: Any = spec, _rev: str = revision) -> Any:
            from dataclasses import replace as _replace

            return load_signal(_replace(_spec, meta_extra={"revision": _rev}))

        probe_signal = _loader()
        triples.append((step, probe_signal.meta.fingerprint, _loader))
        meta[step] = {"repo_id": repo_id, "revision": revision}
    return CheckpointSequence.build(triples, meta=meta)


__all__ = [
    "GENESIS_LINK",
    "Checkpoint",
    "CheckpointLoader",
    "ChainVerification",
    "CheckpointSequence",
    "SyntheticSequence",
    "synthetic_planted_sequence",
    "train_rm_pythia",
    "from_hf_revisions",
    "RM_PYTHIA_RELEASE_INTENT",
]
