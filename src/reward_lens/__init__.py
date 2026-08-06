"""reward-lens: the reference instrument for the science of reward misspecification.

The library is one kernel of subsystems (``core``, ``stats``, ``access``, ``runtime``, ``signals``,
``data``, ``concepts``, ``interventions``, ``geometry``, ``measure``, ``attribution``, ``organisms``,
``dynamics``, ``loops``, ``studies``, ``artifacts``, ``operate``), with the three gates
(calibration, gauge, registration) enforced as runtime policy in the stats and evidence layer.

Import discipline: this top-level module imports nothing. ``import reward_lens``,
``import reward_lens.core`` and ``import reward_lens.stats`` pull nothing heavier than numpy, so the
pure epistemics layer is usable without torch, and a test asserts that in a fresh subprocess rather
than trusting it. Anything that touches models is imported directly from its subsystem
(``from reward_lens.signals import load_signal``), which is where the white-box extra is required.

The public surface here is ``__version__`` plus the two torch-free subsystems ``core`` and ``stats``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "3.0.0"


if TYPE_CHECKING:  # help static analysis without importing torch at runtime
    from reward_lens import core as core
    from reward_lens import stats as stats

__all__ = ["__version__", "core", "stats"]
