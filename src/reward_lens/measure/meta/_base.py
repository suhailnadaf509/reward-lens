"""One shared shape for the six meta-instruments, and the one thing they add to it.

The seam itself is `ControlInstrument`: preflight, compute once, refuse or emit. It was written for
the controls bank and series A reuses it, so a third copy here would be a third place for the gate
logic to drift. What is added is a single hook.

`MetaInstrument.uncertainty` lets a subclass put its interval on the `Evidence` rather than only
inside the value payload. That matters more here than anywhere else in the library, because five of
these six instruments produce a number whose whole content is an uncertainty: a noise floor, a
budget, a between-laboratory spread, an increment, a disagreement between rungs. A reading of that
kind with no interval on it is not a weaker reading, it is a different and worse claim, and the
`Uncertainty` field is where anything reading the store generically will look for one.

The same call forwards the `baselines` key out of the payload onto the `Evidence`'s own field, for
the reason `gstudy.py` gives: a baseline that lives only inside a value dict is invisible to
anything reading the store from outside the instrument, and the mandatory-baseline rule exists to
be checkable from outside.
"""

from __future__ import annotations

from typing import Any, Mapping

from reward_lens.core.evidence import Evidence, Uncertainty
from reward_lens.core.reading import Refusal
from reward_lens.measure.base import Context
from reward_lens.measure.controls._base import ControlInstrument


class MetaInstrument(ControlInstrument):
    """A control instrument that carries its own interval and its own baselines onto the reading."""

    def uncertainty(self, computed: Any) -> Uncertainty | None:
        """The interval, where the instrument has one. `None` is a real answer.

        M7 returns None: its reading *is* an uncertainty budget, and wrapping a combined standard
        uncertainty in a second interval of its own would be a claim nothing in the table supports.
        """
        return None

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value with its remedy."
            )
        payload = self.payload(out)
        declared = payload.get("baselines")
        return ctx.emit(
            payload,
            uncertainty=self.uncertainty(out),
            baselines=declared if isinstance(declared, Mapping) else None,
        )


__all__ = ["MetaInstrument"]
