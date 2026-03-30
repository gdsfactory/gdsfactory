"""EqualFixedLength constraint - enforces equal or fixed route lengths."""

from __future__ import annotations

from gdsfactory.constraints.base import Constraint, register_constraint


@register_constraint("equal_fixed_length")
class EqualFixedLength(Constraint):
    """Constraint that enforces all routes have equal length.

    If target_length is set, all routes must match that value within tolerance.
    Otherwise, all routes must match each other within tolerance.
    """

    target_length: float | None = None
    tolerance: float = 0.1

    def is_satisfied(self) -> bool:
        """Return True if all route lengths are within tolerance."""
        if not self.routes:
            return True

        lengths = [route.length for route in self.routes]

        reference = self.target_length if self.target_length is not None else lengths[0]
        return all(abs(length - reference) <= self.tolerance for length in lengths)
