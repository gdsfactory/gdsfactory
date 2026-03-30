"""MaxPowerLoss constraint - limits total optical power loss across routes."""

from __future__ import annotations

from gdsfactory.constraints.base import Constraint, register_constraint


@register_constraint("max_power_loss")
class MaxPowerLoss(Constraint):
    """Constraint that limits total power loss across all routes.

    Power loss is estimated as route length multiplied by a per-unit-length
    loss coefficient.
    """

    max_loss_db: float = 3.0
    loss_per_unit_length_db: float = 0.001

    def is_satisfied(self) -> bool:
        """Return True if total estimated loss is within the limit."""
        total_loss = sum(
            route.length * self.loss_per_unit_length_db for route in self.routes
        )
        return total_loss <= self.max_loss_db
