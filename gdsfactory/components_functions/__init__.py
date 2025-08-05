"""Component functions without decorators.

This module contains pure functions that create components without any decorators.
The actual components module will import these and apply decorators as needed.
"""

from gdsfactory.components_functions.waveguides import (
    straight_all_angle_function,
    straight_function,
)

__all__ = [
    "straight_all_angle_function",
    "straight_function",
]
