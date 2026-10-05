"""Helper functions for RF layout.

Adapted from PHIcL <https://github.com/amccaugh/phidl/> by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["ytron_round"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["superconductors"])
def ytron_round(
    rho: float = 1,
    arm_lengths: tuple[float, float] = (500, 300),
    source_length: float = 500,
    arm_widths: tuple[float, float] = (200, 200),
    theta: float = 2.5,
    theta_resolution: float = 10,
    layer: LayerSpec = "WG",
) -> Component:
    """Ytron structure for superconducting nanowires.

    McCaughan, A. N., Abebe, N. S., Zhao, Q.-Y. & Berggren, K. K.
    Using Geometry To Sense Current. Nano Lett. 16, 7626-7631 (2016).
    <http://dx.doi.org/10.1021/acs.nanolett.6b03593>

    Args:
        rho: Radius of curvature of ytron intersection point.
        arm_lengths: Lengths of the left and right arms of the yTron, respectively.
        source_length: Length of the source of the yTron.
        arm_widths: Widths of the left and right arms of the yTron, respectively.
        theta: Angle between the two yTron arms.
        theta_resolution: Angle resolution for curvature of ytron intersection point.
        layer: Specific layer(s) to put polygon geometry on.

    Returns:
        Component containing a yTron geometry.
    """
    return cf.ytron_round(
        rho=rho,
        arm_lengths=arm_lengths,
        source_length=source_length,
        arm_widths=arm_widths,
        theta=theta,
        theta_resolution=theta_resolution,
        layer=layer,
    )


if __name__ == "__main__":
    c = ytron_round()
    c.show()
