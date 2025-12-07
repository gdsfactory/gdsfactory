"""Helper functions for RF layout.

Adapted from PHIcL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["ytron_round"]

import numpy as np
from numpy import cos, pi, sin

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
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
    http://dx.doi.org/10.1021/acs.nanolett.6b03593

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
    # ==========================================================================
    #  Create the basic geometry
    # ==========================================================================
    theta = theta * pi / 180
    theta_resolution = theta_resolution * pi / 180
    thetalist = np.linspace(
        -(pi - theta), -theta, int((pi - 2 * theta) / theta_resolution) + 2
    )
    semicircle_x = rho * cos(thetalist)
    semicircle_y = rho * sin(thetalist) + rho

    # Rest of yTron
    xc = rho * cos(theta)
    yc = rho * sin(theta)
    arm_x_left = arm_lengths[0] * sin(theta)
    arm_y_left = arm_lengths[0] * cos(theta)
    arm_x_right = arm_lengths[1] * sin(theta)
    arm_y_right = arm_lengths[1] * cos(theta)

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [
        xc + arm_x_right,
        xc + arm_x_right + arm_widths[1],
        xc + arm_widths[1],
        xc + arm_widths[1],
        0,
        -(xc + arm_widths[0]),
        -(xc + arm_widths[0]),
        -(xc + arm_x_left + arm_widths[0]),
        -(xc + arm_x_left),
    ]
    ypts = semicircle_y.tolist() + [
        yc + arm_y_right,
        yc + arm_y_right,
        yc,
        yc - source_length,
        yc - source_length,
        yc - source_length,
        yc,
        yc + arm_y_left,
        yc + arm_y_left,
    ]

    # ==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    # ==========================================================================
    c = gf.Component()
    c.add_polygon(list(zip(xpts, ypts, strict=True)), layer=layer)
    c.add_port(
        name="left",
        center=(-(xc + arm_x_left + arm_widths[0] / 2), yc + arm_y_left),
        width=arm_widths[0],
        orientation=90,
        layer=layer,
    )
    c.add_port(
        name="right",
        center=(xc + arm_x_right + arm_widths[1] / 2, yc + arm_y_right),
        width=arm_widths[1],
        orientation=90,
        layer=layer,
    )
    c.add_port(
        name="source",
        center=(0 + (arm_widths[1] - arm_widths[0]) / 2, -source_length + yc),
        width=arm_widths[0] + arm_widths[1] + 2 * xc,
        orientation=-90,
        layer=layer,
    )

    # ==========================================================================
    #  Record any parameters you may want to access later
    # ==========================================================================
    c.info["rho"] = rho
    c.info["left_width"] = arm_widths[0]
    c.info["right_width"] = arm_widths[1]
    c.info["source_width"] = arm_widths[0] + arm_widths[1] + 2 * xc
    return c


if __name__ == "__main__":
    c = ytron_round()
    c.show()
