"""Meander taper for superconducting nanowires.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["taper_meander"]

from math import pi

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def taper_meander(
    x_taper: tuple[float, ...] | None = None,
    w_taper: tuple[float, ...] | None = None,
    meander_length: float = 1000,
    spacing_factor: float = 3,
    min_spacing: float = 0.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Create a meander from arrays of x-positions and widths.

    Typically used for creating meandered tapers.

    Args:
        x_taper: The x-coordinates of the data points, must be increasing.
        w_taper: The widths at each x-coordinate, same length as x_taper.
        meander_length: Length of each section of the meander.
        spacing_factor: Multiplicative spacing factor between adjacent meanders.
        min_spacing: Minimum spacing between adjacent meanders.
        layer: Specific layer(s) to put polygon geometry on.

    Returns:
        Component containing the meandered taper.
    """
    # Default taper if none provided
    x_taper = x_taper or (1, 10, 20, 30, 40, 50)
    w_taper = w_taper or (1, 5, 10, 5, 2, 1)

    # Convert to numpy arrays for internal use
    x_taper_arr = np.array(x_taper)
    w_taper_arr = np.array(w_taper)

    def taper_width(x: float) -> float:
        """Interpolate width at a given x position."""
        return float(np.interp(x, x_taper_arr, w_taper_arr))

    @gf.cell
    def taper_section(
        x_start: float, x_end: float, num_pts: int = 30, layer: LayerSpec = layer
    ) -> Component:
        """Create a single taper section.

        Args:
            x_start: Starting x-coordinate.
            x_end: Ending x-coordinate.
            num_pts: Number of points for the taper.
            layer: Layer for the polygon.

        Returns:
            Component containing the taper section.
        """
        c = gf.Component()
        length = x_end - x_start
        x = np.linspace(0, length, num_pts)
        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
        xpts = np.concatenate([x, x[::-1]])
        ypts = np.concatenate([widths / 2, -widths[::-1] / 2])
        points = np.column_stack((xpts, ypts))
        c.add_polygon(points, layer=layer)

        # Snap port widths to even multiples of 0.002 um (2 dbu)
        width1_snapped = round(widths[0] / 0.002) * 0.002
        width2_snapped = round(widths[-1] / 0.002) * 0.002
        c.add_port(
            name="o1",
            center=(0, 0),
            width=width1_snapped,
            orientation=180,
            layer=layer,
        )
        c.add_port(
            name="o2",
            center=(length, 0),
            width=width2_snapped,
            orientation=0,
            layer=layer,
        )
        return c

    @gf.cell
    def arc_tapered(
        radius: float = 10,
        width1: float = 1,
        width2: float = 2,
        theta: float = 45,
        angle_resolution: float = 2.5,
        layer: LayerSpec = layer,
    ) -> Component:
        """Create a tapered arc section.

        Args:
            radius: Radius of the arc.
            width1: Width at the start of the arc.
            width2: Width at the end of the arc.
            theta: Angle of the arc in degrees.
            angle_resolution: Angular resolution in degrees.
            layer: Layer for the polygon.

        Returns:
            Component containing the tapered arc.
        """
        c = gf.Component()
        path1 = gf.path.arc(
            radius=radius,
            angle=theta,
            angular_step=angle_resolution,
        )
        # Snap widths to even multiples of 0.002 um (2 dbu)
        width1_snapped = round(width1 / 0.002) * 0.002
        width2_snapped = round(width2 / 0.002) * 0.002
        # Extrude the path with the first width
        arc_component = gf.path.extrude(path1, width=width1_snapped, layer=layer)
        c.add_ref(arc_component)
        c.add_port(
            name="o1",
            center=(0, 0),
            width=width1_snapped,
            orientation=180,
            layer=layer,
        )
        c.add_port(
            name="o2",
            center=(path1.x, path1.y),
            width=width2_snapped,
            orientation=path1.end_angle + 90,
            layer=layer,
        )
        return c

    c = gf.Component()
    xpos1 = min(x_taper_arr)
    xpos2 = min(x_taper_arr) + meander_length
    t = c.add_ref(taper_section(x_start=xpos1, x_end=xpos2, num_pts=50, layer=layer))
    c.add_port(port=t.ports["o1"], name="o1")
    dir_toggle = -1
    while xpos2 < max(x_taper_arr):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor * arc_width1, min_spacing)
        arc_length = pi * arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        A = arc_tapered(
            radius=arc_radius,
            width1=arc_width1,
            width2=arc_width2,
            theta=180 * dir_toggle,
            layer=layer,
        )
        a = c.add_ref(A)
        a.connect("o1", t.ports["o2"])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = c.add_ref(
            taper_section(x_start=xpos1, x_end=xpos2, num_pts=30, layer=layer)
        )
        t.connect("o1", a.ports["o2"])
    c.add_port(port=t.ports["o2"], name="o2")
    return c


if __name__ == "__main__":
    x_taper = (1, 10, 20, 30, 40, 50)
    w_taper = (1, 5, 10, 5, 2, 1)
    c = taper_meander(x_taper=x_taper, w_taper=w_taper)
    c.show()
