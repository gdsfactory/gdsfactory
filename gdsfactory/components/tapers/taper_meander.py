"""Meander taper for superconducting nanowires.

Adapted from PHIDL <https://github.com/amccaugh/phidl/> by Adam McCaughan
"""

from __future__ import annotations

__all__ = ["taper_meander"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec

from .._schematic import taper_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
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
    return cf.taper_meander(
        x_taper=x_taper,
        w_taper=w_taper,
        meander_length=meander_length,
        spacing_factor=spacing_factor,
        min_spacing=min_spacing,
        layer=layer,
    )


if __name__ == "__main__":
    x_taper = (1, 10, 20, 30, 40, 50)
    w_taper = (1, 5, 10, 5, 2, 1)
    c = taper_meander(x_taper=x_taper, w_taper=w_taper)
    c.show()
