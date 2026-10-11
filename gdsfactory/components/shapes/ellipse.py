from __future__ import annotations

__all__ = ["ellipse"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def ellipse(
    radii: tuple[float, float] = (10.0, 5.0),
    angle_resolution: float = 2.5,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns ellipse component.

    Args:
        radii: Semimajor and semiminor axis lengths of the ellipse.
        angle_resolution: number of degrees per point.
        layer: Specific layer(s) to put polygon geometry on.

    The orientation of the ellipse is determined by the order of the radii variables;
    if the first element is larger, the ellipse will be horizontal and if the second
    element is larger, the ellipse will be vertical.
    """
    return cf.ellipse(radii=radii, angle_resolution=angle_resolution, layer=layer)
