from __future__ import annotations

__all__ = ["cross"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def cross(
    length: float = 10.0,
    width: float = 3.0,
    layer: LayerSpec = "WG",
    port_type: str | None = None,
) -> Component:
    """Returns a cross from two rectangles of length and width.

    Args:
        length: float Length of the cross from one end to the other.
        width: float Width of the arms of the cross.
        layer: layer for geometry.
        port_type: None, optical, electrical.
    """
    return cf.cross(length=length, width=width, layer=layer, port_type=port_type)
