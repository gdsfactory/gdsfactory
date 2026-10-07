from __future__ import annotations

__all__ = ["L"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def L(
    width: int | float = 1,
    size: tuple[int, int] = (10, 20),
    layer: LayerSpec = "MTOP",
    port_type: str = "electrical",
) -> Component:
    """Generates an 'L' geometry with ports on both ends.

    Based on phidl.

    Args:
        width: of the line.
        size: length and height of the base.
        layer: spec.
        port_type: for port.
    """
    return cf.L(width=width, size=size, layer=layer, port_type=port_type)
