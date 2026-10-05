from __future__ import annotations

__all__ = ["hexagon", "octagon", "regular_polygon"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
def regular_polygon(
    sides: int = 6,
    side_length: float = 10,
    layer: LayerSpec = "WG",
    port_width: float
    | None = None,  # port width doesn't need to be side length all the time
    port_type: str | None = "placement",
) -> Component:
    """Returns a regular N-sided polygon, with ports on each edge.

    Args:
        sides: number of sides for the polygon.
        side_length: of the edges.
        layer: Specific layer to put polygon geometry on.
        port_width: the width of port of the polygon (in electrical pads, the port width may not equal to the side length).
        port_type: optical, electrical.
    """
    return cf.regular_polygon(
        sides=sides,
        side_length=side_length,
        layer=layer,
        port_width=port_width,
        port_type=port_type,
    )


hexagon = CellAlias(regular_polygon, sides=6)
octagon = CellAlias(regular_polygon, sides=8)
