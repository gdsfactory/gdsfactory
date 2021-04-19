from typing import Optional, Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.hline import hline
from pp.layers import LAYER
from pp.port import deco_rename_ports
from pp.types import Number

WIRE_WIDTH = 10.0


@deco_rename_ports
@cell
def wire(
    length: Number = 50.0,
    width: Number = WIRE_WIDTH,
    layer: Tuple[int, int] = LAYER.M3,
    port_type: str = "dc",
) -> Component:
    """Straight straight.

    Args:
        length: straiht length
        width: wire width
        layer: layer
        port_type: port_type
    """
    return hline(length=length, width=width, layer=layer, port_type=port_type)


@deco_rename_ports
@cell
def corner(
    width: float = WIRE_WIDTH,
    radius: Optional[Number] = None,
    layer: Tuple[int, int] = LAYER.M3,
    port_type: str = "dc",
) -> Component:
    """90 degrees electrical bend

    Args:
        width: wire width
        radius ignore (passed for consistency with other types of bends)
        layer: layer
        port_type: port_type

    """
    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(
        name="W0",
        midpoint=(-a, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="N0",
        midpoint=(0, a),
        width=width,
        orientation=90,
        layer=layer,
        port_type=port_type,
    )
    c.length = width
    return c


if __name__ == "__main__":

    c = wire()
    c = corner()
    c.show()
