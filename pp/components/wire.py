"""Deprecated
"""

from typing import Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.hline import hline
from pp.layers import LAYER
from pp.port import deco_rename_ports
from pp.types import Number

WIRE_WIDTH = 10.0


@deco_rename_ports
@cell
def wire_straight(
    length: Number = 50.0,
    width: Number = WIRE_WIDTH,
    layer: Tuple[int, int] = LAYER.M3,
    port_type: str = "dc",
    **kwargs
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
def wire_corner(
    width: float = WIRE_WIDTH,
    layer: Tuple[int, int] = LAYER.M3,
    port_type: str = "dc",
    **kwargs
) -> Component:
    """90 degrees electrical corner

    Args:
        width: wire width
        layer: layer
        port_type: port_type
        kwargs: for bend radius

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

    c = wire_straight()
    c = wire_corner()
    c.show()
