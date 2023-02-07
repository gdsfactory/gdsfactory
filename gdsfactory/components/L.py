from __future__ import annotations

from typing import Tuple, Union

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
def L(
    width: Union[int, float] = 1,
    size: Tuple[int, int] = (10, 20),
    layer: LayerSpec = "M3",
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
    D = Component()
    w = width / 2
    s1, s2 = size
    points = [(-w, -w), (s1, -w), (s1, w), (w, w), (w, s2), (-w, s2), (-w, -w)]
    D.add_polygon(points, layer=layer)
    D.add_port(
        name="e1",
        center=(0, s2),
        width=width,
        orientation=90,
        port_type=port_type,
        layer=layer,
    )
    D.add_port(
        name="e2",
        center=(s1, 0),
        width=width,
        orientation=0,
        port_type=port_type,
        layer=layer,
    )
    return D


if __name__ == "__main__":
    c = L()
    c.show(show_ports=True)
