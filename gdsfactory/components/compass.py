from typing import Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.types import Layer


@cell
def compass(
    size: Tuple[float, float] = (4.0, 2.0),
    layer: Layer = gf.LAYER.WG,
    port_type: str = "electrical",
) -> Component:
    """Rectangular contact pad with centered ports on rectangle edges
    (north, south, east, and west)

    Args:
        size: rectangle size
        layer: tuple (int, int)
        port_type:
    """
    c = gf.Component()
    r = gf.components.rectangle(size=size, layer=layer)
    dx = size[0]
    dy = size[1]

    rr = r.ref(position=(-dx / 2, -dy / 2))
    c.add(rr)
    c.add_port(
        name="e1",
        midpoint=[-dx / 2, 0],
        width=dy,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="e2",
        midpoint=[0, dy / 2],
        width=dx,
        orientation=90,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="e3",
        midpoint=[dx / 2, 0],
        width=dy,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="e4",
        midpoint=[0, -dy / 2],
        width=dx,
        orientation=-90,
        layer=layer,
        port_type=port_type,
    )
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = compass(size=(1, 2), layer=gf.LAYER.WG, port_type="optical")
    c.show()
