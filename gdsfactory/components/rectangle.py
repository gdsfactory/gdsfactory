from __future__ import annotations

from functools import partial
from typing import Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.typings import Ints, LayerSpec


@cell
def rectangle(
    size=(4.0, 2.0),
    layer: LayerSpec = "WG",
    centered: bool = False,
    port_type: Optional[str] = "electrical",
    port_orientations: Optional[Ints] = (180, 90, 0, -90),
) -> Component:
    """Returns a rectangle.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0).
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add.
    """
    c = Component()
    ref = c << compass(
        size=size, layer=layer, port_type=port_type, port_orientations=port_orientations
    )
    if not centered:
        ref.move((size[0] / 2, size[1] / 2))
    if port_type:
        c.add_ports(ref.ports)
    return c


fiber_size = 10.4
marker_te = partial(rectangle, size=[fiber_size, fiber_size], layer="TE", centered=True)
marker_tm = partial(rectangle, size=[fiber_size, fiber_size], layer="TM", centered=True)


if __name__ == "__main__":
    c = rectangle(size=(3, 2), centered=False, layer=(2, 3))
    # c = rectangle(size=(3, 2), centered=True, layer=(2, 3))
    print(c.ports)
    print(c.name)
    c.show(show_ports=True)
