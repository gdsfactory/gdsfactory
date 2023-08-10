from __future__ import annotations

from functools import partial

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.typings import Ints, LayerSpec


@cell
def rectangle(
    size=(4.0, 2.0),
    layer: LayerSpec = "WG",
    centered: bool = False,
    port_type: str | None = "electrical",
    port_orientations: Ints | None = (180, 90, 0, -90),
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


@cell
def rectangles(
    size=(4.0, 2.0),
    offsets=(0, 1),
    layers=("WG", "SLAB150"),
    centered: bool = True,
    **kwargs,
) -> Component:
    """Returns overimposed rectangles.

    Args:
        size: (tuple) Width and height of rectangle.
        layers: Specific layer to put polygon geometry on.
        offsets: list of offsets.
        centered: True sets center to (0, 0), False sets south-west of first rectangle to (0, 0).

    Keyword Args:
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add.

    .. code::

            ┌──────────────┐
            │              │
            │   ┌──────┐   │
            │   │      │   │
            │   │      ├───►
            │   │      │offset
            │   └──────┘   │
            │              │
            └──────────────┘

    """
    c = Component()
    size = np.array(size)

    ref0 = None

    if len(offsets) != len(layers):
        raise ValueError(f"len(offsets) != len(layers) {len(offsets)} != {len(layers)}")
    for layer, offset in zip(layers, offsets):
        ref = c << rectangle(
            size=size + 2 * offset, layer=layer, centered=centered, **kwargs
        )
        if ref0:
            ref.center = ref0.center
        ref0 = ref

    return c


if __name__ == "__main__":
    c = rectangles(offsets=(0, 1), centered=False)
    # c = rectangle(size=(3, 2), centered=False, layer=(2, 3))
    # c = rectangle(size=(3, 2), centered=True, layer=(2, 3))
    print(c.ports)
    print(c.name)
    c.show(show_ports=True)
