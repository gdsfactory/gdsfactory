from __future__ import annotations

__all__ = ["fiber_size", "marker_te", "marker_tm", "rectangle", "rectangles"]

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import Ints, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name(tags=["shapes"])
def rectangle(
    size: Size = (4.0, 2.0),
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
        port_orientations: list of port_orientations to add. None adds no ports.
    """
    return cf.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
    )


fiber_size = 10.4
marker_te = CellAlias(
    rectangle, size=(fiber_size, fiber_size), layer="TE", centered=True
)
marker_tm = CellAlias(
    rectangle, size=(fiber_size, fiber_size), layer="TM", centered=True
)


@gf.cell_with_module_name(tags=["shapes"])
def rectangles(
    size: Size = (4.0, 2.0),
    offsets: Sequence[float] | None = None,
    layers: LayerSpecs = ("WG", "SLAB150"),
    centered: bool = True,
    **kwargs: Any,
) -> Component:
    """Returns overimposed rectangles.

    Args:
        size: (tuple) Width and height of rectangle.
        layers: Specific layer to put polygon geometry on.
        offsets: list of offsets. If None, all rectangles have a zero offset.
        centered: True sets center to (0, 0), False sets south-west of first rectangle to (0, 0).
        kwargs: additional arguments to pass to rectangle.

    Keyword Args:
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add.

    ```text
            ┌──────────────┐
            │              │
            │   ┌──────┐   │
            │   │      │   │
            │   │      ├───►
            │   │      │offset
            │   └──────┘   │
            │              │
            └──────────────┘
    ```

    """
    return cf.rectangles(
        size=size, offsets=offsets, layers=layers, centered=centered, **kwargs
    )
