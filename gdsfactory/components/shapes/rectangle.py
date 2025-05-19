from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Ints, LayerSpec, LayerSpecs, Size


@gf.cell_with_module_name
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
    c = Component()
    ref = c << gf.c.compass(
        size=size, layer=layer, port_type=port_type, port_orientations=port_orientations
    )
    if not centered:
        # Avoid tuple creation and division in move arg; reuse size values.
        x_shift, y_shift = size[0] * 0.5, size[1] * 0.5
        ref.move((x_shift, y_shift))
    if port_type:
        c.add_ports(ref.ports)
    c.flatten()
    return c


fiber_size = 10.4
marker_te = partial(rectangle, size=(fiber_size, fiber_size), layer="TE", centered=True)
marker_tm = partial(rectangle, size=(fiber_size, fiber_size), layer="TM", centered=True)


@gf.cell_with_module_name
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
    size_np = np.array(size, dtype=np.float64)
    ref0 = None
    offsets = offsets or [0] * len(layers)

    if len(offsets) != len(layers):
        raise ValueError(f"len(offsets) != len(layers) {len(offsets)} != {len(layers)}")
    for layer, offset in zip(layers, offsets):
        current_size = size_np + 2 * offset
        ref = c << rectangle(
            size=(current_size[0], current_size[1]),
            layer=layer,
            centered=centered,
            **kwargs,
        )
        if ref0:
            ref.center = ref0.center
        ref0 = ref

    return c


def _get_rotated_basis(angle: float):
    """Fast helper, used by route_quad"""
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c, s]), np.array([-s, c])


if __name__ == "__main__":
    # c = rectangle(size=(3, 2), centered=False)
    # c = rectangles(offsets=(0, 1), centered=False)
    # c = rectangle(size=(3, 2), centered=False, layer=(2, 3))
    # c = rectangle(size=(3, 2), centered=True, layer=(2, 3))
    c = rectangle(port_type=None)
    print(c.settings)
    # print(c.ports)
    # print(c.name)
    c.show()
