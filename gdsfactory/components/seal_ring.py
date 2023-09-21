from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.via_stack import via_stack
from gdsfactory.snap import snap_to_grid

Float2 = tuple[float, float]
Coordinate = tuple[Float2, Float2]


@gf.cell_without_validator
def seal_ring(
    bbox: tuple[Coordinate, Coordinate] = ((-1.0, -1.0), (3.0, 4.0)),
    seal: gf.typings.ComponentSpec = via_stack,
    width: float = 10,
    padding: float = 10.0,
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
) -> gf.Component:
    """Returns a continuous seal ring boundary at the chip/die.

    Prevents cracks from spreading and shields when connected to ground.

    Args:
        bbox: to add seal ring around. You can pass Component.bbox.
        seal: function for the seal.
        width: of the seal.
        padding: from component to seal.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    c = gf.Component()
    (xmin, ymin), (xmax, ymax) = bbox

    x = (xmax + xmin) / 2

    sx = xmax - xmin
    sy = ymax - ymin

    snap = partial(snap_to_grid, grid_factor=2)
    sx = snap(sx)
    sy = snap(sy)

    ymin_north = snap(ymax + padding)
    ymax_south = snap(ymax - sy - padding)

    # north south
    size_north_south = (sx + 2 * padding + 2 * width, width)
    size_east_west = (width, sy + 2 * padding)

    if with_north:
        north = c << seal(size=size_north_south)
        north.ymin = ymin_north
        north.x = x

    if with_east:
        east = c << seal(size=size_east_west)
        east.xmin = xmax + padding
        east.ymax = ymin_north

    if with_west:
        west = c << seal(size=size_east_west)
        west.xmax = xmin - padding
        west.ymax = ymin_north

    if with_south:
        south = c << seal(size=size_north_south)
        south.ymax = ymax_south
        south.x = x

    return c


if __name__ == "__main__":
    big_square = partial(rectangle, size=(1300, 2600))
    c = gf.Component("demo")
    c << big_square()
    c << seal_ring(c.bbox + ((0, 0), (10, 0)), with_south=False)
    c.show(show_ports=True)
