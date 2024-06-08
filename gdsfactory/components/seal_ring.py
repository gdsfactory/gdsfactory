from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import ComponentFactory

Float2 = tuple[float, float]
Coordinate = tuple[Float2, Float2]


@gf.cell
def seal_ring(
    component: gf.Component | gf.Instance | ComponentFactory = "rectangle",
    seal: gf.typings.ComponentSpec = "via_stack",
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
        component: to add seal ring around. You can pass Component.bbox.
        seal: function for the seal.
        width: of the seal.
        padding: from component to seal.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    c = gf.Component()

    component = gf.get_component(component)
    bbox = component.dbbox()
    xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

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
        north = c << gf.get_component(seal, size=size_north_south)
        north.dymin = ymin_north
        north.dx = x

    if with_east:
        east = c << gf.get_component(seal, size=size_east_west)
        east.dxmin = xmax + padding
        east.dymax = ymin_north

    if with_west:
        west = c << gf.get_component(seal, size=size_east_west)
        west.dxmax = xmin - padding
        west.dymax = ymin_north

    if with_south:
        south = c << gf.get_component(seal, size=size_north_south)
        south.dymax = ymax_south
        south.dx = x

    return c


if __name__ == "__main__":
    # big_square = partial(rectangle, size=(1300, 2600))
    # c = gf.Component("demo")
    # c << big_square()
    # c << seal_ring(c, with_south=False)
    c = seal_ring()
    c.show()
