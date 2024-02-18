from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.via_stack import (
    via_stack,
    via_stack_corner45_extended,
    via_stack_m1_m3,
)
from gdsfactory.snap import snap_to_grid

Float2 = tuple[float, float]
Coordinate = tuple[Float2, Float2]


@gf.cell
def seal_ring(
    bbox=((-1.0, -1.0), (3.0, 4.0)),
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


@gf.cell
def seal_ring_segmented(
    bbox=((-1.0, -1.0), (3.0, 4.0)),
    length_segment: float = 10,
    width_segment: float = 3,
    spacing_segment: float = 2,
    corner: gf.Component = via_stack_corner45_extended,
    via_stack: gf.Component = via_stack_m1_m3,
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
    padding: float = 10.0,
) -> gf.Component:
    """Segmented Seal ring.

    Args:
        size: size of the seal ring.
        length_segment: length of each segment.
        width_segment: width of each segment.
        spacing_segment: spacing between segments.
        corner: corner component.
        via_stack: via_stack component.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
        padding: from component to seal.
    """
    c = gf.Component()
    corner = gf.get_component(corner, width=width_segment)

    (xmin, ymin), (xmax, ymax) = bbox

    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    tl = c << corner
    tr = c << corner

    tl.xmin = xmin
    tl.ymax = ymax

    tr.mirror()
    tr.xmax = xmax
    tr.ymax = ymax

    bl = c << corner
    br = c << corner
    br.mirror()
    br.mirror_y()
    bl.mirror_y()

    bl.xmin = xmin
    bl.ymin = ymin

    br.xmax = xmax
    br.ymin = ymin

    pitch = length_segment + spacing_segment

    # horizontal
    dx = abs(tl.xmax - tr.xmin)
    segment_horizontal = via_stack(size=(length_segment, width_segment))

    horizontal = gf.c.array(
        component=segment_horizontal,
        columns=int(dx / pitch),
        spacing=(pitch, 0),
    )

    if with_north:
        top = c << horizontal
        top.ymax = tl.ymax
        top.xmin = tl.xmax + spacing_segment

        # horizontal inner
        topi = c << horizontal
        topi.ymax = top.ymin - spacing_segment
        topi.xmin = top.xmin + pitch / 2

    if with_south:
        bot = c << horizontal
        bot.ymin = ymin
        bot.xmin = tl.xmax + spacing_segment

        boti = c << horizontal
        boti.ymin = bot.ymax + spacing_segment
        boti.xmin = bot.xmin + spacing_segment

    # vertical
    segment_vertical = via_stack(size=(width_segment, length_segment))
    dy = abs(tl.ymin - bl.ymax)

    vertical = gf.c.array(
        component=segment_vertical,
        rows=int(dy / pitch),
        columns=1,
        spacing=(0, pitch),
    )

    if with_east:
        right = c << vertical
        right.xmax = xmax
        right.ymin = bl.ymax
        righti = c << vertical
        righti.xmax = right.xmin - spacing_segment
        righti.ymin = right.ymin + pitch / 2

    if with_west:
        left = c << vertical
        left.xmin = xmin
        left.ymin = bl.ymax

        # vertical inner
        lefti = c << vertical
        lefti.xmin = left.xmax + spacing_segment
        lefti.ymin = left.ymin + pitch / 2

    return c


if __name__ == "__main__":
    c = gf.Component()
    ref = c << gf.c.rectangle(size=(500, 100), layer=(1, 0))
    ref.move((500, 300))
    c << seal_ring_segmented(c.bbox, with_south=False, padding=50, spacing_segment=5)
    # big_square = partial(rectangle, size=(1300, 2600))
    # c = gf.Component("demo")
    # c << big_square()
    # c << seal_ring(c.bbox + ((0, 0), (10, 0)), with_south=False)
    c.show(show_ports=True)
