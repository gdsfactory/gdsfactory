from __future__ import annotations

import gdsfactory as gf
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import ComponentSpec, Float2


@gf.cell_with_module_name
def seal_ring(
    size: Float2 = (500, 500),
    seal: ComponentSpec = "via_stack",
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
        size: of the seal.
        seal: function for the seal.
        width: of the seal.
        padding: from component to seal.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    c = gf.Component()

    xmin, ymin = 0, 0
    xmax = size[0]
    ymax = size[1]
    x = (xmax + xmin) / 2
    sx = xmax - xmin
    sy = ymax - ymin

    sx = snap_to_grid(sx, grid_factor=2)
    sy = snap_to_grid(sy, grid_factor=2)

    ymin_north = snap_to_grid(ymax + padding, grid_factor=2)
    ymax_south = snap_to_grid(ymax - sy - padding, grid_factor=2)

    # north south
    size_north_south = (sx + 2 * padding + 2 * width, width)
    size_east_west = (width, sy + 2 * padding)

    if with_north:
        north = c << gf.get_component(
            seal, size=size_north_south, port_orientations=None
        )
        north.ymin = ymin_north
        north.x = x

    if with_east:
        east = c << gf.get_component(seal, size=size_east_west, port_orientations=None)
        east.xmin = xmax + padding
        east.ymax = ymin_north

    if with_west:
        west = c << gf.get_component(seal, size=size_east_west, port_orientations=None)
        west.xmax = xmin - padding
        west.ymax = ymin_north

    if with_south:
        south = c << gf.get_component(
            seal, size=size_north_south, port_orientations=None
        )
        south.ymax = ymax_south
        south.x = x

    return c


@gf.cell_with_module_name
def seal_ring_segmented(
    size: Float2 = (500, 500),
    length_segment: float = 10,
    width_segment: float = 3,
    spacing_segment: float = 2,
    corner: ComponentSpec = "via_stack_corner45_extended",
    via_stack: ComponentSpec = "via_stack_m1_mtop",
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
) -> gf.Component:
    """Segmented Seal ring.

    Args:
        size: of the seal ring.
        length_segment: length of each segment.
        width_segment: width of each segment.
        spacing_segment: spacing between segments.
        corner: corner component.
        via_stack: via_stack component.
        with_north: includes seal.
        with_south: includes seal.
        with_east: includes seal.
        with_west: includes seal.
    """
    c = gf.Component()
    corner_component = gf.get_component(corner, width=width_segment)

    xmin, ymin = 0, 0
    xmax = size[0]
    ymax = size[1]

    tl = c << corner_component
    tr = c << corner_component

    tl.xmin = xmin
    tl.ymax = ymax

    tr.dmirror()
    tr.xmax = xmax
    tr.ymax = ymax

    bl = c << corner_component
    br = c << corner_component
    br.dmirror()
    br.dmirror_y()
    bl.dmirror_y()

    bl.xmin = xmin
    bl.ymin = ymin
    br.xmax = xmax
    br.ymin = ymin

    pitch = length_segment + spacing_segment

    # horizontal
    dx = abs(tl.xmax - tr.xmin)
    segment_horizontal = gf.get_component(
        via_stack, size=(length_segment, width_segment), port_orientations=None
    )
    horizontal = gf.c.array(
        component=segment_horizontal, columns=int(dx / pitch), column_pitch=pitch
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
    segment_vertical = gf.get_component(
        via_stack, size=(width_segment, length_segment), port_orientations=None
    )
    dy = abs(tl.ymin - bl.ymax)

    vertical = gf.c.array(
        component=segment_vertical, rows=int(dy / pitch), columns=1, row_pitch=pitch
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
    c = seal_ring_segmented()
    c.show()
