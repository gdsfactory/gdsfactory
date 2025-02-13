from __future__ import annotations

import gdsfactory as gf
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import ComponentSpec, Float2


@gf.cell
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
        north.dymin = ymin_north
        north.dx = x

    if with_east:
        east = c << gf.get_component(seal, size=size_east_west, port_orientations=None)
        east.dxmin = xmax + padding
        east.dymax = ymin_north

    if with_west:
        west = c << gf.get_component(seal, size=size_east_west, port_orientations=None)
        west.dxmax = xmin - padding
        west.dymax = ymin_north

    if with_south:
        south = c << gf.get_component(
            seal, size=size_north_south, port_orientations=None
        )
        south.dymax = ymax_south
        south.dx = x

    return c


@gf.cell
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

    tl.dxmin = xmin
    tl.dymax = ymax

    tr.dmirror()
    tr.dxmax = xmax
    tr.dymax = ymax

    bl = c << corner_component
    br = c << corner_component
    br.dmirror()
    br.dmirror_y()
    bl.dmirror_y()

    bl.dxmin = xmin
    bl.dymin = ymin
    br.dxmax = xmax
    br.dymin = ymin

    pitch = length_segment + spacing_segment

    # horizontal
    dx = abs(tl.dxmax - tr.dxmin)
    segment_horizontal = gf.get_component(
        via_stack, size=(length_segment, width_segment), port_orientations=None
    )
    horizontal = gf.c.array(
        component=segment_horizontal, columns=int(dx / pitch), column_pitch=pitch
    )

    if with_north:
        top = c << horizontal
        top.dymax = tl.dymax
        top.dxmin = tl.dxmax + spacing_segment

        # horizontal inner
        topi = c << horizontal
        topi.dymax = top.dymin - spacing_segment
        topi.dxmin = top.dxmin + pitch / 2

    if with_south:
        bot = c << horizontal
        bot.dymin = ymin
        bot.dxmin = tl.dxmax + spacing_segment

        boti = c << horizontal
        boti.dymin = bot.dymax + spacing_segment
        boti.dxmin = bot.dxmin + spacing_segment

    # vertical
    segment_vertical = gf.get_component(
        via_stack, size=(width_segment, length_segment), port_orientations=None
    )
    dy = abs(tl.dymin - bl.dymax)

    vertical = gf.c.array(
        component=segment_vertical, rows=int(dy / pitch), columns=1, row_pitch=pitch
    )

    if with_east:
        right = c << vertical
        right.dxmax = xmax
        right.dymin = bl.dymax
        righti = c << vertical
        righti.dxmax = right.dxmin - spacing_segment
        righti.dymin = right.dymin + pitch / 2

    if with_west:
        left = c << vertical
        left.dxmin = xmin
        left.dymin = bl.dymax

        # vertical inner
        lefti = c << vertical
        lefti.dxmin = left.dxmax + spacing_segment
        lefti.dymin = left.dymin + pitch / 2

    return c


if __name__ == "__main__":
    c = seal_ring_segmented()
    c.show()
