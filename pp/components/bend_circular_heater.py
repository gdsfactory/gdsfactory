from typing import Optional

import pp
from pp.cell import cell
from pp.component import Component
from pp.path import arc, component
from pp.snap import snap_to_grid
from pp.tech import TECH_SILICON_C, Tech


@cell
def bend_circular_heater(
    radius: float = 10,
    angle: int = 90,
    npoints: int = 720,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
    width: Optional[float] = None,
    tech: Tech = TECH_SILICON_C,
) -> Component:
    """Creates an arc of arclength ``theta`` starting at angle ``start_angle``

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        heater_to_wg_distance:
        heater_width
        width: straight width (defaults to tech.wg_width)
        tech: Technology
    """
    width = width or tech.wg_width

    x = pp.CrossSection()
    x.add(width=width, offset=0, layer=tech.layer_wg, ports=["in", "out"])

    for layer_cladding in tech.layers_cladding:
        x.add(width=width + 2 * tech.cladding_offset, offset=0, layer=layer_cladding)

    offset = heater_to_wg_distance + width / 2
    x.add(
        width=width,
        offset=+offset,
        layer=tech.layer_heater,
        ports=["top_in", "top_out"],
    )
    x.add(
        width=width,
        offset=-offset,
        layer=tech.layer_heater,
        ports=["bot_in", "bot_out"],
    )
    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = component(p, x)
    c.length = snap_to_grid(p.length())
    c.dx = abs(p.points[0][0] - p.points[-1][0])
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    return c


if __name__ == "__main__":
    c = bend_circular_heater()
    print(c.ports)
    c.show()
