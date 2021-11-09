import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.config import TECH
from gdsfactory.cross_section import strip
from gdsfactory.path import arc, extrude
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import CrossSectionFactory


@gf.cell
def bend_circular_heater(
    radius: float = 10,
    angle: int = 90,
    npoints: int = 720,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
    layer_heater=TECH.layer.HEATER,
    cross_section: CrossSectionFactory = strip,
) -> Component:
    """Creates an arc of arclength ``theta`` starting at angle ``start_angle``

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        heater_to_wg_distance:
        heater_width
        layer_heater
        cross_section:
    """
    x = cross_section()
    width = x.info["width"]
    cladding_offset = x.info["cladding_offset"]
    layers_cladding = x.info["layers_cladding"] or []
    layer = x.info["layer"]

    x = gf.CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=["in", "out"])

    for layer_cladding in layers_cladding:
        x.add(width=width + 2 * cladding_offset, offset=0, layer=layer_cladding)

    offset = heater_to_wg_distance + width / 2
    x.add(
        width=heater_width,
        offset=+offset,
        layer=layer_heater,
    )
    x.add(
        width=heater_width,
        offset=-offset,
        layer=layer_heater,
    )
    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = extrude(p, x)
    c.length = snap_to_grid(p.length())
    c.dx = abs(p.points[0][0] - p.points[-1][0])
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    return c


if __name__ == "__main__":
    c = bend_circular_heater(heater_width=1)
    print(c.ports)
    c.show()
