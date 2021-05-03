"""Straight waveguide."""

from pp.add_padding import get_padding_points
from pp.cell import cell
from pp.component import Component
from pp.cross_section import cross_section
from pp.path import extrude
from pp.path import straight as straight_path
from pp.snap import snap_to_grid
from pp.tech import TECH


@cell
def straight(length: float = 10.0, npoints: int = 2, **settings) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        settings: cross_section settings to extrude paths

    """
    p = straight_path(length=length, npoints=npoints)
    x = cross_section(**settings)
    c = extrude(p, x)
    c.length = snap_to_grid(length)
    c.width = x.info["width"]

    if x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offsetcomponent"]

        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            top=cladding_offset,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    # c = straight(length=10.0)
    # c.pprint()

    # c = straight(
    #     length=10.001,
    #     width=0.5,
    #     cross_section={"clad": dict(width=3, offset=0, layer=(111, 0))},
    # )

    # c = straight(settings=TECH.waveguide.metal_routing)
    # c = straight(settings=TECH.waveguide.rib_slab90)

    settings = TECH.waveguide.strip
    settings.update(width=2)
    c = straight(**settings)

    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
