"""Straight waveguide."""

from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import StrOrDict, get_cross_section
from gdsfactory.path import extrude
from gdsfactory.path import straight as straight_path
from gdsfactory.snap import snap_to_grid


@cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    waveguide: StrOrDict = "strip",
    with_cladding_box: bool = True,
    **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        waveguide: from TECH.waveguide
        with_cladding_box: square in layers_cladding to remove DRC
    """
    p = straight_path(length=length, npoints=npoints)
    x = get_cross_section(waveguide, **kwargs)
    c = extrude(p, x)
    c.length = snap_to_grid(length)
    c.width = x.info["width"]
    c.waveguide_settings = x.info
    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
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
    from gdsfactory.tech import Section

    # c = straight(width=2.0)
    # c = straight(waveguide="metal_routing")
    # c = straight(waveguide="strip_heater_single", width=3)

    wg = dict(width=0.2, sections=(Section(width=3, layer=(2, 0)),))

    c = straight(waveguide=wg, width=2.2)

    # print(c.name)
    # c.pprint()

    # print(c.ports)
    # print(c.get_settings()['settings']['waveguide_settings']['layers_cladding'])

    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
