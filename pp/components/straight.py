"""Straight waveguide."""

import pp
from pp.add_padding import get_padding_points
from pp.component import Component
from pp.cross_section import cross_section, get_waveguide_settings
from pp.path import extrude
from pp.path import straight as straight_path
from pp.snap import snap_to_grid


@pp.cell_with_validator
def straight(
    length: float = 10.0,
    npoints: int = 2,
    waveguide: str = "strip",
    with_cladding_box: bool = True,
    **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: of straight
        npoints: number of points
        waveguide: from TECH.waveguide
        with_cladding_box: to remove DRC
        kwargs: waveguide_settings

    """
    p = straight_path(length=length, npoints=npoints)
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    c = extrude(p, x)
    c.length = snap_to_grid(length)
    c.width = x.info["width"]
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

    # c = straight(width=2.0)
    c = straight(waveguide="strip_heater")
    print(c.name)
    c.pprint()
    # print(c.get_settings()['settings']['waveguide_settings']['layers_cladding'])

    # print(c.name)
    # print(c.length)
    # print(c.ports)
    c.show(show_ports=True)
    # c.plot()
