"""Straight waveguide."""
import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.types import CrossSectionOrFactory


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    with_cladding_box: bool = True,
    cross_section: CrossSectionOrFactory = strip,
    **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length
        npoints: number of points
        with_cladding_box: box in layers_cladding to avoid DRC sharp edges
        cross_section:
        **kwargs: cross_section settings
    """
    p = gf.path.straight(length=length, npoints=npoints)
    x = cross_section(**kwargs) if callable(cross_section) else cross_section

    c = Component()
    path = gf.path.extrude(p, x)
    ref = c << path
    c.add_ports(ref.ports)
    c.info.length = gf.snap.snap_to_grid(length)
    c.info.width = float(x.info["width"])
    if length > 0 and with_cladding_box and x.info["layers_cladding"]:
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
    c.absorb(ref)
    return c


if __name__ == "__main__":
    # c = straight(cross_section=gf.partial(gf.cross_section.metal3, width=2))

    # c = straight(cross_section=gf.partial(gf.cross_section.strip, width=2))
    # c = straight(cladding_offset=2.5)
    # c = straight(width=2.5)
    c = straight(length=0)
    c.assert_ports_on_grid()
    c.show()
    c.pprint()
