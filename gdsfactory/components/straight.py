"""Straight waveguide."""
import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import CrossSectionSpec


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = strip,
    **kwargs
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string, CrossSectionFactory, dict).
        kwargs: cross_section settings.

    .. code::

        o1 -------------- o2
                length

    """
    length = snap_to_grid(length)
    p = gf.path.straight(length=length, npoints=npoints)
    x = gf.get_cross_section(cross_section, **kwargs)

    c = Component()
    path = gf.path.extrude(p, x)
    ref = c << path
    c.add_ports(ref.ports)
    c.info["length"] = length
    c.info["width"] = float(x.width)

    if length:
        for layer, offset in zip(x.bbox_layers, x.bbox_offsets):
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            c.add_polygon(points, layer=layer)
    c.absorb(ref)
    return c


if __name__ == "__main__":
    # c = straight(cross_section=gf.partial(gf.cross_section.metal3, width=2))
    # c = straight(cross_section=gf.partial(gf.cross_section.strip, width=2))
    # c = straight(cladding_offset=2.5)
    # c = straight(width=2.5)

    from gdsfactory.cross_section import strip

    strip2 = strip(layer=(2, 0))
    settings = dict(width=2)

    # c = straight(
    #     length=1, cross_section={"cross_section": "strip", "settings": settings}
    # )
    c = straight(
        length=1,
        cross_section={"cross_section": "strip", "settings": settings},
        width=3,
        # bbox_layers=[(2, 0)],
        # bbox_offsets=[3],
    )
    c.assert_ports_on_grid()
    c.show()
    c.pprint()
