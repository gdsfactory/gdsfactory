"""wires for electrical manhattan routes
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import metal3
from gdsfactory.types import CrossSectionFactory

wire_straight = gf.partial(straight, with_cladding_box=False, cross_section=metal3)


@gf.cell
def wire_corner(cross_section: CrossSectionFactory = metal3, **kwargs) -> Component:
    """90 degrees electrical corner

    Args:
        waveguide:
        kwargs: cross_section settings

    """
    x = cross_section(**kwargs)
    layer = x.info["layer"]
    width = x.info["width"]

    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(
        name="e1",
        midpoint=(-a, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )
    c.add_port(
        name="e2",
        midpoint=(0, a),
        width=width,
        orientation=90,
        layer=layer,
        port_type="electrical",
    )
    c.info.length = width
    return c


if __name__ == "__main__":

    # c = wire_straight()
    c = wire_corner()
    c.show(show_ports=True)
    c.pprint_ports()
