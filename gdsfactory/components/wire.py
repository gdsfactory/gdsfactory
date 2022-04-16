"""wires for electrical manhattan routes
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import metal3
from gdsfactory.types import CrossSectionOrFactory

wire_straight = gf.partial(straight, cross_section=metal3)


@gf.cell
def wire_corner(cross_section: CrossSectionOrFactory = metal3, **kwargs) -> Component:
    """90 degrees electrical corner

    Args:
        waveguide:
        kwargs: cross_section settings

    """
    x = cross_section(**kwargs) if callable(cross_section) else cross_section
    layer = x.layer
    width = x.width

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
    c.info["length"] = width
    return c


if __name__ == "__main__":

    # c = wire_straight()
    c = wire_corner()
    # c.show(show_ports=True)
    # c.pprint_ports()
    c.pprint()

    # print(yaml.dump(c.to_dict()))
