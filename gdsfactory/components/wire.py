"""Wires for electrical manhattan routes."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.types import CrossSectionSpec

wire_straight = gf.partial(straight, cross_section="metal3")


@gf.cell
def wire_corner(cross_section: CrossSectionSpec = "metal3", **kwargs) -> Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        cross_section: spec.
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer
    width = x.width

    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(
        name="e1",
        center=(-a, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )
    c.add_port(
        name="e2",
        center=(0, a),
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
