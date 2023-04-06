"""Wires for electrical manhattan routes."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.typings import CrossSectionSpec
import numpy as np

wire_straight = gf.partial(straight, cross_section="metal_routing")


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "metal_routing", **kwargs
) -> Component:
    """Returns 45 degrees electrical corner wire.

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
    c.info["dy"] = width
    return c


@gf.cell
def wire_corner45(
    cross_section: CrossSectionSpec = "metal_routing", radius: float = 10, **kwargs
) -> Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        cross_section: spec.
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer
    width = x.width
    radius = x.radius if radius is None else radius
    if radius is None:
        raise ValueError(
            "Radius needs to be specified in wire_corner45 or in the cross_section."
        )

    c = Component()

    a = width / 2

    xpts = [0, radius + a, radius + a, -np.sqrt(2) * width]
    ypts = [-a, radius, radius + np.sqrt(2) * width, -a]

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(
        name="e1",
        center=(0, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )
    c.add_port(
        name="e2",
        center=(radius, radius),
        width=width,
        orientation=90,
        layer=layer,
        port_type="electrical",
    )
    c.info["length"] = np.sqrt(2) * radius
    return c


if __name__ == "__main__":
    # c = wire_straight()
    c = wire_corner()
    # c.show(show_ports=True)
    # c.pprint_ports()
    c.pprint()

    # print(yaml.dump(c.to_dict()))
