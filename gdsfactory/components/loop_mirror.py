"""Sagnac loop_mirror."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import ComponentSpec


@gf.cell
def loop_mirror(
    component: ComponentSpec = "mmi1x2", bend90: ComponentSpec = "bend_euler"
) -> Component:
    """Returns Sagnac loop_mirror.

    Args:
        component: 1x2 splitter.
        bend90: 90 deg bend.

    """
    c = Component()
    component = gf.get_component(component)
    bend90 = gf.get_component(bend90)
    cref = c.add_ref(component)
    route_single(
        c,
        cref.ports["o3"],
        cref.ports["o2"],
        straight=gf.components.straight,
        bend=bend90,
    )
    c.add_port(name="o1", port=cref.ports["o1"])
    return c


if __name__ == "__main__":
    c = loop_mirror()
    c.show()
