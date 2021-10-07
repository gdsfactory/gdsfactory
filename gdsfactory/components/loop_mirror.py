"""Sagnac loop_mirror."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.spiral_external_io import spiral_external_io
from gdsfactory.routing.manhattan import route_manhattan
from gdsfactory.types import ComponentFactory


@gf.cell
def loop_mirror(
    component: ComponentFactory = mmi1x2, bend90: ComponentFactory = bend_euler
) -> Component:
    """Returns Sagnac loop_mirror."""
    c = Component()
    component = gf.call_if_func(component)
    bend90 = gf.call_if_func(bend90)
    cref = c.add_ref(component)
    routes = route_manhattan(
        cref.ports["o3"],
        cref.ports["o2"],
        straight=gf.components.straight,
        bend=bend90,
    )
    c.add(routes.references)
    c.add_port(name="o1", port=cref.ports["o1"])
    c.absorb(cref)
    return c


@gf.cell
def loop_mirror_with_delay(loop_mirror=loop_mirror, spiral=spiral_external_io):
    """
    delay = 13e-12
    # delay = length/speed
    # length=delay*speed
    13e-12*3e8/4.2*1e6

    """
    c = Component()
    lm = c << gf.call_if_func(loop_mirror)
    s = c << spiral_external_io()

    lm.connect("o1", s.ports["o1"])
    return c


if __name__ == "__main__":
    # c = loop_mirror()
    # c = loop_mirror_rotated()
    c = loop_mirror_with_delay()
    c.show()
