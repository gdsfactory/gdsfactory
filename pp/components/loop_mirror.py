"""Sagnac loop_mirror."""

import pp
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.mmi1x2 import mmi1x2
from pp.components.spiral_external_io import spiral_external_io
from pp.routing.manhattan import route_manhattan
from pp.types import ComponentFactory


@pp.cell
def loop_mirror(
    component: ComponentFactory = mmi1x2, bend90: ComponentFactory = bend_euler
) -> Component:
    """Returns Sagnac loop_mirror."""
    c = Component()
    component = pp.call_if_func(component)
    bend90 = pp.call_if_func(bend90)
    cref = c.add_ref(component)
    routes = route_manhattan(
        cref.ports["E0"],
        cref.ports["E1"],
        straight_factory=pp.components.straight,
        bend_factory=bend90,
    )
    c.add(routes["references"])
    c.add_port(name="W0", port=cref.ports["W0"])
    c.absorb(cref)
    return c


@pp.cell
def loop_mirror_rotated(component=mmi1x2, bend90=bend_euler):
    c = Component()
    component = pp.call_if_func(component)
    mirror = loop_mirror(component=component, bend90=bend90)
    mirror_rotated = mirror.ref(rotation=90)
    c.add(mirror_rotated)
    c.absorb(mirror_rotated)
    c.add_port(name="S0", port=mirror_rotated.ports["W0"])
    return c


@pp.cell
def loop_mirror_with_delay(loop_mirror=loop_mirror, spiral=spiral_external_io):
    """
    delay = 13e-12
    # delay = length/speed
    # length=delay*speed
    13e-12*3e8/4.2*1e6

    """
    c = Component()
    lm = c << pp.call_if_func(loop_mirror)
    s = c << pp.call_if_func(spiral_external_io)

    lm.connect("W0", s.ports["input"])
    return c


if __name__ == "__main__":
    c = loop_mirror()
    # c = loop_mirror_rotated()
    # c = loop_mirror_with_delay()
    c.show()
