"""
"""

from typing import Callable

import pp
from pp.component import Component
from pp.components.euler.bend_euler import bend_euler90
from pp.components.mmi1x2 import mmi1x2
from pp.components.spiral_external_io import spiral_external_io
from pp.routing import route_manhattan


@pp.cell
def loop_mirror(
    component: Callable = mmi1x2, bend90: Callable = bend_euler90
) -> Component:
    c = pp.Component()
    component = pp.call_if_func(component)
    bend90 = pp.call_if_func(bend90)
    cref = c.add_ref(component)
    elements = route_manhattan(
        cref.ports["E0"],
        cref.ports["E1"],
        bend90=bend90,
        straight_factory=pp.c.waveguide,
    )
    c.add(elements)
    c.add_port(name="W0", port=cref.ports["W0"])
    c.absorb(cref)
    return c


@pp.cell
def loop_mirror_rotated(component=mmi1x2, bend90=bend_euler90):
    c = pp.Component()
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
    c = pp.Component()
    lm = c << pp.call_if_func(loop_mirror)
    s = c << pp.call_if_func(spiral_external_io)

    lm.connect("W0", s.ports["input"])
    return c


if __name__ == "__main__":
    # c = loop_mirror()
    # c = loop_mirror_rotated()
    c = loop_mirror_with_delay()
    pp.show(c)
