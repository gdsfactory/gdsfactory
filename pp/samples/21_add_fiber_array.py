"""Connecting a component with I/O.
"""

import pp
from pp.component import Component
from pp.samples.big_device import big_device


def test_big_device() -> Component:
    component = big_device(N=10)
    bend_radius = 5.0
    c = pp.routing.add_fiber_array(
        component, bend_radius=bend_radius, fanout_length=50.0
    )
    return c


if __name__ == "__main__":
    c = test_big_device()

    pp.show(c)
