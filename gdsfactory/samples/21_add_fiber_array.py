"""Connecting a component with I/O.
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.samples.big_device import big_device


def test_big_device() -> Component:
    component = big_device(N=10)
    radius = 5.0
    c = gf.routing.add_fiber_array(
        component=component, radius=radius, fanout_length=50.0
    )
    return c


if __name__ == "__main__":
    c = test_big_device()

    c.show()
