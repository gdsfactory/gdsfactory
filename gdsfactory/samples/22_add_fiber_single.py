"""This example shows how to add_grating couplers

for single fiber INPUT single fiber OUTPU (no fiber array).
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.samples.big_device import big_device


def test_fiber_single() -> Component:
    w = h = 18 * 50
    c = big_device(port_pitch=50.0, h=h, w=w)
    return gf.routing.add_fiber_single(component=c, zero_port="W1")


if __name__ == "__main__":
    c = test_fiber_single()
    c.show()
