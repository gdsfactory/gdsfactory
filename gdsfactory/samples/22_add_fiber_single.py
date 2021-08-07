"""This example shows how to add_grating couplers for single fiber in single fiber out (no fiber array).
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.samples.big_device import big_device


def test_fiber_single() -> Component:
    w = h = 18 * 50
    c = big_device(port_pitch=50.0, h=h, w=w)
    return gf.routing.add_fiber_single(component=c)


def demo_needs_fix():
    """FIXME"""
    c = big_device()
    return gf.routing.add_fiber_single(component=c)


if __name__ == "__main__":
    c = test_fiber_single()
    # c = demo_needs_fix()
    c.show()
