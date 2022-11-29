"""You can route all component optical ports to a fiber array."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.samples.big_device import big_device


def test_big_device() -> Component:
    component = big_device(nports=10)
    radius = 5.0
    return gf.routing.add_fiber_array(
        component=component, radius=radius, fanout_length=50.0
    )


if __name__ == "__main__":
    c = test_big_device()
    c.show(show_ports=True)
