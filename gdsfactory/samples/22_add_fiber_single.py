"""You can also connect a component with single fiber INPUT and OUTPUTS (no fiber array)."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.samples.big_device import big_device


def test_fiber_single() -> Component:
    w = h = 18 * 50
    c = big_device(spacing=50.0, size=(w, h))
    return gf.routing.add_fiber_single(component=c)


if __name__ == "__main__":
    c = test_fiber_single()
    c.show(show_ports=True)
