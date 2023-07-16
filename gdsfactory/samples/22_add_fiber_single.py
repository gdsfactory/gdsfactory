"""You can also connect a component with single fiber INPUT and OUTPUTS (no fiber array)."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.samples.big_device import big_device


@gf.cell
def big_device_fiber_single() -> gf.Component:
    w = h = 18 * 50
    c = big_device(spacing=50.0, size=(w, h))
    return gf.routing.add_fiber_single(component=c)


def test_big_device_fiber_single() -> None:
    big_device_fiber_single()


if __name__ == "__main__":
    c = big_device_fiber_single()
    c.show(show_ports=True)
