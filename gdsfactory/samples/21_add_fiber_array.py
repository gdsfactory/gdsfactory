"""You can route all component optical ports to a fiber array."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.samples.big_device import big_device


@gf.cell
def big_device_with_gratings() -> gf.Component:
    component = big_device(nports=10)
    radius = 5.0
    return gf.routing.add_fiber_array(
        component=component, radius=radius, fanout_length=50.0
    )


def test_big_device() -> None:
    assert big_device_with_gratings()


if __name__ == "__main__":
    c = big_device_with_gratings()
    c.show(show_ports=True)
