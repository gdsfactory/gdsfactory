"""You can remap layers."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def remap_layers() -> Component:
    c = gf.Component()
    straight = gf.components.straight

    wg1 = c << straight(length=11, width=1, layer=(1, 0))
    wg2 = c << straight(length=11, width=1, layer=(1, 0))
    wg3 = c << straight(length=11, width=1, layer=(1, 0))

    wg2.connect(port="o1", other=wg1.ports["o2"])
    wg3.connect(port="o1", other=wg2.ports["o2"])

    c.remap_layers({(1, 0): (2, 0)})
    return c


def test_remap_layers():
    assert remap_layers()


if __name__ == "__main__":
    c = remap_layers()
    c.show()
