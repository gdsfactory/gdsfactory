"""You can remap layers."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def remap_layers() -> Component:
    c = gf.Component()
    straight = gf.components.straight(length=11, width=1).dup()

    wg1 = c << straight
    wg2 = c << straight
    wg3 = c << straight

    wg2.connect("o1", wg1.ports["o2"])
    wg3.connect("o1", wg2.ports["o2"])

    c = c.remap_layers({(1, 0): (2, 0)}, recursive=True)
    return c


def test_remap_layers() -> None:
    c = remap_layers()
    assert c.layers == [(2, 0)]


if __name__ == "__main__":
    c = remap_layers()
    c.show()
