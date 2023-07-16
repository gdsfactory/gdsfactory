"""You can remap layers."""
from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def remap_layers() -> Component:
    c = gf.Component()
    straight = partial(
        gf.components.straight,
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )

    wg1 = c << straight(length=11, width=1, layer=(1, 0))
    wg2 = c << straight(length=11, width=2, layer=(2, 0))
    wg3 = c << straight(length=11, width=3, layer=(3, 0))

    wg2.connect(port="o1", destination=wg1.ports["o2"])
    wg3.connect(port="o1", destination=wg2.ports["o2"], overlap=1)

    nlayers = len(c.layers)
    assert len(c.layers) == nlayers
    c.remap_layers({(1, 0): (2, 0)})
    assert len(c.layers) == nlayers - 1
    return c


if __name__ == "__main__":
    c = remap_layers()
    c.show(show_ports=True)
