"""You can remove a list of layers from a component."""


from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def test_remove_layers() -> Component:
    c = gf.Component("test_remove_layers")

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=gf.LAYER.WG))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=gf.LAYER.SLAB90))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=gf.LAYER.SLAB150))

    assert len(c.layers) == 3

    c = c.remove_layers(layers=[gf.LAYER.SLAB90, gf.LAYER.SLAB150])

    assert len(c.layers) == 1
    return c


if __name__ == "__main__":
    c = test_remove_layers()
    c.show(show_ports=True)
