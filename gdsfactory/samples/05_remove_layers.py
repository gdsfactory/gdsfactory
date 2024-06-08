"""You can remove a list of layers from a component."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def remove_layers() -> Component:
    c = gf.Component()

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=(1, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=(3, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=(2, 0)))

    assert len(c.layers) == 3

    c = c.remove_layers(layers=[(3, 0), (2, 0)])

    assert len(c.layers) == 1
    return c


if __name__ == "__main__":
    c = remove_layers()
    c.show()
