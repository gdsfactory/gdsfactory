"""From phidl tutorial.

# Flattening a Component

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.

The Component.flatten() method returns a new flatten Component with all the
polygons inside the Component, and removes all the underlying references.

"""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def flatten_device() -> Component:
    c = gf.Component("test_remap_layers")

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=(1, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=(3, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=(2, 0)))

    assert len(c.references) == 3
    c2 = c.flatten()
    assert len(c2.references) == 0
    return c2


if __name__ == "__main__":
    c = flatten_device()
    c.show(show_ports=True)
