"""# Flattening a Component.

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.

The Component.flatten() method returns a new flatten Component with all the
polygons inside the Component, and removes all the underlying references.
Also, if you specify the `single_layer` argument it will move all of the
polygons to that single layer.

FIXME! flatten gives a seg fault

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def flatten_device() -> Component:
    c = gf.Component()

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=(1, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=(3, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=(2, 0)))

    assert len(c.insts) == 3
    c.flatten()
    assert len(c.insts) == 0
    return c


if __name__ == "__main__":
    c = flatten_device()
    c.show()
