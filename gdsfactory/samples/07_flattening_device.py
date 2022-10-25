"""From phidl tutorial.

# Flattening a Component

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.

The Component.flatten() method returns a new flatten Component with all the
polygons inside the Component, and removes all the underlying references.
Also, if you specify the `single_layer` argument it will move all of the
polygons to that single layer.

"""
import gdsfactory as gf
from gdsfactory.component import Component


def test_flatten_device() -> Component:

    c = gf.Component("test_remap_layers")

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=gf.LAYER.WG))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=gf.LAYER.SLAB90))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=gf.LAYER.SLAB150))

    assert len(c.references) == 3
    c2 = c.flatten()
    # c2 = c.flatten(single_layer=(34, 0))
    assert len(c2.references) == 0
    return c2


if __name__ == "__main__":
    c = test_flatten_device()
    c.show(show_ports=True)
