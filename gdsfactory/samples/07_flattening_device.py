""" From phidl tutorial

# Flattening a Component

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.  The c.flatten() keeps all the
polygons in c, but removes all the underlying references it's attached to.
Also, if you specify the `single_layer` argument it will move all of the
polyons to that single layer.

"""
import gdsfactory as gf
from gdsfactory.component import Component


def test_flatten_device() -> Component:

    c = gf.Component("test_remap_layers")

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=gf.LAYER.WG))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=gf.LAYER.SLAB90))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=gf.LAYER.SLAB150))

    assert len(c.references) == 3
    c.flatten()
    assert len(c.references) == 0
    return c


if __name__ == "__main__":
    c = test_flatten_device()
    c.show()
