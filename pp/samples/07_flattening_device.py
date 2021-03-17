""" From phidl tutorial

# Flattening a Component

Sometimes you want to remove cell structure from a Component while keeping all
of the shapes/polygons intact and in place.  The c.flatten() keeps all the
polygons in c, but removes all the underlying references it's attached to.
Also, if you specify the `single_layer` argument it will move all of the
polyons to that single layer.

"""
import pp
from pp.component import Component


def test_flatten_device() -> Component:

    c = pp.Component("test_remap_layers")

    c.add_ref(pp.components.rectangle(size=(10, 1), layer=pp.LAYER.WG))
    c.add_ref(pp.components.rectangle(size=(10, 2), layer=pp.LAYER.SLAB90))
    c.add_ref(pp.components.rectangle(size=(10, 3), layer=pp.LAYER.SLAB150))

    assert len(c.references) == 3
    c.flatten()
    assert len(c.references) == 0
    return c


if __name__ == "__main__":
    c = test_flatten_device()
    c.show()
