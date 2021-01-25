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

    wg1 = c << pp.c.waveguide(length=10, width=1, layer=pp.LAYER.WG)
    wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)
    wg3 = c << pp.c.waveguide(length=10, width=3, layer=pp.LAYER.SLAB150)

    wg2.connect(port="W0", destination=wg1.ports["E0"])
    wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)

    assert len(c.references) == 3
    c.flatten()
    assert len(c.references) == 0
    return c


if __name__ == "__main__":
    c = test_flatten_device()
    c.show()
