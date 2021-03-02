""" You can remap layers."""
import pp
from pp.component import Component


def test_remap_layers() -> Component:
    c = pp.Component("test_remap_layers_sample")

    c.add_ref(pp.c.rectangle(size=(10, 1), layer=pp.LAYER.WG))
    c.add_ref(pp.c.rectangle(size=(10, 2), layer=pp.LAYER.SLAB90))
    c.add_ref(pp.c.rectangle(size=(10, 3), layer=pp.LAYER.SLAB150))

    assert len(c.layers) == 3
    c.remap_layers({pp.LAYER.WG: pp.LAYER.SLAB150})
    assert len(c.layers) == 2
    return c


if __name__ == "__main__":
    c = test_remap_layers()
    c.show()
