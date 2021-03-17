"""You can remove a list of layers."""


import pp
from pp.component import Component


def test_remove_layers() -> Component:
    c = pp.Component("test_remove_layers")

    c.add_ref(pp.components.rectangle(size=(10, 1), layer=pp.LAYER.WG))
    c.add_ref(pp.components.rectangle(size=(10, 2), layer=pp.LAYER.SLAB90))
    c.add_ref(pp.components.rectangle(size=(10, 3), layer=pp.LAYER.SLAB150))

    assert len(c.layers) == 3

    c.remove_layers(layers=[pp.LAYER.SLAB90, pp.LAYER.SLAB150])

    assert len(c.layers) == 1
    assert pp.LAYER.WG in c.layers
    return c


if __name__ == "__main__":
    c = test_remove_layers()
    c.show()
