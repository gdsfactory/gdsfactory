"""You can remove a list of layers."""


import gdsfactory
from gdsfactory.component import Component


def test_remove_layers() -> Component:
    c = gdsfactory.Component("test_remove_layers")

    c.add_ref(gdsfactory.components.rectangle(size=(10, 1), layer=gdsfactory.LAYER.WG))
    c.add_ref(
        gdsfactory.components.rectangle(size=(10, 2), layer=gdsfactory.LAYER.SLAB90)
    )
    c.add_ref(
        gdsfactory.components.rectangle(size=(10, 3), layer=gdsfactory.LAYER.SLAB150)
    )

    assert len(c.layers) == 3

    c.remove_layers(layers=[gdsfactory.LAYER.SLAB90, gdsfactory.LAYER.SLAB150])

    assert len(c.layers) == 1
    return c


if __name__ == "__main__":
    c = test_remove_layers()
    c.show()
