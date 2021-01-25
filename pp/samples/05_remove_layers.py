"""You can remove a list of layers."""


import pp
from pp.component import Component


def test_remove_layers() -> Component:
    c = pp.Component("test_remove_layers")

    wg1 = c << pp.c.waveguide(length=10, width=1, layer=pp.LAYER.WG)
    wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)
    wg3 = c << pp.c.waveguide(length=10, width=3, layer=pp.LAYER.SLAB150)

    wg2.connect(port="W0", destination=wg1.ports["E0"])
    wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)

    assert len(c.layers) == 3

    c.remove_layers(layers=[pp.LAYER.SLAB90, pp.LAYER.SLAB150])

    assert len(c.layers) == 1
    assert pp.LAYER.WG in c.layers
    return c


if __name__ == "__main__":
    c = test_remove_layers()
    c.show()
