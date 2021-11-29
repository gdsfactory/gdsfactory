"""You can remap layers."""
import gdsfactory as gf
from gdsfactory.component import Component


def test_remap_layers() -> Component:
    c = gf.Component("test_remap_layers_sample_device")

    wg1 = c << gf.components.straight(length=11, width=1, layer=gf.LAYER.WG)
    wg2 = c << gf.components.straight(length=11, width=2, layer=gf.LAYER.SLAB90)
    wg3 = c << gf.components.straight(length=11, width=3, layer=gf.LAYER.SLAB150)

    wg2.connect(port="o1", destination=wg1.ports["o2"])
    wg3.connect(port="o1", destination=wg2.ports["o2"], overlap=1)

    nlayers = len(c.layers)
    assert len(c.layers) == nlayers
    c.remap_layers({gf.LAYER.WG: gf.LAYER.SLAB150})
    assert len(c.layers) == nlayers - 1
    return c


if __name__ == "__main__":
    c = test_remap_layers()
    c.show()
