""" You can remap layers."""
import gdsfactory
from gdsfactory.component import Component


def test_remap_layers() -> Component:
    c = gdsfactory.Component("test_remap_layers_sample_device")

    wg1 = c << gdsfactory.components.straight(
        length=11, width=1, layer=gdsfactory.LAYER.WG
    )
    wg2 = c << gdsfactory.components.straight(
        length=11, width=2, layer=gdsfactory.LAYER.SLAB90
    )
    wg3 = c << gdsfactory.components.straight(
        length=11, width=3, layer=gdsfactory.LAYER.SLAB150
    )

    wg2.connect(port="W0", destination=wg1.ports["E0"])
    wg3.connect(port="W0", destination=wg2.ports["E0"], overlap=1)

    nlayers = len(c.layers)
    assert len(c.layers) == nlayers
    c.remap_layers({gdsfactory.LAYER.WG: gdsfactory.LAYER.SLAB150})
    assert len(c.layers) == nlayers - 1
    return c


if __name__ == "__main__":
    c = test_remap_layers()
    c.show()
