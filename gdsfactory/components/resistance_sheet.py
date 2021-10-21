from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.contact import contact_slab_npp_m3
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Floats, Layers

pad_contact_slab_npp = partial(contact_slab_npp_m3, size=(100, 100))


@cell
def resistance_sheet(
    width: float = 10,
    length: float = 5.0,
    layers: Layers = (LAYER.SLAB90, LAYER.NPP),
    layer_offsets: Floats = (0, 0.2),
    pad: ComponentFactory = pad_contact_slab_npp,
) -> Component:
    """Sheet resistance.
    Ensures connectivity is kept for pads and the first layer in layers

    Args:
        width:
        length:
        layers: for the middle part
        layer_offsets:
        pad: function to create a pad
    """
    c = Component()

    pad = pad()
    pad1 = c << pad
    pad2 = c << pad
    r0 = c << compass(
        size=(length + layer_offsets[0], width + layer_offsets[0]), layer=layers[0]
    )

    for layer, offset in zip(layers[1:], layer_offsets[1:]):
        c << compass(size=(length + offset, width + offset), layer=layer)

    pad1.connect("e3", r0.ports["e1"])
    pad2.connect("e1", r0.ports["e3"])
    return c


if __name__ == "__main__":
    c = resistance_sheet(length=50)
    c.show()
