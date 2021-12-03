from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.contact import contact_slab_npp_m3
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Floats, Layers, Optional

pad_contact_slab_npp = partial(contact_slab_npp_m3, size=(80, 80))


@cell
def resistance_sheet(
    width: float = 10,
    layers: Layers = (LAYER.SLAB90, LAYER.NPP),
    layer_offsets: Floats = (0, 0.2),
    pad: ComponentFactory = pad_contact_slab_npp,
    pad_pitch: float = 100.0,
    ohms_per_square: Optional[float] = None,
) -> Component:
    """Sheet resistance.
    keeps connectivity for pads and first layer in layers

    Args:
        width:
        layers: for the middle part
        layer_offsets: from edge, positive: over, negative: inclusion
        pad: function to create a pad
        pad_pitch:
        ohms_per_square: optional sheet resistance to compute info.resistance
    """
    c = Component()

    pad = pad()
    length = pad_pitch - pad.info_child.size[0]

    pad1 = c << pad
    pad2 = c << pad
    r0 = c << compass(
        size=(length + layer_offsets[0], width + layer_offsets[0]), layer=layers[0]
    )

    for layer, offset in zip(layers[1:], layer_offsets[1:]):
        c << compass(size=(length + offset, width + offset), layer=layer)

    pad1.connect("e3", r0.ports["e1"])
    pad2.connect("e1", r0.ports["e3"])

    c.info_child.resistance = (
        ohms_per_square * width * length if ohms_per_square else None
    )

    c.add_port("pad1", port_type="vertical_dc", midpoint=pad1.center)
    c.add_port("pad2", port_type="vertical_dc", midpoint=pad2.center)
    return c


if __name__ == "__main__":
    c = resistance_sheet(width=40)
    c.show()
