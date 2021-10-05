from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via_stack import via_stack_slab_npp
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Floats, Layers

pad_via_stack_slab_npp = partial(via_stack_slab_npp, size=(100, 100))


@cell
def resistance_sheet(
    layers: Layers = (LAYER.SLAB90, LAYER.NPP),
    offsets: Floats = (0, 0.2),
    length: float = 5.0,
    width: float = 10,
    pad: ComponentFactory = pad_via_stack_slab_npp,
) -> Component:
    """Sheet resistance.
    Ensures connectivity is kept for pads and the first layer in layers

    Args:
        layers: for the middle part
        offsets:
        length:
        width:
        pad: function to create a pad
    """
    c = Component()

    pad = pad()
    pad1 = c << pad
    pad2 = c << pad
    r0 = c << compass(size=(length + offsets[0], width + offsets[0]), layer=layers[0])

    for layer, offset in zip(layers[1:], offsets[1:]):
        c << compass(size=(length + offset, width + offset), layer=layer)

    pad1.connect("e3", r0.ports["e1"])
    pad2.connect("e1", r0.ports["e3"])
    return c


if __name__ == "__main__":
    c = resistance_sheet()
    c.show()
