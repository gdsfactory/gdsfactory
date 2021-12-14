from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, viac
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Layer


@gf.cell
def contact_slot(
    size: Tuple[float, float] = (11.0, 11.0),
    layers: Tuple[Layer, ...] = (LAYER.M1, LAYER.M2),
    layer_offsets: Tuple[float, ...] = (0, 1.0),
    layer_port: Optional[Layer] = None,
    via: ComponentOrFactory = via1,
    enclosure: float = 1.0,
    ysize: float = 0.5,
    yspacing: float = 2.0,
) -> Component:
    """Rectangular contact with slotted via in X direction

    Args:
        size: of the layers
        layers: layers on which to draw rectangles
        layer_offsets: cladding_offset for each layer
        layer_port: if None asumes port is on the last layer
        via: via to use to fill the rectangles
        enclosure: of the via by rectangle
        ysize: via height in y
        yspacing: via spacing pitch in y

    .. code::

        enclosure
        _____________________________________
        |<--->                              |
        |      ______________________       |
        |     |                      |      |
        |     |                      | ysize|
        |     |______________________|      |
        |  |                                |
        |  | yspacing                       |
        |  |                                |  size[1]
        |  |   ______________________       |
        |  |  |                      |      |
        |  |  |                      | ysize|
        |  |  |______________________|      |
        |                                   |
        |___________________________________|
                        size[0]


    """
    if size[0] - 2 * enclosure < 0:
        raise ValueError(
            f"Contact length (size[0] = {size[0]}) < 2*enclosure ({2*enclosure}). "
        )
    if size[1] - 2 * enclosure < 0:
        raise ValueError(
            f"Contact width (size[1] = {size[1]}) < 2*enclosure ({2*enclosure}). "
        )

    layer_port = layer_port or layers[-1]

    c = Component()
    c.info.size = (float(size[0]), float(size[1]))

    for layer, offset in zip(layers, list(layer_offsets) + [0] * len(layers)):
        ref = c << compass(
            size=(size[0] + 2 * offset, size[1] + 2 * offset), layer=layer
        )

        if layer == layer_port:
            c.add_ports(ref.ports)

    via = via(size=(size[0] - 2 * enclosure, ysize)) if callable(via) else via

    nb_vias_y = (size[1] - 2 * enclosure) / yspacing
    nb_vias_y = int(floor(nb_vias_y)) or 1
    ref = c.add_array(via, columns=1, rows=nb_vias_y, spacing=(0, yspacing))
    dy = (size[1] - (nb_vias_y - 1) * yspacing - size[1]) / 2
    ref.move((0, dy))
    return c


contact_slot_m1_m2 = gf.partial(contact_slot, layers=(LAYER.M1, LAYER.M2), via=via1)

contact_slot_slab_m1 = gf.partial(contact_slot, layers=(LAYER.M1,), via=viac)


if __name__ == "__main__":
    c = contact_slot_m1_m2(layer_offsets=(0.5, 1), enclosure=2, size=(1, 1))
    c.show()
