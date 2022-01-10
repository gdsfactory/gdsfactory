from typing import Optional

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, viac
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Float2, Floats, Layer, Layers


@gf.cell
def contact_slot(
    size: Float2 = (11.0, 11.0),
    layers: Layers = (LAYER.M1, LAYER.M2),
    layer_offsets: Optional[Floats] = (0, 1.0),
    layer_offsetsx: Optional[Floats] = None,
    layer_offsetsy: Optional[Floats] = None,
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
        layer_offsetsx: optional xoffset for layers, defaults to layer_offsets
        layer_offsetsx: optional yoffset for layers, defaults to layer_offsets
        layer_port: if None asumes port is on the last layer
        via: via to use to fill the rectangles
        enclosure: of the via by rectangle
        ysize: via height in y
        yspacing: via spacing pitch in y

    .. code::


         __________________________________________
        |                |                        |
        |                | layer_offsetsy[1]      |
        |  ______________|______________________  |
        |  |<--->                              |<>|
        |  |enclosure                          | layer_offsetsx[1]
        |  |      ______________________       |  |
        |  |     |                      |      |  |
        |  |     |     via              | ysize|  |
        |  |     |______________________|      |  |
        |  |  |                                |  |
        |  |  | yspacing                size[1]|  |
        |  |  |                                |  |
        |  |  |   ______________________       |  |
        |  |  |  |                      |      |  |
        |  |  |  |     via              | ysize|  |
        |  |  |  |______________________|      |  |
        |  |                                   |  |
        |  |                                   |  |
        |  |___________________________________|  |
        |                  size[0]                |
        |                                         |
        |_________________________________________|

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

    layer_offsetsx = layer_offsetsx or layer_offsets
    layer_offsetsy = layer_offsetsy or layer_offsets

    layer_offsetsx = list(layer_offsetsx) + [0] * len(layers)
    layer_offsetsy = list(layer_offsetsy) + [0] * len(layers)

    for layer, offsetx, offsety in zip(layers, layer_offsetsx, layer_offsetsy):
        ref = c << compass(
            size=(size[0] + 2 * offsetx, size[1] + 2 * offsety), layer=layer
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
