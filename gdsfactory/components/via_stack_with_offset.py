from __future__ import annotations

from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.via import viac
from gdsfactory.components.compass import compass
from gdsfactory.typings import ComponentSpec, LayerSpecs, Float2


@gf.cell
def via_stack_with_offset(
    layers: LayerSpecs = ("PPP", "M1"),
    size: Float2 = (10, 10),
    sizes: Optional[Tuple[Float2, ...]] = None,
    vias: Tuple[Optional[ComponentSpec], ...] = (None, viac),
    offsets: Optional[Tuple[float, ...]] = None,
) -> Component:
    """Rectangular layer transition with offset between layers.

    Args:
        layers: layer specs between vias.
        size: for all vias array.
        sizes: Optional size for each via array. Overrides size.
        vias: via spec for previous layer. None for no via.
        offsets: optional offset for each layer relatively to the previous one.
            By default it only offsets by size[1] if there is a via.

    .. code::

        side view

         __________________________
        |                          |
        |        2 x sizes[2]      | layers[2] vias[1]=None
        |__________________________|
        |                          |
        |        2 x sizes[1]      | layers[1]
        |__________________________|
            |     |
            vias[0]
         ___|_____|__
        |            |
        |  sizes[0]  |  layers[0]
        |____________|


    """
    c = Component()
    y0 = 0

    offsets = offsets or [0] * len(layers)
    sizes = sizes or [size] * len(layers)

    for layer, via, size, offset in zip(layers, vias, sizes, offsets):
        width, height = size
        x0 = -width / 2
        ref_layer = c << compass(
            size=(width, 2 * height), layer=layer, port_type="electrical"
        )
        ref_layer.ymin = y0

        if via:
            via = gf.get_component(via)
            w, h = via.info["size"]
            enclosure = via.info["enclosure"]
            pitch_x, pitch_y = via.info["spacing"]

            nb_vias_x = (width - w - 2 * enclosure) / pitch_x + 1
            nb_vias_y = (height - h - 2 * enclosure) / pitch_y + 1

            nb_vias_x = int(floor(nb_vias_x)) or 1
            nb_vias_y = int(floor(nb_vias_y)) or 1

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2

            x00 = x0 + cw + w / 2
            y00 = y0 + ch + h / 2 + offset

            ref = c.add_array(
                via, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )
            ref.move((x00, y00))
            y0 += height

        y0 += offset

    c.add_ports(ref_layer.ports)
    return c


via_stack_with_offset_ppp_m1 = gf.partial(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, viac),
)

via_stack_with_offset_ppp_m1 = gf.partial(
    via_stack_with_offset,
    layers=("PPP", "M1"),
    vias=(None, viac),
)

via_stack_with_offset_m1_m3 = gf.partial(
    via_stack_with_offset,
    layers=("M1", "M2", "M3"),
    vias=(None, "via1", "via2"),
)


if __name__ == "__main__":
    # c = via_stack_with_offset_ppp_m1(
    #     layers=("SLAB90", "M1"),
    #     sizes=((20, 10), (20, 10)),
    #     vias=(viac(size=(18, 2), spacing=(5, 5)), None),
    # )
    c = via_stack_with_offset_m1_m3()
    # c = via_stack_with_offset(vias=(None, None))
    c.show(show_ports=True)
