from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.via import viac
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Layers


@gf.cell
def via_stack_with_offset(
    layers: Layers = (LAYER.PPP, LAYER.M1),
    sizes: Tuple[Tuple[float, float], ...] = ((10, 10), (10, 10)),
    vias: Tuple[Optional[ComponentOrFactory], ...] = (None, viac),
    offsets: Optional[Tuple[float, ...]] = None,
    port_orientation: int = 180,
) -> Component:
    """Rectangular transition thru metal layers with offset between layers

    Args:
        layers:
        vias: factory for via or None for no via
        sizes:
        offsets: for next layer
        port_orientation: 180: W0, 0: E0, 90: N0, 270: S0
    """
    c = Component()
    x0 = x1 = y0 = y1 = 0

    offsets = offsets or [0] * len(layers)

    for layer, via, size, offset in zip(layers, vias, sizes, offsets):
        width, height = size
        x0 = -width / 2
        x1 = +width / 2
        y1 = y0 + height
        rect_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        c.add_polygon(rect_pts, layer=layer)

        if via:
            via = via() if callable(via) else via
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

        y0 += offset
        y1 = y0 + height
        rect_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        c.add_polygon(rect_pts, layer=layer)

    port_width = height if port_orientation in [0, 180] else width

    if port_orientation not in [0, 90, 270, 180]:
        raise ValueError(
            f"Invalid port_orientation = {port_orientation} not in [0, 90, 180, 270]"
        )
    c.add_port(
        name="e1",
        width=port_width,
        orientation=port_orientation,
        midpoint=(0, y1),
        port_type="electrical",
    )
    return c


if __name__ == "__main__":
    c = via_stack_with_offset(
        layers=(LAYER.SLAB90, LAYER.M1),
        sizes=((20, 10), (20, 10)),
        vias=(viac(size=(18, 2), spacing=(5, 5)), None),
    )
    c.show()
