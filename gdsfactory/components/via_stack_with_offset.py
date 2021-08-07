from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.via import via1
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Layer


@gf.cell
def via_stack_with_offset(
    layer_via_width_height_offset: Tuple[
        Tuple[Layer, Optional[ComponentFactory], float, float, float], ...
    ] = (
        (LAYER.Ppp, None, 10.0, 10.0, 0.0),
        (LAYER.M1, via1, 10.0, 10.0, 0.0),
    ),
    port_orientation: int = 180,
) -> Component:
    """Rectangular transition thru metal layers with offset between layers

    Args:
        layer_via_width_height_offset:
            layer:
            via:
            width: via_stack width
            height: height:
            offset: for next layer
        port_orientation: 180: W0, 0: E0, 90: N0, 270: S0
    """
    c = Component()
    x0 = x1 = y0 = y1 = 0

    i = 0
    for i, (layer, via, width, height, offset) in enumerate(
        layer_via_width_height_offset
    ):
        x0 = -width / 2
        x1 = +width / 2
        y1 = y0 + height
        rect_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        c.add_polygon(rect_pts, layer=layer)

        if via:
            via = via()
            w = via.info["width"]
            h = via.info["height"]
            g = via.info["enclosure"]
            pitch_x = via.info["pitch_x"]
            pitch_y = via.info["pitch_y"]

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(floor(nb_vias_x)) or 1
            nb_vias_y = int(floor(nb_vias_y)) or 1

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2

            x00 = x0 + cw + w / 2
            y00 = y0 + ch + h / 2 + offset

            for i in range(nb_vias_x):
                for j in range(nb_vias_y):
                    c.add(via.ref(position=(x00 + i * pitch_x, y00 + j * pitch_y)))
        y0 += offset
        y1 = y0 + height
        rect_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        c.add_polygon(rect_pts, layer=layer)

    if port_orientation == 0:
        port_name = "E0"
        port_width = height
    elif port_orientation == 180:
        port_name = "W0"
        port_width = height
    elif port_orientation == 90:
        port_name = "N0"
        port_width = width
    elif port_orientation == 270:
        port_name = "S0"
        port_width = width
    else:
        raise ValueError(
            f"Invalid port_orientation = {port_orientation} not in [0, 90, 180, 270]"
        )
    c.add_port(
        name=port_name,
        width=port_width,
        orientation=port_orientation,
        midpoint=(0, y1),
        port_type="dc",
    )

    return c


if __name__ == "__main__":

    c = via_stack_with_offset(
        layer_via_width_height_offset=(
            (LAYER.Ppp, gf.components.via1, 10, 10, 0),
            (LAYER.M1, None, 10, 10, 10),
        )
    )
    c = via_stack_with_offset(
        layer_via_width_height_offset=(
            (LAYER.Ppp, gf.components.via1, 10, 10, 10),
            (LAYER.M1, gf.components.via2, 10, 10, 0),
            (LAYER.M2, None, 10, 10, 10),
        )
    )
    c = via_stack_with_offset(
        layer_via_width_height_offset=(
            (LAYER.Ppp, gf.components.via1, 5, 10, 0),
            (LAYER.M1, gf.components.via2, 5, 10, 10),
            (LAYER.M2, gf.components.via3, 5, 10, 0),
            # (LAYER.M3, None, 5, 10, 0),
        )
    )
    # c.pprint()
    c = via_stack_with_offset()
    print(c)
    c.show()
