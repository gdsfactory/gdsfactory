from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.via import via1, via2, via3
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentOrFactory, Layer

orientation_to_anchor = {
    0: "ce",
    90: "nc",
    180: "cw",
    270: "sc",
}

valid_anchors = [
    "ce",
    "cw",
    "nc",
    "ne",
    "nw",
    "sc",
    "se",
    "sw",
    "center",
    "cc",
]


@gf.cell
def via_stack(
    size: Tuple[float, float] = (11.0, 11.0),
    layers: Tuple[Layer, ...] = (LAYER.M1, LAYER.M2, LAYER.M3),
    vias: Optional[Tuple[Optional[ComponentOrFactory], ...]] = (via2, via3),
    layer: Layer = LAYER.M3,
    port_orientation: int = 90,
    port_location: Optional[str] = None,
) -> Component:
    """Rectangular via_stack

    Args:
        size: (tuple) Width and height of rectangle.
        layers: layers on which to draw rectangles
        vias: vias to use to fill the rectangles
        layer: port layer
        port_orientation: 180: West, 0: East, 90: N0, 270: S0
        port_location: cc: center
    """
    port_location = port_location or orientation_to_anchor[port_orientation]

    if port_location not in valid_anchors:
        raise ValueError(f"port_location = {port_location} not in {valid_anchors}")

    width, height = size
    a = width / 2
    b = height / 2
    rect_pts = [(-a, -b), (a, -b), (a, b), (-a, b)]

    c = Component()
    c.height = height

    # Add metal rectangles
    for layer in layers:
        c.add_polygon(rect_pts, layer=layer)

    vias = vias or []
    # Add vias
    for via in vias:
        if via is not None:
            via = via() if callable(via) else via

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

            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2

            for i in range(nb_vias_x):
                for j in range(nb_vias_y):
                    c.add(via.ref(position=(x0 + i * pitch_x, y0 + j * pitch_y)))

    c.add_port(
        name="e1",
        width=width if port_orientation in [90, 270] else height,
        orientation=port_orientation,
        layer=layer,
        port_type="electrical",
        midpoint=getattr(c.size_info, port_location),
    )
    return c


via_stack_metal = gf.partial(
    via_stack,
    layers=(LAYER.M1, LAYER.M2, LAYER.M3),
    vias=(via2, via3),
)

via_stack_slab = gf.partial(
    via_stack,
    layers=(LAYER.SLAB90, LAYER.M1, LAYER.M2, LAYER.M3),
    vias=(via1, via2, via3),
)
via_stack_slab_npp = gf.partial(
    via_stack,
    layers=(LAYER.SLAB90, LAYER.Npp, LAYER.M1, LAYER.M2, LAYER.M3),
    vias=(via1, via2, via3),
)
via_stack_heater = gf.partial(
    via_stack, layers=(LAYER.HEATER, LAYER.M2, LAYER.M3), vias=(via2, via3)
)


if __name__ == "__main__":
    c = via_stack(port_location="sc")
    print(c)
    c.show()
