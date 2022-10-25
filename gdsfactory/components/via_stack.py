from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via1, via2, viac
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell
def via_stack(
    size: Tuple[float, float] = (11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    vias: Optional[Tuple[Optional[ComponentSpec], ...]] = (via1, via2),
    layer_port: LayerSpec = None,
) -> Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    http://www.vlsi-expert.com/2017/12/vias.html

    spacing = via.info['spacing']
    enclosure = via.info['enclosure']

    Args:
        size: of the layers.
        layers: layers on which to draw rectangles.
        vias: vias to use to fill the rectangles.
        layer_port: if None assumes port is on the last layer.
    """
    width, height = size
    a = width / 2
    b = height / 2

    layers = layers or []

    if layers:
        layer_port = layer_port or layers[-1]

    c = Component()
    c.height = height
    c.info["size"] = (float(size[0]), float(size[1]))
    c.info["layer"] = layer_port

    for layer in layers:
        if layer == layer_port:
            ref = c << compass(
                size=(width, height), layer=layer, port_type="electrical"
            )
            c.add_ports(ref.ports)
        else:
            ref = c << compass(size=(width, height), layer=layer, port_type="placement")

    vias = vias or []
    for via in vias:
        if via is not None:
            via = gf.get_component(via)

            w, h = via.info["size"]
            g = via.info["enclosure"]
            pitch_x, pitch_y = via.info["spacing"]

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(floor(nb_vias_x)) or 1
            nb_vias_y = int(floor(nb_vias_y)) or 1
            ref = c.add_array(
                via, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))

    return c


via_stack_m1_m3 = gf.partial(
    via_stack,
    layers=(LAYER.M1, LAYER.M2, LAYER.M3),
    vias=(via1, via2),
)

via_stack_slab_m3 = gf.partial(
    via_stack,
    layers=(LAYER.SLAB90, LAYER.M1, LAYER.M2, LAYER.M3),
    vias=(viac, via1, via2),
)
via_stack_npp_m1 = gf.partial(
    via_stack,
    layers=(LAYER.WG, LAYER.NPP, LAYER.M1),
    vias=(None, None, viac),
)
via_stack_slab_npp_m3 = gf.partial(
    via_stack,
    layers=(LAYER.SLAB90, LAYER.NPP, LAYER.M1),
    vias=(None, None, viac),
)
via_stack_heater_m3 = gf.partial(
    via_stack, layers=(LAYER.HEATER, LAYER.M2, LAYER.M3), vias=(via1, via2)
)


if __name__ == "__main__":
    c = via_stack_m1_m3()
    c.show(show_ports=True)
