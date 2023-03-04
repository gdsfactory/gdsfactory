from __future__ import annotations

from typing import Optional, Tuple

from numpy import floor

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.compass import compass
from gdsfactory.components.via import via, via1, via2, viac
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell
def via_stack(
    size: Tuple[float, float] = (11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "M3"),
    layer_offsets: Optional[Tuple[float, ...]] = None,
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
        layer_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
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

    layer_offsets = layer_offsets or [0] * len(layers)

    for layer, offset in zip(layers, layer_offsets):
        size = (width + 2 * offset, height + 2 * offset)
        if layer == layer_port:
            ref = c << compass(size=size, layer=layer, port_type="electrical")
            c.add_ports(ref.ports)
        else:
            ref = c << compass(size=size, layer=layer, port_type="placement")

    vias = vias or []
    for via_type in vias:
        if via_type is not None:
            via_type = gf.get_component(via_type)

            w, h = via_type.info["size"]
            g = via_type.info["enclosure"]
            pitch_x, pitch_y = via_type.info["spacing"]

            nb_vias_x = (width - w - 2 * g) / pitch_x + 1
            nb_vias_y = (height - h - 2 * g) / pitch_y + 1

            nb_vias_x = int(floor(nb_vias_x)) or 1
            nb_vias_y = int(floor(nb_vias_y)) or 1
            ref = c.add_array(
                via_type, columns=nb_vias_x, rows=nb_vias_y, spacing=(pitch_x, pitch_y)
            )

            cw = (width - (nb_vias_x - 1) * pitch_x - w) / 2
            ch = (height - (nb_vias_y - 1) * pitch_y - h) / 2
            x0 = -a + cw + w / 2
            y0 = -b + ch + h / 2
            ref.move((x0, y0))

    return c


def optimized_via(
    size: Tuple[float, float] = (11.0, 11.0),
    min_via_size: Tuple[float, float] = (0.3, 0.3),
    min_via_gap: Tuple[float, float] = (0.1, 0.1),
    min_via_enclosure: float = 0.2,
    layer: LayerSpecs = "VIA1",
    bbox_layers: Optional[LayerSpecs] = None,
    bbox_offset: float = 0,
) -> Component:
    """Given a target total inclusion size, returns an optimized dimension for the via.

    Uses inclusion, minimum width, and minimum spacing rules to place the maximum number of individual vias, with maximum via area.

    Arguments:
        size: of the target enclosing medium
        min_via_size: minimum size the vias can take
        min_via_gap: minimum distance between vias
        min_via_enclosure: minimum distance between edge of enclosing medium and nearest via edge
        layer: of the returned via component
        bbox_layers: of the returned via component
        bbox_offset: of the returned via component
    """
    via_size = [0, 0]
    for dim in [0, 1]:
        via_area = size[dim] - 2 * min_via_enclosure + min_via_gap[dim]
        num_vias = int(via_area / (min_via_size[dim] + min_via_gap[dim]))
        try:
            via_size[dim] = float(via_area / num_vias) - min_via_gap[dim]
        except ZeroDivisionError as e:
            raise RuntimeError(
                "Cannot fit vias with specific minimum dimensions in provided space."
            ) from e

    return via(
        size=via_size,
        gap=min_via_gap,
        spacing=None,
        enclosure=min_via_enclosure,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offset=min_via_enclosure,
    )


via_stack_m1_m3 = gf.partial(
    via_stack,
    layers=("M1", "M2", "M3"),
    vias=(via1, via2),
)

via_stack_slab_m3 = gf.partial(
    via_stack,
    layers=("SLAB90", "M1", "M2", "M3"),
    vias=(viac, via1, via2),
)
via_stack_npp_m1 = gf.partial(
    via_stack,
    layers=("WG", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_slab_npp_m3 = gf.partial(
    via_stack,
    layers=("SLAB90", "NPP", "M1"),
    vias=(None, None, viac),
)
via_stack_heater_mtop = via_stack_heater_m3 = gf.partial(
    via_stack, layers=("HEATER", "M2", "M3"), vias=(via1, via2)
)


if __name__ == "__main__":
    # c = via_stack_m1_m3()
    # print(c.to_dict())
    # c.show(show_ports=True)

    c = via_stack(
        size=(2.0, 2.0),
        layers=("M1", "M2", "M3"),
        vias=(optimized_via(layer="VIA1"), optimized_via(layer="VIA2")),
    )
    c.show(show_ports=True)
