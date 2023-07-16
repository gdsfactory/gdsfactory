"""High speed GSG pads."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.components.pad import pad as pad_function
from gdsfactory.components.rectangle import rectangle
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec

rectangle_m3 = partial(rectangle, layer="M3")


@gf.cell
def pad_gsg_short(
    via_stack: ComponentSpec = rectangle_m3,
    size: Float2 = (22, 7),
    layer_metal: LayerSpec = "M3",
    metal_spacing: float = 5.0,
    short: bool = True,
    pad: ComponentSpec = pad_function,
    pad_spacing: float = 150,
) -> gf.Component:
    """Returns high speed GSG pads for calibrating the RF probes.

    Args:
        via_stack: where the RF pads connect to.
        size: for the via_stack.
        layer_metal: for the short.
        metal_spacing: in um.
        short: if False returns an open.
        pad: function for pad.
        pad_spacing: in um.
    """
    c = gf.Component()

    via = gf.get_component(via_stack, size=size)
    gnd_top = c << via

    if short:
        short = c << rectangle(size=size, layer=layer_metal)
    gnd_bot = c << via

    gnd_bot.ymax = via.ymin
    gnd_top.ymin = via.ymax

    gnd_top.movex(-metal_spacing)
    gnd_bot.movex(-metal_spacing)

    pads = c << gf.components.array(pad, columns=1, rows=3, spacing=(0, pad_spacing))
    pads.xmin = via.xmax + 50
    pads.y = 0

    c << gf.routing.route_quad(
        gnd_bot.ports["e4"], pads.ports["e1_1_1"], layer=layer_metal
    )
    c << gf.routing.route_quad(
        gnd_top.ports["e2"], pads.ports["e1_3_1"], layer=layer_metal
    )
    c << gf.routing.route_quad(via.ports["e3"], pads.ports["e1_2_1"], layer=layer_metal)
    return c


pad_gsg_open = partial(pad_gsg_short, short=False)


if __name__ == "__main__":
    # c = pad_array_double()
    c = pad_gsg_short()
    # c = pad_gsg_open()
    c.show(show_ports=True)
