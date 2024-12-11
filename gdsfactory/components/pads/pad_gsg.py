"""High speed GSG pads."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec


@gf.cell
def pad_gsg_short(
    size: Float2 = (22, 7),
    layer_metal: LayerSpec = "MTOP",
    metal_spacing: float = 5.0,
    short: bool = True,
    pad: ComponentSpec = "pad",
    pad_pitch: float = 150,
    route_xsize: float = 50,
) -> gf.Component:
    """Returns high speed GSG pads for calibrating the RF probes.

    Args:
        size: for the short.
        layer_metal: for the short.
        metal_spacing: in um.
        short: if False returns an open.
        pad: function for pad.
        pad_pitch: in um.
        route_xsize: in um.
    """
    c = gf.Component()
    via = gf.c.rectangle(size=size, layer=layer_metal)
    gnd_top = c << via

    if short:
        c << gf.c.rectangle(size=size, layer=layer_metal)
    gnd_bot = c << via

    gnd_bot.dymax = via.dymin
    gnd_top.dymin = via.dymax

    gnd_top.dmovex(-metal_spacing)
    gnd_bot.dmovex(-metal_spacing)

    pads = c << gf.components.array(
        pad, columns=1, rows=3, spacing=(0, pad_pitch), centered=True
    )
    pads.dxmin = via.dxmax + route_xsize
    pads.dy = 0

    gf.routing.route_quad(
        c, gnd_bot.ports["e4"], pads.ports["e1_1_1"], layer=layer_metal
    )
    gf.routing.route_quad(
        c, gnd_top.ports["e2"], pads.ports["e1_3_1"], layer=layer_metal
    )
    gf.routing.route_quad(c, via.ports["e3"], pads.ports["e1_2_1"], layer=layer_metal)
    return c


pad_gsg_open = partial(pad_gsg_short, short=False)


if __name__ == "__main__":
    # c = pad_array_double()
    c = pad_gsg_short()
    # c = pad_gsg_open()
    c.show()
