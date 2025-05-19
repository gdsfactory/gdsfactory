"""High speed GSG pads."""

from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec


@gf.cell_with_module_name
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
        c << via
    gnd_bot = c << via

    v_xmin, v_xmax = via.xmin, via.xmax
    v_ymin, v_ymax = via.ymin, via.ymax

    # --- direct ymin/ymax assignments ---
    gnd_bot.ymax = v_ymin
    gnd_top.ymin = v_ymax
    gnd_top.movex(-metal_spacing)
    gnd_bot.movex(-metal_spacing)

    pads = c << gf.components.array(
        pad, columns=1, rows=3, column_pitch=0, row_pitch=pad_pitch, centered=True
    )
    pads.xmin = v_xmax + route_xsize
    pads.y = 0

    # --- Route efficiently, no string construction in tight loops, reference ports by name directly ---
    p_gnd_bot = gnd_bot.ports["e4"]
    p_gnd_top = gnd_top.ports["e2"]
    p_via_e3 = via.ports["e3"]
    ports_pads = pads.ports
    gf.routing.route_quad(c, p_gnd_bot, ports_pads["e1_1_1"], layer=layer_metal)
    gf.routing.route_quad(c, p_gnd_top, ports_pads["e1_3_1"], layer=layer_metal)
    gf.routing.route_quad(c, p_via_e3, ports_pads["e1_2_1"], layer=layer_metal)
    return c


def _get_rotated_basis(angle: float):
    """Fast helper, used by route_quad"""
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c, s]), np.array([-s, c])


pad_gsg_open = partial(pad_gsg_short, short=False)


if __name__ == "__main__":
    # c = pad_array_double()
    c = pad_gsg_short()
    # c = pad_gsg_open()
    c.show()
