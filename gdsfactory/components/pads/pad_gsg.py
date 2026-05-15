"""High speed GSG pads."""

from __future__ import annotations

__all__ = ["pad_gs", "pad_gsg", "pad_gsg_open", "pad_gsg_short"]

from functools import partial
from typing import cast

import kfactory as kf

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec

from .._schematic import pad_schematic


@gf.cell_with_module_name(tags=["pads"])
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
        _ = c << via
    gnd_bot = c << via

    gnd_bot.ymax = via.ymin
    gnd_top.ymin = via.ymax

    gnd_top.movex(-metal_spacing)
    gnd_bot.movex(-metal_spacing)

    pads = c << gf.components.array(
        pad, columns=1, rows=3, column_pitch=0, row_pitch=pad_pitch, centered=True
    )
    pads.xmin = via.xmax + route_xsize
    pads.y = 0

    gf.routing.route_quad(
        c, gnd_bot.ports["e4"], pads.ports["e1_1_1"], layer=layer_metal
    )
    gf.routing.route_quad(
        c,
        cast("kf.DPort", gnd_top.ports["e2"]),  # type: ignore[redundant-cast]
        cast("kf.DPort", pads.ports["e1_3_1"]),  # type: ignore[redundant-cast]
        layer=layer_metal,
    )
    gf.routing.route_quad(
        c,
        cast("kf.DPort", via.ports["e3"]),  # type: ignore[redundant-cast]
        cast("kf.DPort", pads.ports["e1_2_1"]),  # type: ignore[redundant-cast]
        layer=layer_metal,
    )
    return c


pad_gsg_open = partial(pad_gsg_short, short=False)


@gf.cell_with_module_name(schematic_function=pad_schematic, tags=["pads"])
def pad_gsg(length: float = 100, cross_section: str = "gsg") -> gf.Component:
    c = gf.c.straight(cross_section=cross_section, length=length)
    for port in c.ports:
        if port.port_type == "electrical":
            c.create_pin(ports=[port], name=port.name)
    return c


@gf.cell_with_module_name(tags=["pads"])
def pad_gs(length: float = 100, cross_section: str = "gs") -> gf.Component:
    return gf.c.straight(cross_section=cross_section, length=length)


if __name__ == "__main__":
    c = pad_gs()
    c.show()
