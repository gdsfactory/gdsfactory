"""High speed GSG pads."""

from __future__ import annotations

__all__ = ["pad_gs", "pad_gsg", "pad_gsg_short"]

from typing import cast

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec


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
    via = get_component("rectangle", size=size, layer=layer_metal)
    gnd_top = c << via

    if short:
        _ = c << via
    gnd_bot = c << via

    gnd_bot.ymax = via.ymin
    gnd_top.ymin = via.ymax

    gnd_top.movex(-metal_spacing)
    gnd_bot.movex(-metal_spacing)

    pads = c << get_component(
        "array",
        component=pad,
        columns=1,
        rows=3,
        column_pitch=0,
        row_pitch=pad_pitch,
        centered=True,
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


def pad_gsg(length: float = 100, cross_section: str = "gsg") -> gf.Component:
    """Returns a ground-signal-ground pad with electrical pins.

    Args:
        length: length of the GSG transmission line, in um.
        cross_section: GSG cross_section spec.
    """
    # Copy, since a PDK may register straight as a cached cell.
    c = get_component("straight", cross_section=cross_section, length=length).copy()
    for port in c.ports:
        if port.port_type == "electrical":
            c.create_pin(ports=[port], name=port.name)
    return c


def pad_gs(length: float = 100, cross_section: str = "gs") -> gf.Component:
    """Returns a ground-signal pad.

    Args:
        length: length of the GS transmission line, in um.
        cross_section: GS cross_section spec.
    """
    return get_component("straight", cross_section=cross_section, length=length).copy()
