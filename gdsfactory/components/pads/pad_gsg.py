"""High speed GSG pads."""

from __future__ import annotations

__all__ = ["pad_gs", "pad_gsg", "pad_gsg_open", "pad_gsg_short"]

from functools import partial

import gdsfactory as gf
from gdsfactory import component_functions as cf
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
    return cf.pad_gsg_short(
        size=size,
        layer_metal=layer_metal,
        metal_spacing=metal_spacing,
        short=short,
        pad=pad,
        pad_pitch=pad_pitch,
        route_xsize=route_xsize,
    )


pad_gsg_open = partial(pad_gsg_short, short=False)


@gf.cell_with_module_name(schematic_function=pad_schematic, tags=["pads"])
def pad_gsg(length: float = 100, cross_section: str = "gsg") -> gf.Component:
    """Returns a ground-signal-ground pad with electrical pins.

    Args:
        length: length of the GSG transmission line, in um.
        cross_section: GSG cross_section spec.
    """
    return cf.pad_gsg(
        length=length,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(tags=["pads"])
def pad_gs(length: float = 100, cross_section: str = "gs") -> gf.Component:
    """Returns a ground-signal pad.

    Args:
        length: length of the GS transmission line, in um.
        cross_section: GS cross_section spec.
    """
    return cf.pad_gs(
        length=length,
        cross_section=cross_section,
    )


if __name__ == "__main__":
    c = pad_gs()
    c.show()
