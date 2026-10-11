from __future__ import annotations

__all__ = ["resistance_sheet"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Floats, LayerSpecs, Size


@gf.cell_with_module_name(tags=["pcms"])
def resistance_sheet(
    width: float = 10.0,
    layers: LayerSpecs = ("HEATER",),
    layer_offsets: Floats = (0, 0.2),
    pad: ComponentSpec = "via_stack_heater_mtop",
    pad_size: Size = (50.0, 50.0),
    pad_pitch: float = 100.0,
    ohms_per_square: float | None = None,
    pad_port_name: str = "e4",
) -> Component:
    """Returns Sheet resistance.

    keeps connectivity for pads and first layer in layers

    Args:
        width: in um.
        layers: for the middle part.
        layer_offsets: from edge, positive: over, negative: inclusion.
        pad: function to create a pad.
        pad_size: in um.
        pad_pitch: in um.
        ohms_per_square: optional sheet resistance to compute info.resistance.
        pad_port_name: port name for the pad.
    """
    return cf.resistance_sheet(
        width=width,
        layers=layers,
        layer_offsets=layer_offsets,
        pad=pad,
        pad_size=pad_size,
        pad_pitch=pad_pitch,
        ohms_per_square=ohms_per_square,
        pad_port_name=pad_port_name,
    )
