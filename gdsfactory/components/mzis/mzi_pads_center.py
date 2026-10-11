from __future__ import annotations

__all__ = ["mzi_pads_center"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ckt_schematic


@gf.cell_with_module_name(schematic_function=ckt_schematic, tags=["mzis"])
def mzi_pads_center(
    ps_top: ComponentSpec = "straight_heater_metal",
    ps_bot: ComponentSpec = "straight_heater_metal",
    mzi: ComponentSpec = "mzi",
    pad: ComponentSpec = "pad_small",
    length_x: float = 500,
    length_y: float = 40,
    mzi_sig_top: str | None = "top_r_e2",
    mzi_gnd_top: str | None = "top_l_e2",
    mzi_sig_bot: str | None = "bot_l_e2",
    mzi_gnd_bot: str | None = "bot_r_e2",
    pad_sig_bot: str = "e1_1_1",
    pad_sig_top: str = "e3_1_3",
    pad_gnd_bot: str = "e4_1_2",
    pad_gnd_top: str = "e2_1_2",
    delta_length: float = 40.0,
    cross_section: CrossSectionSpec = "strip",
    cross_section_metal: CrossSectionSpec = "metal_routing",
    pad_pitch: float | str = "pad_pitch",
    auto_taper: bool = False,
    **kwargs: Any,
) -> gf.Component:
    """Return Mzi phase shifter with pads in the middle.

    GND is the middle pad and is shared between top and bottom phase shifters.

    Args:
        ps_top: phase shifter top.
        ps_bot: phase shifter bottom.
        mzi: interferometer.
        pad: pad function.
        length_x: horizontal length.
        length_y: vertical length.
        mzi_sig_top: port name for top phase shifter signal. None if no connection.
        mzi_gnd_top: port name for top phase shifter GND. None if no connection.
        mzi_sig_bot: port name for top phase shifter signal. None if no connection.
        mzi_gnd_bot: port name for top phase shifter GND. None if no connection.
        pad_sig_bot: port name for top pad.
        pad_sig_top: port name for top pad.
        pad_gnd_bot: port name for top pad.
        pad_gnd_top: port name for top pad.
        delta_length: mzi length imbalance.
        cross_section: for the mzi.
        cross_section_metal: for routing metal.
        pad_pitch: pad pitch in um.
        auto_taper: add taper if cross_section width is different between mzi and pad.
        kwargs: routing settings.
    """
    return cf.mzi_pads_center(
        ps_top=ps_top,
        ps_bot=ps_bot,
        mzi=mzi,
        pad=pad,
        length_x=length_x,
        length_y=length_y,
        mzi_sig_top=mzi_sig_top,
        mzi_gnd_top=mzi_gnd_top,
        mzi_sig_bot=mzi_sig_bot,
        mzi_gnd_bot=mzi_gnd_bot,
        pad_sig_bot=pad_sig_bot,
        pad_sig_top=pad_sig_top,
        pad_gnd_bot=pad_gnd_bot,
        pad_gnd_top=pad_gnd_top,
        delta_length=delta_length,
        cross_section=cross_section,
        cross_section_metal=cross_section_metal,
        pad_pitch=pad_pitch,
        auto_taper=auto_taper,
        **kwargs,
    )
