"""Straight heater meander doped."""

from __future__ import annotations

__all__ = ["straight_heater_meander_doped", "via_stack_heater_meander_doped"]

from functools import partial

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpecs

from .._schematic import straight_schematic
from ..vias.via import via
from ..vias.via_stack import via_stack

via_stack_heater_meander_doped = partial(
    via_stack,
    size=(1.5, 1.5),
    layers=("M1", "M2"),
    vias=(
        partial(
            via,
            layer="VIAC",
            size=(0.1, 0.1),
            pitch=0.2,
            enclosure=0.1,
        ),
        partial(
            via,
            layer="VIA1",
            size=(0.1, 0.1),
            pitch=0.2,
            enclosure=0.1,
        ),
    ),
)


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_meander_doped(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: CrossSectionSpec = "strip",
    heater_width: float = 1.5,
    extension_length: float = 15.0,
    layers_doping: LayerSpecs = ("P", "PP", "PPP"),
    radius: float = 5.0,
    via_stack: ComponentSpec | None = "via_stack_heater_meander_doped",
    port_orientation1: float | None = None,
    port_orientation2: float | None = None,
    straight_widths: Floats = (0.8, 0.9, 0.8),
    taper_length: float = 10,
) -> Component:
    """Returns a meander based heater.

    based on SungWon Chung, Makoto Nakai, and Hossein Hashemi,
    Low-power thermo-optic silicon modulator for large-scale photonic integrated systems
    Opt. Express 27, 13430-13459 (2019)
    <https://www.osapublishing.org/oe/abstract.cfm?URI=oe-27-9-13430>

    Args:
        length: total length of the optical path.
        spacing: waveguide spacing (center to center).
        cross_section: for waveguide.
        heater_width: for heater.
        extension_length: of input and output optical ports.
        layers_doping: doping layers to be used for heater.
        radius: for the meander bends.
        via_stack: for the heater to via_stack metal.
        port_orientation1: in degrees. None adds all orientations.
        port_orientation2: in degrees. None adds all orientations.
        straight_widths: width of the straight sections.
        taper_length: from the cross_section.
    """
    return cf.straight_heater_meander_doped(
        length=length,
        spacing=spacing,
        cross_section=cross_section,
        heater_width=heater_width,
        extension_length=extension_length,
        layers_doping=layers_doping,
        radius=radius,
        via_stack=via_stack,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        straight_widths=straight_widths,
        taper_length=taper_length,
    )
