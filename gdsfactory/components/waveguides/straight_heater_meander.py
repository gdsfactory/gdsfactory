from __future__ import annotations

__all__ = ["straight_heater_meander"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec

from .._schematic import straight_schematic


@gf.cell_with_module_name(schematic_function=straight_schematic, tags=["waveguides"])
def straight_heater_meander(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: CrossSectionSpec = "strip",
    heater_width: float = 2.5,
    extension_length: float = 15.0,
    layer_heater: LayerSpec = "HEATER",
    radius: float | None = None,
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: float | None = None,
    port_orientation2: float | None = None,
    heater_taper_length: float = 10.0,
    straight_widths: Floats | None = None,
    taper_length: float = 10.0,
    n: int | None = 3,
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
        layer_heater: for top heater, if None, it does not add a heater.
        radius: for the meander bends. Defaults to cross_section radius.
        via_stack: for the heater to via_stack metal.
        port_orientation1: in degrees. None adds all orientations.
        port_orientation2: in degrees. None adds all orientations.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        straight_widths: widths of the straight sections.
        taper_length: from the cross_section.
        n: number of straight sections.
    """
    return cf.straight_heater_meander(
        length=length,
        spacing=spacing,
        cross_section=cross_section,
        heater_width=heater_width,
        extension_length=extension_length,
        layer_heater=layer_heater,
        radius=radius,
        via_stack=via_stack,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        heater_taper_length=heater_taper_length,
        straight_widths=straight_widths,
        taper_length=taper_length,
        n=n,
    )


if __name__ == "__main__":
    c = straight_heater_meander(port_orientation1=None, port_orientation2=90)
    c.show()
