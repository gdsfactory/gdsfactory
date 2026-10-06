from __future__ import annotations

__all__ = ["ring_double_heater", "ring_single_heater"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import AngleInDegrees, ComponentSpec, CrossSectionSpec, Float2

from .._schematic import ring_double_schematic


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_double_heater(
    gap: float = 0.2,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float | None = None,
    length_x: float = 1.0,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = "via_stack_heater_mtop_mini",
    port_orientation: AngleInDegrees | None = None,
    via_stack_offset: Float2 = (1, 0),
    via_stack_size: Float2 | None = None,
    with_drop: bool = True,
    length_extension: float | None = None,
    length_extension_top: float | None = None,
    length_extension_bot: float | None = None,
) -> Component:
    """Returns a double bus ring with heater on top.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        coupler_ring_top: ring coupler spec for coupler away from vias (defaults to coupler_ring)
        straight: straight spec.
        bend: bend spec.
        cross_section_heater: for heater.
        cross_section_waveguide_heater: for waveguide with heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_size: size of via_stack.
        via_stack_offset: x,y offset for via_stack.
        with_drop: adds drop ports.
        length_extension: straight length extension at the end of the coupler bottom ports.
        length_extension_top: straight length extension at the end of the coupler top ports.
        length_extension_bot: straight length extension at the end of the coupler bottom ports.

    ```text
           o2──────▲─────────o3
                   │gap_top
           xx──────▼─────────xxx
          xxx                   xxx
        xxx                       xxx
       xx                           xxx
       x                             xxx
      xx                              xx▲
      xx                              xx│length_y
      xx                              xx▼
      xx                             xx
       xx          length_x          x
        xx     ◄───────────────►    x
         xx                       xxx
           xx                   xxx
            xxx──────▲─────────xxx
                     │gap
             o1──────▼─────────o4
    ```
    """
    return cf.ring_double_heater(
        gap=gap,
        gap_top=gap_top,
        gap_bot=gap_bot,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        coupler_ring=coupler_ring,
        coupler_ring_top=coupler_ring_top,
        straight=straight,
        bend=bend,
        cross_section_heater=cross_section_heater,
        cross_section_waveguide_heater=cross_section_waveguide_heater,
        cross_section=cross_section,
        via_stack=via_stack,
        port_orientation=port_orientation,
        via_stack_offset=via_stack_offset,
        via_stack_size=via_stack_size,
        with_drop=with_drop,
        length_extension=length_extension,
        length_extension_top=length_extension_top,
        length_extension_bot=length_extension_bot,
    )


ring_single_heater = CellAlias(ring_double_heater, with_drop=False)
