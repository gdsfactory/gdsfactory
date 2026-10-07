from __future__ import annotations

__all__ = ["ring_double"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ring_double_schematic


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_double(
    gap: float = 0.2,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float | None = None,
    length_x: float = 0.01,
    length_y: float = 0.01,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
    length_extension: float | None = None,
) -> Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        coupler_ring_top: top ring coupler spec. Defaults to coupler_ring.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.

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
             o1──────▼─────────◄──────────────► o4
                                length_extension
    ```
    """
    return cf.ring_double(
        gap=gap,
        gap_top=gap_top,
        gap_bot=gap_bot,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend=bend,
        straight=straight,
        coupler_ring=coupler_ring,
        coupler_ring_top=coupler_ring_top,
        cross_section=cross_section,
        length_extension=length_extension,
    )
