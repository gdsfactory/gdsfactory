from __future__ import annotations

__all__ = ["coupler_ring"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import coupler_ring_schematic


@gf.cell_with_module_name(schematic_function=coupler_ring_schematic, tags=["couplers"])
def coupler_ring(
    gap: float = 0.2,
    radius: float | None = None,
    length_x: float = 4.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float | None = None,
) -> Component:
    r"""Coupler for ring.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends. Default is None, which uses the default radius of the cross_section.
        length_x: length of the parallel coupled straight waveguides.
        bend: 90 degrees bend spec.
        straight: straight spec.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.

    ```text
          o2                              o3
          xx                              xx
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
    return cf.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        length_extension=length_extension,
    )
