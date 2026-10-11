from __future__ import annotations

__all__ = ["ring_asymmetric", "ring_crow"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import ring_double_schematic


@gf.cell_with_module_name(schematic_function=ring_double_schematic, tags=["rings"])
def ring_crow(
    gaps: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2),
    radius: tuple[float, ...] = (10.0, 10.0, 10.0),
    bends: tuple[ComponentSpec, ...] | None = None,
    ring_cross_sections: tuple[CrossSectionSpec, ...] = ("strip", "strip", "strip"),
    length_x: float = 0,
    lengths_y: tuple[float, ...] = (0, 0, 0),
    input_straight_cross_section: CrossSectionSpec | None = None,
    output_straight_cross_section: CrossSectionSpec | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Coupled ring resonators.

    Args:
        gaps: gap between rings.
        radius: for each ring.
        bends: bend spec for each ring.
        ring_cross_sections: cross_section spec for each ring.
        length_x: ring coupler length.
        lengths_y: vertical straight length.
        input_straight_cross_section: cross_section spec for input and output straight. Defaults to cross_section.
        output_straight_cross_section: cross_section spec for input and output straight. Defaults to cross_section.
        cross_section: cross_section spec for input and output straight.

         --==ct==-- gap[N-1]
          |      |
          sl     sr ring[N-1]
          |      |
         --==cb==-- gap[N-2]

             .
             .
             .

         --==ct==--
          |      |
          sl     sr lengths_y[1], ring[1]
          |      |
         --==cb==-- gap[1]

         --==ct==--
          |      |
          sl     sr lengths_y[0], ring[0]
          |      |
         --==cb==-- gap[0]

          length_x
    """
    return cf.ring_crow(
        gaps=gaps,
        radius=radius,
        bends=bends,
        ring_cross_sections=ring_cross_sections,
        length_x=length_x,
        lengths_y=lengths_y,
        input_straight_cross_section=input_straight_cross_section,
        output_straight_cross_section=output_straight_cross_section,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(tags=["rings"])
def ring_asymmetric(
    radius: float = 10.0,
    length_x: float = 2.0,
    length_y: float = 4.0,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """An asymmetric ring with straight waveguides between the bends.

    Args:
        radius: of the ring.
        length_x: horizontal straight length.
        length_y: vertical straight length.
        straight: straight component spec.
        bend: bend component spec.
        cross_section: cross_section spec.
    """
    return cf.ring_asymmetric(
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        straight=straight,
        bend=bend,
        cross_section=cross_section,
    )
