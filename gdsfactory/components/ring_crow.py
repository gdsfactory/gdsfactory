from __future__ import annotations

from typing import List

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_crow(
    gaps: List[float] = [0.2] * 4,
    radii: List[float] = [10.0] * 3,
    input_straight_cross_section: CrossSectionSpec = strip,
    output_straight_cross_section: CrossSectionSpec = strip,
    bends: List[ComponentSpec] = [bend_circular] * 3,
    ring_cross_sections: List[CrossSectionSpec] = [strip] * 3,
) -> Component:
    """Coupled ring resonators.

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler: ring coupler spec.
        straight: straight spec.
        bend: bend spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

    .. code::

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
    c = Component()

    # Input bus
    input_straight = gf.get_component(
        straight, length=2 * radii[0], cross_section=input_straight_cross_section
    )
    input_straight_width = input_straight_cross_section().width
    input_straight_waveguide = c.add_ref(input_straight).movex(-radii[0])
    c.add_port(name="o1", port=input_straight_waveguide.ports["o1"])
    c.add_port(name="o2", port=input_straight_waveguide.ports["o2"])

    # Cascade rings
    cum_y_dist = input_straight_width / 2
    for gap, radius, bend, cross_section in zip(
        gaps, radii, bends, ring_cross_sections
    ):
        gap = gf.snap.snap_to_grid(gap, nm=2)
        # Create ring from 4 bends
        ring = Component()
        bend_c = gf.get_component(bend, radius=radius, cross_section=cross_section)
        bend_width = cross_section().width
        bend1 = ring.add_ref(bend_c)
        bend2 = ring.add_ref(bend_c)
        bend3 = ring.add_ref(bend_c)
        bend4 = ring.add_ref(bend_c)

        bend2.connect("o1", bend1.ports["o2"])
        bend3.connect("o1", bend2.ports["o2"])
        bend4.connect("o1", bend3.ports["o2"])

        c.add_ref(ring).movey(cum_y_dist + gap + bend_width / 2)
        cum_y_dist += gap + bend_width + 2 * radius

    # Output bus
    output_straight = gf.get_component(
        straight,
        length=2 * radii[-1],
        cross_section=output_straight_cross_section,
    )
    output_straight_width = output_straight_cross_section().width
    output_straight_waveguide = (
        c.add_ref(output_straight)
        .movey(cum_y_dist + gaps[-1] + output_straight_width / 2)
        .movex(-radii[-1])
    )
    c.add_port(name="o3", port=output_straight_waveguide.ports["o1"])
    c.add_port(name="o4", port=output_straight_waveguide.ports["o2"])

    print()

    return c


if __name__ == "__main__":

    c = ring_crow()
    # c = ring_crow(gaps = [0.3, 0.4, 0.5, 0.2],
    #     radii = [20.0, 5.0, 15.0],
    #     input_straight_cross_section = strip,
    #     output_straight_cross_section = strip,
    #     bends = [bend_circular] * 3,
    #     ring_cross_sections = [strip] * 3,
    # )

    c.show(show_ports=True, show_subports=False)
