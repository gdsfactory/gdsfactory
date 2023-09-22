from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import strip
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_crow(
    gaps: list[float] = [0.2] * 4,
    radius: list[float] = [10.0] * 3,
    input_straight_cross_section: CrossSectionSpec = strip,
    output_straight_cross_section: CrossSectionSpec = strip,
    bends: list[ComponentSpec] = [bend_circular] * 3,
    ring_cross_sections: list[CrossSectionSpec] = [strip] * 3,
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
        straight, length=2 * radius[0], cross_section=input_straight_cross_section
    )
    input_straight_cross_section = gf.get_cross_section(input_straight_cross_section)
    input_straight_width = input_straight_cross_section.width

    input_straight_waveguide = c.add_ref(input_straight).movex(-radius[0])
    c.add_port(name="o1", port=input_straight_waveguide.ports["o1"])
    c.add_port(name="o2", port=input_straight_waveguide.ports["o2"])

    # Cascade rings
    cum_y_dist = input_straight_width / 2

    for index, (gap, r, bend, cross_section) in enumerate(
        zip(gaps, radius, bends, ring_cross_sections)
    ):
        gap = gf.snap.snap_to_grid(gap, grid_factor=2)
        ring = Component(f"ring{index}")

        bend_c = gf.get_component(bend, radius=r, cross_section=cross_section)
        xs = gf.get_cross_section(cross_section)
        bend_width = xs.width
        bend1 = ring.add_ref(bend_c, alias=f"bot_right_bend_ring_{index}")
        bend2 = ring.add_ref(bend_c, alias=f"top_right_bend_ring_{index}")
        bend3 = ring.add_ref(bend_c, alias=f"top_left_bend_ring_{index}")
        bend4 = ring.add_ref(bend_c, alias=f"bot_left_bend_ring_{index}")

        bend2.connect("o1", bend1.ports["o2"])
        bend3.connect("o1", bend2.ports["o2"])
        bend4.connect("o1", bend3.ports["o2"])

        ring_ref = c.add_ref(ring)
        ring_ref.movey(cum_y_dist + gap + bend_width / 2)
        c.absorb(ring_ref)
        cum_y_dist += gap + bend_width + 2 * r

    # Output bus
    output_straight = gf.get_component(
        straight,
        length=2 * radius[-1],
        cross_section=output_straight_cross_section,
    )
    output_straight_width = output_straight_cross_section().width
    output_straight_waveguide = (
        c.add_ref(output_straight)
        .movey(cum_y_dist + gaps[-1] + output_straight_width / 2)
        .movex(-radius[-1])
    )
    c.add_port(name="o3", port=output_straight_waveguide.ports["o1"])
    c.add_port(name="o4", port=output_straight_waveguide.ports["o2"])
    return c


if __name__ == "__main__":
    c = ring_crow(
        input_straight_cross_section="rib", ring_cross_sections=["rib", "strip", "rib"]
    )
    # c = ring_crow(gaps = [0.3, 0.4, 0.5, 0.2],
    #     radius = [20.0, 5.0, 15.0],
    #     input_straight_cross_section = strip,
    #     output_straight_cross_section = strip,
    #     bends = [bend_circular] * 3,
    #     ring_cross_sections = [strip] * 3,
    # )

    c.show(show_ports=True, show_subports=False)
