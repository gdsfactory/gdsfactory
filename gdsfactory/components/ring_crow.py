from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.straight import straight
from gdsfactory.typings import (
    ComponentFactories,
    CrossSectionSpec,
    CrossSectionSpecs,
)


@gf.cell
def ring_crow(
    gaps: tuple[float, ...] = (0.2,) * 4,
    radius: tuple[float, ...] = (10.0,) * 3,
    input_straight_cross_section: CrossSectionSpec = "xs_sc",
    output_straight_cross_section: CrossSectionSpec = "xs_sc",
    bends: ComponentFactories = (bend_circular,) * 3,
    ring_cross_sections: CrossSectionSpecs = ("xs_sc",) * 3,
) -> Component:
    """Coupled ring resonators.

    Args:
        gaps: gap between for coupler.
        radius: for the bend and coupler.
        input_straight_cross_section: input straight cross_section.
        output_straight_cross_section: output straight cross_section.
        bends: bend component.
        ring_cross_sections: ring cross_sections.

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

    xs1 = gf.get_cross_section(input_straight_cross_section)
    xs2 = gf.get_cross_section(output_straight_cross_section)

    # Input bus
    input_straight = straight(length=2 * radius[0], cross_section=xs1)
    input_straight_width = xs1.width

    input_straight_waveguide = c.add_ref(input_straight)
    input_straight_waveguide.movex(-radius[0])
    c.add_port(name="o1", port=input_straight_waveguide.ports["o1"])
    c.add_port(name="o2", port=input_straight_waveguide.ports["o2"])

    # Cascade rings
    cum_y_dist = input_straight_width / 2

    for index, (gap, r, bend, cross_section) in enumerate(
        zip(gaps, radius, bends, ring_cross_sections)
    ):
        gap = gf.snap.snap_to_grid(gap, grid_factor=2)
        ring = Component(f"ring{index}")

        bend_c = bend(radius=r, cross_section=cross_section)
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
        ring_ref.d.movey(cum_y_dist + gap + bend_width / 2)
        c.absorb(ring_ref)
        cum_y_dist += gap + bend_width + 2 * r

    # Output bus
    output_straight = gf.get_component(
        straight,
        length=2 * radius[-1],
        cross_section=xs2,
    )
    output_straight_width = xs2.width
    output_straight_waveguide = c.add_ref(output_straight)
    output_straight_waveguide.d.movey(cum_y_dist + gaps[-1] + output_straight_width / 2)
    output_straight_waveguide.d.movex(-radius[-1])
    c.add_port(name="o3", port=output_straight_waveguide.ports["o1"])
    c.add_port(name="o4", port=output_straight_waveguide.ports["o2"])
    return c


if __name__ == "__main__":
    c = ring_crow(
        input_straight_cross_section="xs_rc",
        ring_cross_sections=("xs_rc", "xs_sc", "xs_rc"),
    )
    # c = ring_crow(gaps = [0.3, 0.4, 0.5, 0.2],
    #     radius = [20.0, 5.0, 15.0],
    #     input_straight_cross_section = strip,
    #     output_straight_cross_section = strip,
    #     bends = [bend_circular] * 3,
    #     ring_cross_sections = [strip] * 3,
    # )

    c.show()
