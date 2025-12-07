from __future__ import annotations

__all__ = ["ring_asymmetric", "ring_crow"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
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

    bends = bends or (gf.c.bend_circular,) * len(radius)
    input_straight_cross_section = input_straight_cross_section or cross_section
    output_straight_cross_section = output_straight_cross_section or cross_section

    output_straight_cross_section = gf.get_cross_section(output_straight_cross_section)
    input_straight_cross_section = gf.get_cross_section(input_straight_cross_section)

    straight = gf.c.straight

    # Input bus
    input_straight = gf.get_component(
        straight,
        length=2 * radius[0] + length_x,
        cross_section=input_straight_cross_section,
    )
    input_straight_cross_section = gf.get_cross_section(input_straight_cross_section)
    input_straight_width = input_straight_cross_section.width

    input_straight_waveguide = c.add_ref(input_straight).movex(-radius[0])
    c.add_port(name="o1", port=input_straight_waveguide.ports["o1"])
    c.add_port(name="o2", port=input_straight_waveguide.ports["o2"])

    # Cascade rings
    cum_y_dist = input_straight_width / 2

    for gap, r, bend, cross_section, length_y in zip(
        gaps, radius, bends, ring_cross_sections, lengths_y, strict=False
    ):
        gap = gf.snap.snap_to_grid(gap, grid_factor=2)
        ring = ring_asymmetric(
            radius=r,
            length_x=length_x,
            length_y=length_y,
            straight=straight,
            bend=bend,
            cross_section=cross_section,
        )
        xs = gf.get_cross_section(cross_section)
        bend_width = xs.width
        ring_ref = c.add_ref(ring)
        ring_ref.movey(cum_y_dist + gap + bend_width / 2)
        cum_y_dist += gap + bend_width + 2 * r + length_y

    # Output bus
    output_straight = gf.get_component(
        straight,
        length=2 * radius[-1] + length_x,
        cross_section=output_straight_cross_section,
    )
    output_straight_width = output_straight_cross_section.width
    output_straight_waveguide = (
        c.add_ref(output_straight)
        .movey(cum_y_dist + gaps[-1] + output_straight_width / 2)
        .movex(-radius[-1])
    )
    c.add_port(name="o3", port=output_straight_waveguide.ports["o1"])
    c.add_port(name="o4", port=output_straight_waveguide.ports["o2"])
    return c


@gf.cell_with_module_name
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
    ring = Component()

    bend_c = gf.get_component(bend, radius=radius, cross_section=cross_section)

    bend1 = ring.add_ref(bend_c, name="bot_right_bend_ring")
    bend2 = ring.add_ref(bend_c, name="top_right_bend_ring")
    bend3 = ring.add_ref(bend_c, name="top_left_bend_ring")
    bend4 = ring.add_ref(bend_c, name="bot_left_bend_ring")

    straight_hor_c = gf.get_component(
        straight, length=length_x, cross_section=cross_section
    )
    straight_ver_c = gf.get_component(
        straight, length=length_y, cross_section=cross_section
    )
    straight_hor1 = ring.add_ref(straight_hor_c, name="bot_hor_waveguide_ring")
    straight_hor2 = ring.add_ref(straight_hor_c, name="top_hor_waveguide_ring")
    straight_ver1 = ring.add_ref(straight_ver_c, name="right_ver_waveguide_ring")
    straight_ver2 = ring.add_ref(straight_ver_c, name="left_ver_waveguide_ring")

    bend1.connect("o1", straight_hor1.ports["o2"])
    straight_ver1.connect("o1", bend1.ports["o2"])
    bend2.connect("o1", straight_ver1.ports["o2"])
    straight_hor2.connect("o1", bend2.ports["o2"])
    bend3.connect("o1", straight_hor2.ports["o2"])
    straight_ver2.connect("o1", bend3.ports["o2"])
    bend4.connect("o1", straight_ver2.ports["o2"])

    return ring
