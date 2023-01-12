from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.types import CrossSectionSpec


@gf.cell
def coupler_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 120.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Compact curved coupler with bezier escape.

    Args:
        radius: um.
        gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.

        .. code::
            r   4
            |   |
            |  / ___3
            | / /
        2____/ /
        1_____/
    """
    c = Component()

    xi = gf.get_cross_section(cross_section_inner)
    xo = gf.get_cross_section(cross_section_outer)

    angle_inner = 90
    angle_outer = coupling_angle_coverage / 2
    gap = coupler_gap
    bend = gf.components.bend_circular

    width = xo.width / 2 + xi.width / 2
    spacing = gap + width

    bend90_inner_right = gf.get_component(
        bend, radius=radius, cross_section=cross_section_inner, angle=angle_inner
    )
    bend_outer_right = gf.get_component(
        bend,
        radius=radius + spacing,
        cross_section=cross_section_outer,
        angle=angle_outer,
    )
    bend_inner_ref = c << bend90_inner_right
    bend_outer_ref = c << bend_outer_right

    output = gf.get_component(bend_euler, angle=angle_outer).mirror()

    output_ref = c << output
    output_ref.connect("o1", bend_outer_ref.ports["o2"])

    pbw = bend_inner_ref.ports["o1"]
    bend_inner_ref.movey(pbw.center[1] + spacing)

    # This component is a leaf cell => using absorb
    c.absorb(bend_outer_ref)
    c.absorb(bend_inner_ref)
    c.absorb(output_ref)

    c.add_port("o1", port=bend_outer_ref.ports["o1"])
    c.add_port("o2", port=bend_inner_ref.ports["o1"])
    c.add_port("o3", port=output_ref.ports["o2"])
    c.add_port("o4", port=bend_inner_ref.ports["o2"])
    return c


@gf.cell
def coupler_ring_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 180.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Two back-to-back coupler_bend.

    Args:
        radius: um.
        gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
    """
    c = Component()

    coupler_right = c << coupler_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
    )
    coupler_left = (
        c
        << coupler_bend(
            radius=radius,
            coupler_gap=coupler_gap,
            coupling_angle_coverage=coupling_angle_coverage,
            cross_section_inner=cross_section_inner,
            cross_section_outer=cross_section_outer,
        ).mirror()
    )

    coupler_left.connect("o1", coupler_right.ports["o1"])

    # This component is a leaf cell => using absorb
    c.absorb(coupler_right)
    c.absorb(coupler_left)

    c.add_port("o1", port=coupler_left.ports["o3"])
    c.add_port("o2", port=coupler_left.ports["o4"])
    c.add_port("o3", port=coupler_right.ports["o3"])
    c.add_port("o4", port=coupler_right.ports["o4"])
    return c


def ring_single_bend_coupler(
    radius: float = 5.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 90.0,
    bend: CrossSectionSpec = bend_euler,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns ring with curved coupler.

    Args:
        radius: um.
        gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
    """
    c = Component()

    coupler = c << coupler_ring_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
    )

    bend_right = c << bend(radius=radius, cross_section=cross_section_inner)
    bend_left = c << bend(radius=radius, cross_section=cross_section_inner)

    bend_right.connect("o1", coupler.ports["o4"])
    bend_left.connect("o1", bend_right.ports["o2"])

    cross_section_inner = gf.get_cross_section(cross_section_inner)

    p_in = gf.path.arc(radius=radius - 2, angle=180, start_angle=0)
    p_in = c << p_in.extrude(cross_section_inner(width=1)).movey(2.5)

    p_out = gf.path.arc(radius=radius + 2, angle=160, start_angle=10)
    p_out = c << p_out.extrude(cross_section_inner(width=2)).movey(-0.25).movex(0)

    c.add_port("o1", port=coupler.ports["o1"])
    c.add_port("o2", port=coupler.ports["o3"])
    return c


if __name__ == "__main__":
    c = ring_single_bend_coupler(
        # radius=5,
        # coupler_gap=0.2,
        # cross_section_outer="rib",
        # cross_section_inner="rib",
        # coupling_angle_coverage=30,
    )
    # c.assert_ports_on_grid()
    c.show(show_ports=True)
