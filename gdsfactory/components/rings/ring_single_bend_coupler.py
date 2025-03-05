from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bends.bend_circular import bend_circular_all_angle
from gdsfactory.typings import AnyComponentFactory, ComponentSpec, CrossSectionSpec


@gf.cell
def coupler_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 120.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: AnyComponentFactory = bend_circular_all_angle,
    bend_output: ComponentSpec = "bend_euler",
) -> Component:
    r"""Compact curved coupler with bezier escape.

    TODO: fix for euler bends.

    Args:
        radius: um.
        coupler_gap: um.
        coupling_angle_coverage: degrees.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        bend: for bend.
        bend_output: for bend.

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

    width = xo.width / 2 + xi.width / 2
    spacing = gap + width

    bend90_inner_right = gf.get_component(
        bend,  # type: ignore[arg-type]
        radius=radius,
        cross_section=cross_section_inner,
        angle=angle_inner,
    )
    bend_output_right = gf.get_component(
        bend,  # type: ignore[arg-type]
        radius=radius + spacing,
        cross_section=cross_section_outer,
        angle=angle_outer,
    )
    bend_inner_ref = c.create_vinst(bend90_inner_right)
    bend_output_ref = c.create_vinst(bend_output_right)

    output = gf.get_component(
        bend_output, angle=angle_outer, cross_section=cross_section_outer
    )
    output_ref = c.create_vinst(output)
    output_ref.connect("o1", bend_output_ref.ports["o2"], mirror=True)

    pbw = bend_inner_ref.ports["o1"]
    bend_inner_ref.dmovey(pbw.center[1] + spacing)

    c.add_port("o1", port=bend_output_ref.ports["o1"])
    c.add_port("o2", port=bend_inner_ref.ports["o1"])
    c.add_port("o3", port=output_ref.ports["o2"])
    c.add_port("o4", port=bend_inner_ref.ports["o2"])
    return c


@gf.cell
def coupler_ring_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 90.0,
    length_x: float = 0.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: AnyComponentFactory = bend_circular_all_angle,
    bend_output: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
) -> Component:
    r"""Two back-to-back coupler_bend.

    Args:
        radius: um.
        coupler_gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        coupling_angle_coverage: degrees.
        length_x: horizontal straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        bend: for bend.
        bend_output: for bend.
        straight: for straight.
    """
    c = Component()
    cp = coupler_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend,
        bend_output=bend_output,
    )
    sin = gf.get_component(straight, length=length_x, cross_section=cross_section_inner)
    sout = gf.get_component(
        straight, length=length_x, cross_section=cross_section_outer
    )

    coupler_right = c << cp
    coupler_left = c << cp
    straight_inner = c << sin
    straight_inner.dmovex(-length_x / 2)
    straight_outer = c << sout
    straight_outer.dmovex(-length_x / 2)

    coupler_left.connect("o1", straight_outer.ports["o1"])
    straight_inner.connect("o1", coupler_left.ports["o2"])
    coupler_right.connect("o2", straight_inner.ports["o2"], mirror=True)
    straight_outer.connect("o2", coupler_right.ports["o1"])

    c.add_port("o1", port=coupler_left.ports["o3"])
    c.add_port("o2", port=coupler_left.ports["o4"])
    c.add_port("o4", port=coupler_right.ports["o3"])
    c.add_port("o3", port=coupler_right.ports["o4"])
    # c.flatten()
    return c


@gf.cell
def ring_single_bend_coupler(
    radius: float = 5.0,
    gap: float = 0.2,
    coupling_angle_coverage: float = 180.0,
    bend_all_angle: AnyComponentFactory = bend_circular_all_angle,
    bend: ComponentSpec = "bend_circular",
    bend_output: ComponentSpec = "bend_euler",
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    r"""Returns ring with curved coupler.

    TODO: enable euler bends.

    Args:
        radius: um.
        gap: um.
        coupling_angle_coverage: degrees.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend_all_angle: for bend.
        bend: for bend.
        bend_output: for bend.
        length_x: horizontal straight length.
        length_y: vertical straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        kwargs: cross_section settings.
    """
    c = Component()

    coupler = coupler_ring_bend(
        radius=radius,
        coupler_gap=gap,
        coupling_angle_coverage=coupling_angle_coverage,
        length_x=length_x,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend_all_angle,
        bend_output=bend_output,
    )
    cb = c << coupler

    cross_section = cross_section_inner
    straight = gf.c.straight
    sx = gf.get_component(
        straight, length=length_x, cross_section=cross_section, **kwargs
    )
    sy = gf.get_component(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )
    b = gf.get_component(bend, cross_section=cross_section, radius=radius, **kwargs)
    sl = c << sy
    sr = c << sy
    bl = c << b
    br = c << b
    st = c << sx

    sl.connect(port="o1", other=cb["o2"])
    bl.connect(port="o2", other=sl["o2"], mirror=True)
    st.connect(port="o2", other=bl["o1"])
    sr.connect(port="o1", other=br["o1"])
    sr.connect(port="o2", other=cb["o3"])
    br.connect(port="o2", other=st["o1"], mirror=True)

    c.add_port("o2", port=cb["o4"])
    c.add_port("o1", port=cb["o1"])
    c.flatten()
    return c


if __name__ == "__main__":
    # c = coupler_bend()
    # n = c.get_netlist()
    c = coupler_ring_bend()
    # c = ring_single_bend_coupler()
    c.pprint_ports()
    c.show()
