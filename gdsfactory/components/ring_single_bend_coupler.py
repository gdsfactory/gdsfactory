from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.components.bend_circular import bend_circular_all_angle
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.vcell
def coupler_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 120.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: ComponentSpec = bend_circular_all_angle,
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

    .. code::

            r   4
            |   |
            |  / ___3
            | / /
        2____/ /
        1_____/
    """
    c = ComponentAllAngle()

    xi = gf.get_cross_section(cross_section_inner)
    xo = gf.get_cross_section(cross_section_outer)

    angle_inner = 90
    angle_outer = coupling_angle_coverage / 2
    gap = coupler_gap

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

    output = gf.get_component(bend_euler, angle=angle_outer)
    output_ref = c << output
    output_ref.connect("o1", bend_outer_ref.ports["o2"], mirror=True)

    pbw = bend_inner_ref.ports["o1"]
    bend_inner_ref.dmovey(pbw.dcenter[1] + spacing)

    c.add_port("o1", port=bend_outer_ref.ports["o1"])
    c.add_port("o2", port=bend_inner_ref.ports["o1"])
    c.add_port("o3", port=output_ref.ports["o2"])
    c.add_port("o4", port=bend_inner_ref.ports["o2"])
    return c


@gf.vcell
def coupler_ring_bend(
    radius: float = 10.0,
    coupler_gap: float = 0.2,
    coupling_angle_coverage: float = 90.0,
    length_x: float = 0.0,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    bend: ComponentSpec = bend_circular_all_angle,
) -> ComponentAllAngle:
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
    """
    c = ComponentAllAngle()
    cp = coupler_bend(
        radius=radius,
        coupler_gap=coupler_gap,
        coupling_angle_coverage=coupling_angle_coverage,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend,
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
    bend: ComponentSpec = bend_circular_all_angle,
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns ring with curved coupler.

    TODO: enable euler bends.

    Args:
        radius: um.
        gap: um.
        coupling_angle_coverage: degrees.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
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
        bend=bend,
    )
    cb = c.create_vinst(coupler)

    cross_section = cross_section_inner
    sx = gf.get_component(
        straight, length=length_x, cross_section=cross_section, **kwargs
    )
    sy = gf.get_component(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )
    b = gf.get_component(bend, cross_section=cross_section, radius=radius, **kwargs)
    sl = c.create_vinst(sy)
    sr = c.create_vinst(sy)
    bl = c.create_vinst(b)
    br = c.create_vinst(b)
    st = c.create_vinst(sx)

    sl.connect(port="o1", other=cb["o2"])
    bl.connect(port="o2", other=sl["o2"], mirror=True)
    st.connect(port="o2", other=bl["o1"])
    sr.connect(port="o1", other=br["o1"])
    sr.connect(port="o2", other=cb["o3"])
    br.connect(port="o2", other=st["o1"])

    c.add_port("o2", port=cb["o4"])
    c.add_port("o1", port=cb["o1"])
    return c


if __name__ == "__main__":
    # c = coupler_bend(radius=5)
    # c = coupler_ring_bend()
    c = ring_single_bend_coupler()
    c.show()
