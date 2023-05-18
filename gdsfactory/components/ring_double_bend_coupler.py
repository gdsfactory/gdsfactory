from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.ring_single_bend_coupler import coupler_ring_bend
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_double_bend_coupler(
    radius: float = 5.0,
    gap: float = 0.2,
    coupling_angle_coverage: float = 70.0,
    bend: ComponentSpec = bend_circular,
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns ring with double curved couplers.

    TODO: enable euler bends.

    Args:
        radius: um.
        gap: um.
        angle_inner: of the inner bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        angle_outer: of the outer bend, from beginning to end. Depending on the bend chosen, gap may not be preserved.
        bend: for bend.
        length_y: vertical straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
        kwargs: cross_section settings.
    """
    c = Component()

    c_halfring = coupler_ring_bend(
        radius=radius,
        coupler_gap=gap,
        coupling_angle_coverage=coupling_angle_coverage,
        length_x=length_x,
        cross_section_inner=cross_section_inner,
        cross_section_outer=cross_section_outer,
        bend=bend,
    )

    xi = gf.get_cross_section(cross_section_inner)
    xo = gf.get_cross_section(cross_section_outer)
    half_height = radius + xi.width / 2 + gap + xo.width + length_y / 2

    if c_halfring.ysize > half_height:
        raise ValueError(
            "The coupling_angle_coverage is too large for the given bend radius: "
            + "the coupling waveguides will overlap."
        )

    cb = c << c_halfring
    ct = c << c_halfring

    cross_section = cross_section_inner
    sy = straight(length=length_y, cross_section=cross_section, **kwargs)
    sl = c << sy
    sr = c << sy

    sl.connect(port="o1", destination=cb.ports["o2"])
    ct.connect(port="o3", destination=sl.ports["o2"])
    sr.connect(port="o1", destination=ct.ports["o2"])
    cb.connect(port="o3", destination=sr.ports["o2"])

    c.absorb(cb)
    c.absorb(ct)
    c.absorb(sl)
    c.absorb(sr)

    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=ct.ports["o4"])
    c.add_port("o3", port=ct.ports["o1"])
    c.add_port("o4", port=cb.ports["o4"])
    return c


if __name__ == "__main__":
    # c = coupler_bend(radius=5)
    # c = coupler_ring_bend()
    c = ring_double_bend_coupler()
    c.assert_ports_on_grid()
    c.show(show_ports=True)
