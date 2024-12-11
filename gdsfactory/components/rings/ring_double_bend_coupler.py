from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bends.bend_circular import bend_circular_all_angle
from gdsfactory.typings import ComponentAllAngleFactory, CrossSectionSpec


@gf.cell
def ring_double_bend_coupler(
    radius: float = 5.0,
    gap: float = 0.2,
    coupling_angle_coverage: float = 70.0,
    bend: ComponentAllAngleFactory = bend_circular_all_angle,
    length_x: float = 0.6,
    length_y: float = 0.6,
    cross_section_inner: CrossSectionSpec = "strip",
    cross_section_outer: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns ring with double curved couplers.

    Args:
        radius: um.
        gap: um.
        coupling_angle_coverage: degrees.
        bend: for bend.
        length_x: horizontal straight length.
        length_y: vertical straight length.
        cross_section_inner: spec inner bend.
        cross_section_outer: spec outer bend.
    """
    c = Component()

    c_halfring = gf.c.coupler_ring_bend(
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

    if c_halfring.dysize > half_height:
        raise ValueError(
            "The coupling_angle_coverage is too large for the given bend radius: "
            + "the coupling waveguides will overlap."
        )

    cb = c << c_halfring
    ct = c << c_halfring

    cross_section = cross_section_inner
    sy = gf.c.straight(length=length_y, cross_section=cross_section)
    sl = c << sy
    sr = c << sy

    sl.connect(port="o1", other=cb.ports["o2"])
    ct.connect(port="o3", other=sl.ports["o2"])
    sr.connect(port="o1", other=ct.ports["o2"])
    cb.connect(port="o3", other=sr.ports["o2"])

    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=ct.ports["o4"])
    c.add_port("o3", port=ct.ports["o1"])
    c.add_port("o4", port=cb.ports["o4"])
    c.flatten()
    return c


if __name__ == "__main__":
    # c = coupler_bend(radius=5)
    # c = coupler_ring_bend()
    c = ring_double_bend_coupler()
    c.show()
