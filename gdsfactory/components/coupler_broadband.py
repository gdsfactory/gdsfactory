from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


@gf.cell
def coupler_broadband(
    w_sc: float = 0.5,  # width of waveguides in the symmetric coupler section
    gap_sc: float = 0.2,  # gap size between the waveguides in the symmetric coupler section
    w_top: float = 0.6,  # width of the top waveguide in the phase control section
    gap_pc: float = 0.3,  # gap size in the phase control section
    legnth_taper: float = 1.0,  # length of the tapers
    bend: ComponentFactory = bend_euler,
    coupler_straight: ComponentFactory = coupler_straight,
    length_coupler_straight: float = 12.4,  # optimal L_1 from the 3d fdtd analysis
    lenght_coupler_big_gap: float = 4.7,  # optimal L_2 from the 3d fdtd analysis
    cross_section: CrossSectionSpec = "strip",
    radius: float = 10.0,
) -> Component:
    """Returns broadband coupler component.

    https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/BroadbandDirectionalCoupler.html
    proposed in Zeqin Lu, Han Yun, Yun Wang, Zhitian Chen, Fan Zhang, Nicolas A. F. Jaeger, and Lukas Chrostowski,
    "Broadband silicon photonic directional coupler using asymmetric-waveguide based phase control,"
    Opt. Express 23, 3795-3808 (2015), DOI: 10.1364/OE.23.003795.

    Args:
        w_sc: width of waveguides in the symmetric coupler section.
        gap_sc: gap size between the waveguides in the symmetric coupler section.
        w_top: width of the top waveguide in the phase control section.
        gap_pc: gap size in the phase control section.
        legnth_taper: length of the tapers.
        bend: bend factory.
        coupler_straight: coupler_straight factory.
        length_coupler_straight: optimal L_1 from the 3d fdtd analysis.
        lenght_coupler_big_gap: optimal L_2 from the 3d fdtd analysis.
        cross_section: cross_section of the waveguides.
        radius: bend radius.
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    layer = gf.get_layer(xs.layer)

    L_t = legnth_taper
    c = Component()
    L_2 = lenght_coupler_big_gap
    L_1 = length_coupler_straight

    y_coupler = -w_sc + xs.width / 2 + gap_pc / 2

    coupler = coupler_straight(length=L_1, cross_section=cross_section, gap=gap_sc)
    coupler1 = c << coupler
    coupler1.xmin = -L_2 / 2 - L_t - L_1
    coupler1.y = y_coupler

    _bend = bend(radius=radius, cross_section=cross_section)
    bend_lt = c << _bend
    bend_lb = c << _bend
    bend_lt.mirror()

    bend_lb.connect("o1", coupler1.ports["o1"])
    bend_lt.connect("o1", coupler1.ports["o2"])

    # define vertices of the top waveguide
    vertices_top = [
        (L_2 / 2 + L_t + L_1, 0),
        (L_2 / 2 + L_t + L_1, w_sc),
        (L_2 / 2 + L_t, w_sc),
        (L_2 / 2, w_top),
        (-L_2 / 2, w_top),
        (-L_2 / 2 - L_t, w_sc),
        (-L_2 / 2 - L_t - L_1, w_sc),
        (-L_2 / 2 - L_t - L_1, 0),
    ]

    c.add_polygon(vertices_top, layer=layer)

    # define vertices of the bottom waveguide
    vertices_bot = [
        (L_2 / 2 + L_t + L_1, -gap_sc - w_sc),
        (L_2 / 2 + L_t + L_1, -gap_sc),
        (L_2 / 2 + L_t, -gap_sc),
        (L_2 / 2, -gap_pc),
        (-L_2 / 2, -gap_pc),
        (-L_2 / 2 - L_t, -gap_sc),
        (-L_2 / 2 - L_t - L_1, -gap_sc),
        (-L_2 / 2 - L_t - L_1, -gap_sc - w_sc),
    ]
    c.add_polygon(vertices_bot, layer=layer)

    coupler2 = c << coupler
    coupler2.xmax = L_2 / 2 + L_t + L_1
    coupler2.y = y_coupler

    _bend = bend(radius=radius, cross_section=cross_section)
    bend_rt = c << _bend
    bend_rb = c << _bend
    bend_rt.mirror()

    bend_rb.connect("o1", coupler2.ports["o3"])
    bend_rt.connect("o1", coupler2.ports["o4"])

    c.add_port("o1", port=bend_lb.ports["o2"])
    c.add_port("o2", port=bend_lt.ports["o2"])
    c.add_port("o3", port=bend_rt.ports["o2"])
    c.add_port("o4", port=bend_rb.ports["o2"])
    return c


if __name__ == "__main__":
    c = coupler_broadband(cross_section="strip")
    c.show(show_ports=True)
