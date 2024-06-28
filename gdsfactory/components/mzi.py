from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mzi(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float | None = 0.1,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    straight_y: ComponentSpec | None = None,
    straight_x_top: ComponentSpec | None = None,
    straight_x_bot: ComponentSpec | None = None,
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec | None = None,
    with_splitter: bool = True,
    port_e1_splitter: str = "o2",
    port_e0_splitter: str = "o3",
    port_e1_combiner: str = "o2",
    port_e0_combiner: str = "o3",
    nbends: int = 2,
    cross_section: CrossSectionSpec = "strip",
    cross_section_x_top: CrossSectionSpec | None = None,
    cross_section_x_bot: CrossSectionSpec | None = None,
    mirror_bot: bool = False,
    add_optical_ports_arms: bool = False,
    min_length: float = 10e-3,
    auto_rename_ports: bool = True,
) -> Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length.
        length_y: vertical length for both and top arms.
        length_x: horizontal length. None uses to the straight_x_bot/top defaults.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        splitter: splitter function.
        combiner: combiner function.
        with_splitter: if False removes splitter.
        port_e1_splitter: east top splitter port.
        port_e0_splitter: east bot splitter port.
        port_e1_combiner: east top combiner port.
        port_e0_combiner: east bot combiner port.
        nbends: from straight top/bot to combiner (at least 2).
        cross_section: for routing (sxtop/sxbot to combiner).
        cross_section_x_top: optional top cross_section (defaults to cross_section).
        cross_section_x_bot: optional bottom cross_section (defaults to cross_section).
        mirror_bot: if true, mirrors the bottom arm.
        add_optical_ports_arms: add all other optical ports in the arms
            with top_ and bot_ prefix.
        min_length: minimum length for the straight.
        auto_rename_ports: if True, renames ports.

    .. code::

                       b2______b3
                      |  sxtop  |
              straight_y        |
                      |         |
                      b1        b4
            splitter==|         |==combiner
                      b5        b8
                      |         |
              straight_y        |
                      |         |
        delta_length/2          |
                      |         |
                     b6__sxbot__b7
                          Lx
    """
    combiner = combiner or splitter

    straight_x_top = straight_x_top or straight
    straight_x_bot = straight_x_bot or straight
    straight_y = straight_y or straight

    cross_section_x_bot = cross_section_x_bot or cross_section
    cross_section_x_top = cross_section_x_top or cross_section
    bend = gf.get_component(bend, cross_section=cross_section)

    c = Component()
    cp1 = gf.get_component(splitter)
    cp2 = gf.get_component(combiner) if combiner else cp1

    if with_splitter:
        cp1 = c << cp1
        cp1.name = "cp1"

    cp2 = c << cp2
    b5 = c << bend
    b5.connect("o1", cp1.ports[port_e0_splitter], mirror=True)

    syl = c << gf.get_component(
        straight_y, length=delta_length / 2 + length_y, cross_section=cross_section
    )
    syl.connect("o1", b5.ports["o2"])
    b6 = c << bend
    b6.connect("o1", syl.ports["o2"])

    straight_x_top = (
        gf.get_component(
            straight_x_top, length=length_x, cross_section=cross_section_x_top
        )
        if length_x
        else gf.get_component(straight_x_top)
    )
    sxt = c << straight_x_top

    length_x = length_x or abs(sxt.ports["o1"].dx - sxt.ports["o2"].dx)

    straight_x_bot = (
        gf.get_component(
            straight_x_bot, length=length_x, cross_section=cross_section_x_bot
        )
        if length_x
        else gf.get_component(straight_x_bot)
    )
    sxb = c << straight_x_bot
    sxb.connect("o1", b6.ports["o2"], mirror=mirror_bot)

    b1 = c << bend
    b1.connect("o1", cp1.ports[port_e1_splitter])

    sytl = c << gf.get_component(
        straight_y, length=length_y, cross_section=cross_section
    )
    sytl.connect("o1", b1.ports["o2"])

    b2 = c << bend
    b2.connect("o2", sytl.ports["o2"])
    sxt.connect("o1", b2.ports["o1"])
    cp2.mirror_x()
    cp2.dxmin = sxt.ports["o2"].dx + bend.info["radius"] * nbends + 2 * min_length

    gap_ports_combiner = cp1.ports[port_e0_splitter].dy - cp1.ports[port_e1_splitter].dy
    gap_ports_splitter = cp2.ports[port_e0_combiner].dy - cp2.ports[port_e1_combiner].dy
    delta_gap_ports = gap_ports_combiner - gap_ports_splitter

    # Top arm
    b3 = c << bend
    b3.connect("o2", sxt.ports["o2"])

    sytr = c << gf.get_component(
        straight_y, length=length_y - delta_gap_ports / 2, cross_section=cross_section
    )
    sytr.connect("o2", b3.ports["o1"])
    b4 = c << bend
    b4.connect("o1", sytr.ports["o1"])

    # Bot arm
    b7 = c << bend
    b7.connect("o1", sxb.ports["o2"])

    sybr = c << gf.get_component(
        straight_y,
        length=delta_length / 2 + length_y - delta_gap_ports / 2,
        cross_section=cross_section,
    )
    sybr.connect("o1", b7.ports["o2"])
    b8 = c << bend
    b8.connect("o2", sybr.ports["o2"])

    cp2.connect(port_e1_combiner, b4.ports["o2"])

    sytl.name = "sytl"
    syl.name = "syl"
    sxt.name = "sxt"
    sxb.name = "sxb"
    cp2.name = "cp2"

    if with_splitter:
        c.add_ports(cp1.ports.filter(orientation=180), prefix="in_")
    else:
        c.add_port("o1", port=b1.ports["o1"])
        c.add_port("o2", port=b5.ports["o1"])
    c.add_ports(cp2.ports.filter(orientation=0), prefix="ou_")
    c.add_ports(sxt.ports.filter(port_type="electrical"), prefix="top_")
    c.add_ports(sxb.ports.filter(port_type="electrical"), prefix="bot_")
    c.add_ports(sxt.ports.filter(port_type="placement"), prefix="top_")
    c.add_ports(sxb.ports.filter(port_type="placement"), prefix="bot_")

    # c.auto_rename_ports(port_type="optical", prefix="o")

    if add_optical_ports_arms:
        c.add_ports(sxt.ports.filter(port_type="optical"), prefix="top_")
        c.add_ports(sxb.ports.filter(port_type="optical"), prefix="bot_")

    if auto_rename_ports:
        c.auto_rename_ports()
    return c


mzi1x2 = partial(mzi, splitter="mmi1x2", combiner="mmi1x2")
mzi2x2_2x2 = partial(
    mzi,
    splitter="mmi2x2",
    combiner="mmi2x2",
    port_e1_splitter="o3",
    port_e0_splitter="o4",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
    length_x=None,
)

mzi1x2_2x2 = partial(
    mzi,
    combiner="mmi2x2",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
)

mzi_coupler = partial(
    mzi2x2_2x2,
    splitter="coupler",
    combiner="coupler",
)

mzi_pin = partial(
    mzi,
    straight_x_top="straight_pin",
    cross_section_x_top="pin",
    delta_length=0.0,
    length_x=100,
)

mzi_phase_shifter = partial(mzi, straight_x_top="straight_heater_metal", length_x=200)

mzi2x2_2x2_phase_shifter = partial(
    mzi2x2_2x2, straight_x_top="straight_heater_metal", length_x=200
)

mzi_phase_shifter_top_heater_metal = partial(
    mzi_phase_shifter, straight_x_top=straight_heater_metal
)

mzm = partial(
    mzi_phase_shifter, straight_x_top="straight_pin", straight_x_bot="straight_pin"
)

if __name__ == "__main__":
    c = mzi1x2(combiner=partial(gf.c.mmi1x2, gap_mmi=0))
    # c = mzi_coupler()
    # c = mzi_pin()
    # c = mzm()
    # from gdsfactory import get_generic_pdk

    # pdk = get_generic_pdk()
    # pdk.activate()

    # c = mzi(cross_section="strip")
    # c = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")
    # c.show( )

    # c = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")
    # c = gf.routing.add_fiber_array(c)
    # gdspath = c.write_gds(flatten_invalid_refs=True)
    # gf.show(gdspath)
    c.show()

    # c1.write_gds("a.gds")

    # c2 = gf.read.import_gds("a.gds")
    # c2 = c2.flatten()

    # c3 = gf.grid([c2, c1])
    # c3.show()
