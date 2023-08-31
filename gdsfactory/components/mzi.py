from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.routing.get_route import get_route
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
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

    bend_spec = bend
    bend = gf.get_component(bend, cross_section=cross_section)

    c = Component()
    cp1 = gf.get_component(splitter)
    cp2 = gf.get_component(combiner) if combiner else cp1

    if with_splitter:
        cp1 = c << cp1

    cp2 = c << cp2
    b5 = c << bend
    b5.mirror()
    b5.connect("o1", cp1.ports[port_e0_splitter])

    syl = c << gf.get_component(
        straight_y, length=delta_length / 2 + length_y, cross_section=cross_section
    )
    syl.connect("o1", b5.ports["o2"])
    b6 = c << bend
    b6.connect("o1", syl.ports["o2"])

    straight_x_bot = (
        gf.get_component(
            straight_x_bot, length=length_x, cross_section=cross_section_x_bot
        )
        if length_x
        else gf.get_component(straight_x_bot)
    )
    sxb = c << straight_x_bot
    if mirror_bot:
        sxb.mirror()
    sxb.connect("o1", b6.ports["o2"])

    b1 = c << bend
    b1.connect("o1", cp1.ports[port_e1_splitter])

    sytl = c << gf.get_component(
        straight_y, length=length_y, cross_section=cross_section
    )
    sytl.connect("o1", b1.ports["o2"])

    b2 = c << bend
    b2.connect("o2", sytl.ports["o2"])
    straight_x_top = (
        gf.get_component(
            straight_x_top, length=length_x, cross_section=cross_section_x_top
        )
        if length_x
        else gf.get_component(straight_x_top)
    )
    sxt = c << straight_x_top
    sxt.connect("o1", b2.ports["o1"])

    cp2.mirror()
    xs = gf.get_cross_section(cross_section)
    cp2.xmin = sxt.ports["o2"].x + bend.info["radius"] * nbends + 2 * xs.min_length

    route = get_route(
        sxt.ports["o2"],
        cp2.ports[port_e1_combiner],
        straight=straight,
        bend=bend_spec,
        cross_section=cross_section,
        with_sbend=False,
    )
    c.add(route.references)
    route = get_route(
        sxb.ports["o2"],
        cp2.ports[port_e0_combiner],
        straight=straight,
        bend=bend_spec,
        cross_section=cross_section,
        with_sbend=False,
    )
    c.add(route.references)

    sytl.name = "sytl"
    syl.name = "syl"
    sxt.name = "sxt"
    sxb.name = "sxb"
    cp1.name = "cp1"
    cp2.name = "cp2"

    if with_splitter:
        c.add_ports(cp1.get_ports_list(orientation=180), prefix="in_")
    else:
        c.add_port("o1", port=b1.ports["o1"])
        c.add_port("o2", port=b5.ports["o1"])
    c.add_ports(cp2.get_ports_list(orientation=0), prefix="ou_")

    c.add_ports(sxt.get_ports_list(port_type="electrical"), prefix="top_")
    c.add_ports(sxb.get_ports_list(port_type="electrical"), prefix="bot_")
    c.add_ports(sxt.get_ports_list(port_type="placement"), prefix="top_")
    c.add_ports(sxb.get_ports_list(port_type="placement"), prefix="bot_")

    c.auto_rename_ports(port_type="optical", prefix="o")

    if add_optical_ports_arms:
        c.add_ports(sxt.get_ports_list(port_type="optical"), prefix="top_")
        c.add_ports(sxb.get_ports_list(port_type="optical"), prefix="bot_")

    return c


mzi1x2 = partial(mzi, splitter=mmi1x2, combiner=mmi1x2)
mzi2x2_2x2 = partial(
    mzi,
    splitter=mmi2x2,
    combiner=mmi2x2,
    port_e1_splitter="o3",
    port_e0_splitter="o4",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
)

mzi1x2_2x2 = partial(
    mzi,
    combiner=mmi2x2,
    port_e1_combiner="o3",
    port_e0_combiner="o4",
)

mzi_coupler = partial(
    mzi2x2_2x2,
    splitter=coupler,
    combiner=coupler,
)


if __name__ == "__main__":
    c = mzi()
    print(sorted([i.name for i in c.get_dependencies()]))
    # from gdsfactory import get_generic_pdk

    # pdk = get_generic_pdk()
    # pdk.activate()

    # c = mzi(cross_section="strip")
    # c = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")
    # c.show(show_ports=True)

    # c = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")
    c = gf.routing.add_fiber_array(c)
    # gdspath = c.write_gds(flatten_invalid_refs=True)
    # gf.show(gdspath)
    c.show(show_ports=True)

    # c1.write_gds("a.gds")

    # c2 = gf.read.import_gds("a.gds")
    # c2 = c2.flatten()

    # c3 = gf.grid([c2, c1])
    # c3.show(show_ports=False)
