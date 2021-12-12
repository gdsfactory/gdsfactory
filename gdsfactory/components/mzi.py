from functools import partial
from typing import Optional

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler import coupler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.routing import get_route
from gdsfactory.types import ComponentFactory, ComponentOrFactory, CrossSectionFactory


@cell
def mzi(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: Optional[float] = 0.1,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentFactory = straight_function,
    straight_y: Optional[ComponentFactory] = None,
    straight_x_top: Optional[ComponentFactory] = None,
    straight_x_bot: Optional[ComponentFactory] = None,
    splitter: ComponentOrFactory = mmi1x2,
    combiner: Optional[ComponentFactory] = None,
    with_splitter: bool = True,
    port_e1_splitter: str = "o2",
    port_e0_splitter: str = "o3",
    port_e1_combiner: str = "o2",
    port_e0_combiner: str = "o3",
    nbends: int = 2,
    cross_section: CrossSectionFactory = strip,
) -> Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length
        length_y: vertical length for both and top arms
        length_x: horizontal length. None uses to the straight_x_bot/top defaults
        bend: 90 degrees bend library
        straight: straight function
        straight_y: straight for length_y and delta_length
        straight_x_top: top straight for length_x
        straight_x_bot: bottom straight for length_x
        splitter: splitter function
        combiner: combiner function
        with_splitter: if False removes splitter
        port_e1_combiner: east top combiner port
        port_e0_splitter: east bot splitter port
        port_e1_splitter: east top splitter port
        port_e0_combiner: east bot combiner port
        nbends: from straight top/bot to combiner (at least 2)
        cross_section: for routing (sxtop/sxbot to combiner)

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
    straight = partial(straight, cross_section=cross_section)

    straight_x_top = straight_x_top or straight
    straight_x_bot = straight_x_bot or straight
    straight_y = straight_y or straight
    bend_factory = bend
    bend = bend_factory(cross_section=cross_section)

    c = Component()
    cp1 = splitter() if callable(splitter) else splitter
    cp2 = combiner() if combiner else cp1

    if with_splitter:
        cp1 = c << cp1

    cp2 = c << cp2
    b5 = c << bend
    b5.mirror()
    b5.connect("o1", cp1.ports[port_e0_splitter])

    syl = c << straight_y(
        length=delta_length / 2 + length_y,
    )
    syl.connect("o1", b5.ports["o2"])
    b6 = c << bend
    b6.connect("o1", syl.ports["o2"])

    straight_x_bot = straight_x_bot(length=length_x) if length_x else straight_x_bot()
    sxb = c << straight_x_bot
    sxb.connect("o1", b6.ports["o2"])

    b1 = c << bend
    b1.connect("o1", cp1.ports[port_e1_splitter])

    sy = c << straight_y(length=length_y)
    sy.connect("o1", b1.ports["o2"])

    b2 = c << bend
    b2.connect("o2", sy.ports["o2"])
    straight_x_top = straight_x_top(length=length_x) if length_x else straight_x_top()
    sxt = c << straight_x_top
    sxt.connect("o1", b2.ports["o1"])

    cp2.mirror()
    cp2.xmin = sxt.ports["o2"].x + bend.info["radius"] * nbends + 0.1

    route = get_route(
        sxt.ports["o2"],
        cp2.ports[port_e1_combiner],
        straight=straight,
        bend=bend_factory,
        cross_section=cross_section,
    )
    c.add(route.references)
    route = get_route(
        sxb.ports["o2"],
        cp2.ports[port_e0_combiner],
        straight=straight,
        bend=bend_factory,
        cross_section=cross_section,
    )
    c.add(route.references)

    if with_splitter:
        c.add_ports(cp1.get_ports_list(orientation=180), prefix="in")
    else:
        c.add_port("o1", port=b1.ports["o1"])
        c.add_port("o2", port=b5.ports["o1"])
    c.add_ports(cp2.get_ports_list(orientation=0), prefix="out")
    c.add_ports(sxt.get_ports_list(port_type="electrical"), prefix="top")
    c.add_ports(sxb.get_ports_list(port_type="electrical"), prefix="bot")
    c.auto_rename_ports()
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

    # delta_length = 116.8 / 2
    # print(delta_length)
    # c = mzi(delta_length=delta_length, with_splitter=False)
    # c.pprint_netlist()
    # mmi2x2 = gf.partial(gf.c.mmi2x2, width_mmi=5, gap_mmi=2)
    # c = mzi(delta_length=10, combiner=gf.c.mmi1x2, splitter=mmi2x2)

    # c = mzi1x2_2x2()
    # c = mzi_coupler(length_x=5)
    # c = mzi2x2()
    # c.show()

    import gdsfactory as gf

    c = mzi(
        delta_length=100,
        straight_x_top=gf.c.straight_heater_meander,
        # straight_x_bot=gf.c.straight_heater_meander,
        # straight_x_top=gf.c.straight_heater_metal,
        # straight_x_bot=gf.c.straight_heater_metal,
        # length_x=None,
        length_x=None,
        # length_y=1.8,
        with_splitter=False,
    )
    c.show()
    # c.show(show_subports=True)
    # c.pprint()
    # n = c.get_netlist()
    # c.plot()
    # print(c.settings)
