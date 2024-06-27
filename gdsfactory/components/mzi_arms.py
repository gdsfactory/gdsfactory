from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mzi_arm import mzi_arm
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.typings import ComponentSpec


# @cell
def mzi_arms(
    delta_length: float = 10.0,
    length_y: float = 0.8,
    length_x: float = 0.1,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    straight_y: ComponentSpec | None = None,
    straight_x_top: ComponentSpec | None = None,
    straight_x_bot: ComponentSpec | None = None,
    splitter: ComponentSpec = mmi1x2,
    combiner: ComponentSpec | None = None,
    with_splitter: bool = True,
    **kwargs,
) -> Component:
    """MZI made with arms.

    This MZI code is slightly deprecated.
    You can find a more robust MZI in gf.components.mzi.

    Args:
        delta_length: Bottom arm vertical extra length.
        length_y: Vertical length for both and top arms.
        length_x: Horizontal length.
        bend: 90 degrees bend library.
        straight: Straight spec.
        straight_y: Straight for length_y and delta_length.
        straight_x_top: Top straight for length_x.
        straight_x_bot: Bottom straight for length_x.
        splitter: Splitter spec.
        combiner: Combiner spec.
        with_splitter: If False removes splitter.
        kwargs: Cross-section settings.

    .. code::

                   __Lx__
                  |      |
                  Ly     Lyr (not a parameter)
                  |      |
        splitter==|      |==combiner
                  |      |
                  Ly     Lyr (not a parameter)
                  |      |
                  | delta_length/2
                  |      |
                  |__Lx__|

            ____________           __________
            |          |          |
            |          |       ___|
        ____|          |____
            | splitter   d1     d2  combiner
        ____|           ____
            |          |       ____
            |          |          |
            |__________|          |__________
    """
    combiner = combiner or splitter
    straight_x_top = straight_x_top or straight
    straight_x_bot = straight_x_bot or straight
    straight_y = straight_y or straight

    c = Component()
    cp1 = gf.get_component(splitter)
    cp2 = gf.get_component(combiner)

    if with_splitter:
        cin = c << cp1
    cout = c << cp2

    ports_cp1 = gf.port.get_ports_list(cp1.ports, clockwise=False)
    ports_cp2 = gf.port.get_ports_list(cp2.ports, clockwise=False)

    n_ports_cp1 = len(ports_cp1)
    n_ports_cp2 = len(ports_cp2)

    port_e1_cp1 = ports_cp1[n_ports_cp1 - 2]
    port_e0_cp1 = ports_cp1[n_ports_cp1 - 1]

    port_e1_cp2 = ports_cp2[n_ports_cp2 - 2]
    port_e0_cp2 = ports_cp2[n_ports_cp2 - 1]
    gap_ports_splitter = port_e0_cp1.dy - port_e1_cp1.dy
    gap_ports_combiner = port_e0_cp2.dy - port_e1_cp2.dy
    delta_gap_ports = gap_ports_combiner - gap_ports_splitter
    length_y_right = length_y + delta_gap_ports / 2
    length_y_left = length_y

    _top_arm = mzi_arm(
        straight_x=straight_x_top,
        straight_y=straight_y,
        length_x=length_x,
        length_y_left=length_y_left,
        length_y_right=length_y_right,
        bend=bend,
        **kwargs,
    )
    top_arm = c << _top_arm
    bot_arm = c << mzi_arm(
        straight_x=straight_x_bot,
        straight_y=straight_y,
        length_x=length_x,
        length_y_left=length_y_left + delta_length / 2,
        length_y_right=length_y_right + delta_length / 2 + delta_gap_ports / 2,
        bend=bend,
        **kwargs,
    )

    top_arm.connect("o1", port_e1_cp1)
    bot_arm.connect("o1", port_e0_cp1, mirror=True)
    cout.connect(port_e1_cp2.name, bot_arm.ports["o2"])

    if with_splitter:
        c.add_ports(cin.ports.filter(orientation=180), prefix="in")
    else:
        c.add_port("o1", port=bot_arm.ports["o1"])
        c.add_port("o2", port=top_arm.ports["o1"])

    c.add_ports(cout.ports.filter(orientation=0), prefix="out")
    c.add_ports(top_arm.ports.filter(port_type="electrical"), prefix="top")
    c.add_ports(bot_arm.ports.filter(port_type="electrical"), prefix="bot")
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    from functools import partial

    c = mzi_arms(combiner=partial(gf.c.mmi1x2, gap_mmi=1))
    c.show()
