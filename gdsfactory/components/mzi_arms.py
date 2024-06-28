from __future__ import annotations

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mzi_arm import mzi_arm
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.typings import ComponentSpec


@cell
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
    delta_yright: float = 0,
    **kwargs,
) -> Component:
    """Mzi made with arms.

    This MZI code is slightly deprecated.
    You can find a more robust mzi in gf.components.mzi

    Args:
        delta_length: bottom arm vertical extra length.
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight spec.
        straight_y: straight for length_y and delta_length.
        straight_x_top: top straight for length_x.
        straight_x_bot: bottom straight for length_x.
        splitter: splitter spec.
        combiner: combiner spec.
        with_splitter: if False removes splitter.
        delta_yright: extra length for right y-oriented waveguide.
        kwargs: cross_section settings.

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
    from gdsfactory.pdk import get_component

    combiner = combiner or splitter

    straight_x_top = straight_x_top or straight
    straight_x_bot = straight_x_bot or straight
    straight_y = straight_y or straight

    c = Component()
    cp1 = get_component(splitter)
    cp2 = get_component(combiner)

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

    y1t = port_e1_cp1.dy
    y1b = port_e0_cp1.dy

    y2t = port_e1_cp2.dy
    y2b = port_e0_cp2.dy

    d1 = abs(y1t - y1b)  # splitter ports distance
    d2 = abs(y2t - y2b)  # combiner ports distance

    delta_symm_half = -delta_yright / 2

    if d2 > d1:
        length_y_left = length_y + (d2 - d1) / 2
        length_y_right = length_y
    else:
        length_y_right = length_y + (d1 - d2) / 2
        length_y_left = length_y

    _top_arm = mzi_arm(
        straight_x=straight_x_top,
        straight_y=straight_y,
        length_x=length_x,
        length_y_left=length_y_left + delta_symm_half,
        length_y_right=length_y_right + delta_symm_half + delta_yright,
        bend=bend,
        **kwargs,
    )

    top_arm = c << _top_arm

    bot_arm = c << mzi_arm(
        straight_x=straight_x_bot,
        straight_y=straight_y,
        length_x=length_x,
        length_y_left=length_y_left + delta_length / 2,
        length_y_right=length_y_right + delta_length / 2,
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

    import gdsfactory as gf

    c = mzi_arms(splitter=partial(gf.c.mmi1x2, gap_mmi=1), combiner=gf.c.mmi1x2)
    c.show()
