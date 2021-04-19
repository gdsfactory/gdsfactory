from typing import Dict, Optional, Union

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.mmi1x2 import mmi1x2 as mmi1x2_function
from pp.components.straight import straight as straight_function
from pp.port import rename_ports_by_orientation
from pp.types import ComponentFactory, ComponentOrFactory


@cell
def mzi(
    delta_length: float = 10.0,
    length_y: float = 0.1,
    length_x: float = 0.1,
    bend_radius: float = 10.0,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentFactory = straight_function,
    straight_vertical: Optional[ComponentFactory] = None,
    straight_delta_length: Optional[ComponentFactory] = None,
    straight_horizontal: Optional[ComponentFactory] = None,
    splitter: ComponentFactory = mmi1x2_function,
    combiner: Optional[ComponentFactory] = None,
    with_splitter: bool = True,
    pins: bool = False,
    splitter_settings: Optional[Dict[str, Union[int, float]]] = None,
    combiner_settings: Optional[Dict[str, Union[int, float]]] = None,
) -> Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length
        length_y: vertical length for both and top arms
        length_x: horizontal length
        bend_radius: 10.0
        bend: 90 degrees bend factory
        straight: straight function
        straight_horizontal: straight for length_x
        straight_vertical: straight for length_y and delta_length
        splitter: splitter function
        combiner: combiner function
        with_splitter: if False removes splitter
        pins: add pins cell and child cells
        splitter_settings: settings dict for splitter function
        combiner_settings: settings dict for combiner function

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


    """
    L2 = length_x
    L0 = length_y
    DL = delta_length

    splitter_settings = splitter_settings or {}
    combiner_settings = combiner_settings or {}

    c = Component()
    cp1 = splitter(**splitter_settings)
    if combiner:
        cp2 = combiner(**combiner_settings)
    else:
        cp2 = cp1

    straight_vertical = straight_vertical or straight
    straight_horizontal = straight_horizontal or straight
    straight_delta_length = straight_delta_length or straight
    b90 = bend(radius=bend_radius) if callable(bend) else bend
    l0 = straight_vertical(length=L0)

    cp1 = rename_ports_by_orientation(cp1)
    cp2 = rename_ports_by_orientation(cp2)

    y1l = cp1.ports["E0"].y
    y1r = cp2.ports["E0"].y

    y2l = cp1.ports["E1"].y
    y2r = cp2.ports["E1"].y

    dl = abs(y2l - y1l)  # splitter ports distance
    dr = abs(y2r - y1r)  # cp2 ports distance
    delta_length_combiner = dl - dr
    assert delta_length_combiner + L0 > 0, (
        f"cp1 and cp2 port height offset delta_length ({delta_length_combiner}) +"
        f" length_y ({length_y}) >0"
    )

    l0r = straight_vertical(length=L0 + delta_length_combiner / 2)
    l1 = straight_delta_length(length=DL / 2)
    l2 = straight_horizontal(length=L2)

    cin = cp1.ref()
    cout = c << cp2

    # top arm
    blt = c << b90
    bltl = c << b90
    bltr = c << b90
    blmr = c << b90  # bend left medium right

    l0tl = c << l0
    l2t = c << l2
    l0tr = c << l0r

    blt.connect(port="W0", destination=cin.ports["E1"])
    l0tl.connect(port="W0", destination=blt.ports["N0"])
    bltl.connect(port="N0", destination=l0tl.ports["E0"])
    l2t.connect(port="W0", destination=bltl.ports["W0"])
    bltr.connect(port="N0", destination=l2t.ports["E0"])
    l0tr.connect(port="W0", destination=bltr.ports["W0"])
    blmr.connect(port="W0", destination=l0tr.ports["E0"])
    cout.connect(port="E0", destination=blmr.ports["N0"])

    # bot arm
    blb = c << b90
    l0bl = c << l0
    l1l = c << l1
    blbl = c << b90
    l2t = c << l2
    brbr = c << b90
    l1r = c << l1
    l0br = c << l0r
    blbmrb = c << b90  # bend left medium right bottom

    blb.connect(port="N0", destination=cin.ports["E0"])
    l0bl.connect(port="W0", destination=blb.ports["W0"])
    l1l.connect(port="W0", destination=l0bl.ports["E0"])
    blbl.connect(port="W0", destination=l1l.ports["E0"])
    l2t.connect(port="W0", destination=blbl.ports["N0"])
    brbr.connect(port="W0", destination=l2t.ports["E0"])

    l1r.connect(port="W0", destination=brbr.ports["N0"])
    l0br.connect(port="W0", destination=l1r.ports["E0"])
    blbmrb.connect(port="N0", destination=l0br.ports["E0"])
    blbmrb.connect(port="W0", destination=cout.ports["E1"])  # just for netlist

    # west ports
    if with_splitter:
        c.add(cin)
        for port_name, port in cin.ports.items():
            if port.angle == 180:
                c.add_port(name=port_name, port=port)
    else:
        c.add_port(name="W1", port=blt.ports["W0"])
        c.add_port(name="W0", port=blb.ports["N0"])

    # east ports
    for i, port in enumerate(cout.ports.values()):
        if port.angle == 0:
            c.add_port(name=f"E{i}", port=port)

    rename_ports_by_orientation(c)
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":
    delta_length = 116.8 / 2
    # print(delta_length)

    # c = mzi(delta_length=delta_length, with_splitter=False)
    c = mzi(delta_length=10)

    c = mzi(delta_length=20)

    # add_markers(c)
    # print(c.ports["E0"].midpoint[1])
    # c.plot_netlist()
    # print(c.ports.keys())
    # print(c.ports["E0"].midpoint)

    c.show()
    # c.plot()
    # print(c.get_settings())
