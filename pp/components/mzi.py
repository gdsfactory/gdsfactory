from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.port import rename_ports_by_orientation
from pp.tech import LIBRARY, Library
from pp.types import StrOrDict


@cell
def mzi(
    delta_length: float = 10.0,
    length_y: float = 0.1,
    length_x: float = 0.1,
    bend: StrOrDict = "bend_euler",
    straight: StrOrDict = "straight",
    straight_vertical: Optional[StrOrDict] = None,
    straight_delta_length: Optional[float] = None,
    straight_horizontal_top: Optional[StrOrDict] = None,
    straight_horizontal_bot: Optional[StrOrDict] = None,
    splitter: StrOrDict = "mmi1x2",
    combiner: Optional[StrOrDict] = None,
    with_splitter: bool = True,
    library: Library = LIBRARY,
    **kwargs,
) -> Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length
        length_y: vertical length for both and top arms
        length_x: horizontal length
        bend: 90 degrees bend library
        straight: straight function
        straight_horizontal_top: straight for length_x
        straight_horizontal_bot: straight for length_x
        straight_vertical: straight for length_y and delta_length
        splitter: splitter function
        combiner: combiner function
        with_splitter: if False removes splitter
        kwargs: cross_section settings
        library: library with components

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
    get = library.get_component
    bend = get(bend, **kwargs)
    splitter = get(splitter)

    c = Component()
    cp1 = splitter
    cp2 = get(combiner) if combiner else splitter

    straight_vertical = straight_vertical or straight
    straight_horizontal_top = straight_horizontal_top or straight
    straight_horizontal_bot = straight_horizontal_bot or straight
    straight_delta_length = straight_delta_length or straight
    b90 = bend
    l0 = get(straight, length=length_y, **kwargs)

    cp1 = rename_ports_by_orientation(cp1)
    cp2 = rename_ports_by_orientation(cp2)

    y1l = cp1.ports["E0"].y
    y1r = cp2.ports["E0"].y

    y2l = cp1.ports["E1"].y
    y2r = cp2.ports["E1"].y

    dl = abs(y2l - y1l)  # splitter ports y distance
    dr = abs(y2r - y1r)  # combiner ports y distance
    delta_length_combiner = dl - dr
    if delta_length_combiner + length_y < 0:
        raise ValueError(
            f"splitter and combiner port yoffset + length_y = {delta_length_combiner:.3f} < 0 "
            f"you can swap combiner and splitter or increase length_y by {-delta_length_combiner-length_y:.3f}"
        )

    l0r = get(straight_vertical, length=length_y + delta_length_combiner / 2, **kwargs)
    l1 = get(straight_delta_length, length=delta_length / 2, **kwargs)
    lxt = get(straight_horizontal_top, length=length_x, **kwargs)
    lxb = get(straight_horizontal_bot, length=length_x, **kwargs)

    cin = cp1.ref()
    cout = c << cp2

    # top arm
    blt = c << b90
    bltl = c << b90
    bltr = c << b90
    blmr = c << b90  # bend left medium right
    l0tl = c << l0
    lxtop = c << lxt
    l0tr = c << l0r

    blt.connect(port="W0", destination=cin.ports["E1"])
    l0tl.connect(port="W0", destination=blt.ports["N0"])
    bltl.connect(port="N0", destination=l0tl.ports["E0"])
    lxtop.connect(port="W0", destination=bltl.ports["W0"])
    bltr.connect(port="N0", destination=lxtop.ports["E0"])
    l0tr.connect(port="W0", destination=bltr.ports["W0"])
    blmr.connect(port="W0", destination=l0tr.ports["E0"])
    cout.connect(port="E0", destination=blmr.ports["N0"])

    # bot arm
    blb = c << b90
    l0bl = c << l0
    l1l = c << l1
    blbl = c << b90
    lxbot = c << lxb
    brbr = c << b90
    l1r = c << l1
    l0br = c << l0r
    blbmrb = c << b90  # bend left medium right bottom

    blb.connect(port="N0", destination=cin.ports["E0"])
    l0bl.connect(port="W0", destination=blb.ports["W0"])
    l1l.connect(port="W0", destination=l0bl.ports["E0"])
    blbl.connect(port="W0", destination=l1l.ports["E0"])
    lxbot.connect(port="W0", destination=blbl.ports["N0"])
    brbr.connect(port="W0", destination=lxbot.ports["E0"])

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

    # Add any non-optical ports from bottom and bottom arms
    for i, port in enumerate(lxbot.get_ports_list()):
        if port.port_type != "optical":
            c.add_port(name=f"DC_BOT_{i}", port=port)

    for i, port in enumerate(lxtop.get_ports_list()):
        if port.port_type != "optical":
            c.add_port(name=f"DC_TOP_{i}", port=port)

    rename_ports_by_orientation(c)

    # aliases
    # top arm
    c.aliases["blt"] = blt
    c.aliases["bltl"] = bltl
    c.aliases["bltr"] = bltr
    c.aliases["blmr"] = blmr
    c.aliases["l0tl"] = l0tl
    c.aliases["lxtop"] = lxtop
    c.aliases["l0tr"] = l0tr

    # bot arm
    c.aliases["blb"] = blb
    c.aliases["l0bl"] = l0bl
    c.aliases["l1l"] = l1l
    c.aliases["blbl"] = blbl
    c.aliases["lxbot"] = lxbot
    c.aliases["brbr"] = brbr
    c.aliases["l1r"] = l1r
    c.aliases["l0br"] = l0br
    c.aliases["blbmrb"] = blbmrb

    return c


if __name__ == "__main__":
    delta_length = 116.8 / 2
    # print(delta_length)

    # c = mzi(delta_length=delta_length, with_splitter=False)
    # c = mzi(delta_length=10)

    # c.pprint_netlist()

    # add_markers(c)
    # print(c.ports["E0"].midpoint[1])
    # c.plot_netlist()
    # print(c.ports.keys())
    # print(c.ports["E0"].midpoint)

    c = mzi(
        delta_length=20,
        waveguide="nitride",
        bend=dict(component="bend_euler", radius=50),
        splitter=dict(component="mmi1x2", waveguide="nitride"),
    )
    c.show()
    c.pprint()
    # c.plot()
    # print(c.get_settings())
