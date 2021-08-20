from typing import Dict, Optional, Union

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentFactory, ComponentOrFactory, Layer


@cell
def mzi(
    delta_length: float = 10.0,
    length_y: float = 0.8,
    length_x: float = 0.1,
    bend: ComponentOrFactory = bend_euler,
    straight: ComponentFactory = straight_function,
    straight_vertical: Optional[ComponentFactory] = None,
    straight_delta_length: Optional[ComponentFactory] = None,
    straight_horizontal_top: Optional[ComponentFactory] = None,
    straight_horizontal_bot: Optional[ComponentFactory] = None,
    splitter: ComponentOrFactory = mmi1x2,
    combiner: Optional[ComponentFactory] = None,
    with_splitter: bool = True,
    splitter_settings: Optional[Dict[str, Union[int, float]]] = None,
    combiner_settings: Optional[Dict[str, Union[int, float]]] = None,
    layer: Layer = (1, 0),
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
    combiner = combiner or splitter

    splitter_settings = splitter_settings or {}
    combiner_settings = combiner_settings or {}

    c = Component()
    cp1 = splitter(**splitter_settings, **kwargs) if callable(splitter) else splitter
    cp2 = combiner(**combiner_settings, **kwargs) if combiner else cp1

    straight_vertical = straight_vertical or straight
    straight_horizontal_top = straight_horizontal_top or straight
    straight_horizontal_bot = straight_horizontal_bot or straight
    straight_delta_length = straight_delta_length or straight
    b90 = bend(**kwargs) if callable(bend) else bend
    l0 = straight_vertical(length=length_y, **kwargs)

    y1l = cp1.ports["o2"].y
    y1r = cp2.ports["o2"].y

    cin = cp1.ref()
    cout = c << cp2

    c1map = cin.ports_layer
    c2map = cout.ports_layer

    cp1_e1_port_name = c1map[f"{layer[0]}_{layer[1]}_E1"]
    cp1_e0_port_name = c1map[f"{layer[0]}_{layer[1]}_E0"]

    cp2_E1_port_name = c2map[f"{layer[0]}_{layer[1]}_E1"]
    cp2_E0_port_name = c2map[f"{layer[0]}_{layer[1]}_E0"]

    y2l = cp1.ports[cp1_e1_port_name].y
    y2r = cp2.ports[cp2_E1_port_name].y

    dl = abs(y2l - y1l)  # splitter ports distance
    dr = abs(y2r - y1r)  # cp2 ports distance
    delta_length_combiner = dl - dr
    assert delta_length_combiner + length_y > 0, (
        f"cp1 and cp2 port height offset delta_length ({delta_length_combiner}) +"
        f" length_y ({length_y}) >0"
    )

    l0r = straight_vertical(length=length_y + delta_length_combiner / 2, **kwargs)
    l1 = straight_delta_length(length=delta_length / 2, **kwargs)
    lxt = straight_horizontal_top(length=length_x, **kwargs)
    lxb = straight_horizontal_bot(length=length_x, **kwargs)

    # top arm
    blt = c << b90
    bltl = c << b90
    bltr = c << b90
    blmr = c << b90  # bend left medium right
    l0tl = c << l0
    lxtop = c << lxt
    l0tr = c << l0r

    lxtop_map = lxtop.ports_layer
    lxtop_E0 = lxtop_map[f"{layer[0]}_{layer[1]}_E0"]
    lxtop_W0 = lxtop_map[f"{layer[0]}_{layer[1]}_W0"]

    blt.connect(port="o1", destination=cin.ports[cp1_e1_port_name])
    l0tl.connect(port="o1", destination=blt.ports["o2"])
    bltl.connect(port="o2", destination=l0tl.ports["o2"])
    lxtop.connect(port=lxtop_W0, destination=bltl.ports["o1"])
    bltr.connect(port="o2", destination=lxtop.ports[lxtop_E0])
    l0tr.connect(port="o1", destination=bltr.ports["o1"])
    blmr.connect(port="o1", destination=l0tr.ports["o2"])
    cout.connect(port=cp2_E0_port_name, destination=blmr.ports["o2"])

    # bot arm
    blb = c << b90
    l0bl = c << l0
    l1l = c << l1
    blbl = c << b90
    brbr = c << b90
    l1r = c << l1
    l0br = c << l0r
    blbmrb = c << b90  # bend left medium right bottom
    lxbot = c << lxb

    lxtop_map = lxtop.ports_layer
    lxbot_E0 = lxtop_map[f"{layer[0]}_{layer[1]}_E0"]
    lxbot_W0 = lxtop_map[f"{layer[0]}_{layer[1]}_W0"]

    blb.connect(port="o2", destination=cin.ports[cp1_e0_port_name])
    l0bl.connect(port="o1", destination=blb.ports["o1"])
    l1l.connect(port="o1", destination=l0bl.ports["o2"])
    blbl.connect(port="o1", destination=l1l.ports["o2"])
    lxbot.connect(port=lxbot_W0, destination=blbl.ports["o2"])
    brbr.connect(port="o1", destination=lxbot.ports[lxbot_E0])

    l1r.connect(port="o1", destination=brbr.ports["o2"])
    l0br.connect(port="o1", destination=l1r.ports["o2"])
    blbmrb.connect(port="o2", destination=l0br.ports["o2"])
    blbmrb.connect(
        port="o1", destination=cout.ports[cp2_E1_port_name]
    )  # just for netlist
    # l0br.connect('o2', blbmrb.ports['o2'])

    # west ports
    if with_splitter:
        c.add(cin)
        for port_name, port in cin.ports.items():
            if port.angle == 180:
                c.add_port(name=port_name, port=port)
    else:
        c.add_port(name="o2", port=blt.ports["o1"])
        c.add_port(name="o1", port=blb.ports["o2"])

    # east ports
    i0 = len(cp1.get_ports_list(orientation=180))
    for i, port in enumerate(cout.ports.values()):
        if port.angle == 0:
            c.add_port(name=f"o{i+i0}", port=port)

    # Add any non-optical ports from bottom and bottom arms

    c.add_ports(lxtop.get_ports_list(layers_excluded=(layer,)), prefix="etop_")
    c.add_ports(lxbot.get_ports_list(layers_excluded=(layer,)), prefix="ebot_")

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
    import gdsfactory as gf

    # delta_length = 116.8 / 2
    # print(delta_length)
    # c = mzi(delta_length=delta_length, with_splitter=False)
    # c.pprint_netlist()

    c = mzi(delta_length=10, combiner=gf.c.mmi2x2)
    c = mzi(
        delta_length=20,
        straight_horizontal_top=gf.c.straight_heater_metal,
        straight_horizontal_bot=gf.c.straight_heater_metal,
        length_x=50,
        length_y=1.8,
    )
    c.show(show_subports=False)
    # c.pprint()
    # n = c.get_netlist()
    # c.plot()
    # print(c.get_settings())
