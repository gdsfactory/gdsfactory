from typing import Callable, Optional
import pp

from pp.component import Component
from pp.port import deco_rename_ports, rename_ports_by_orientation
from pp.components import bend_circular
from pp.components import waveguide
from pp.components import mmi1x2


@deco_rename_ports
@pp.autoname
def mzi(
    L0: float = 1.0,
    DL: float = 0.1,
    L2: float = 20.1,
    bend_radius: float = 10.0,
    bend90: Callable = bend_circular,
    waveguide: Callable = waveguide,
    waveguide_vertical: Optional[Callable] = None,
    waveguide_horizontal: Optional[Callable] = None,
    coupler: Callable = mmi1x2,
    combiner: Optional[Callable] = None,
) -> Component:
    """Mzi

    Args:
        L0: vertical length for both and top arms
        DL: bottom arm extra length
        L2: L_top horizontal length
        bend_radius: 10.0
        bend90: bend_circular
        waveguide_vertical: waveguide
        coupler: coupler
        combiner: coupler

    .. code::

               __L2__
              |      |
              L0     L0r
              |      |
     coupler==|      |==combiner
              |      |
              L0     L0r
              |      |
             DL/2   DL/2
              |      |
              |__L2__|


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi(L0=0.1, DL=0, L2=10)
      pp.plotgds(c)

    """
    c = pp.Component()
    coupler = pp.call_if_func(coupler)
    if combiner:
        combiner = pp.call_if_func(combiner)
    else:
        combiner = coupler

    waveguide_vertical = waveguide_vertical or waveguide
    waveguide_horizontal = waveguide_horizontal or waveguide
    b90 = bend90(radius=bend_radius) if callable(bend90) else bend90
    l0 = waveguide_vertical(length=L0)

    coupler = rename_ports_by_orientation(coupler)
    combiner = rename_ports_by_orientation(combiner)

    y1l = coupler.ports["E0"].y
    y1r = combiner.ports["E0"].y

    y2l = coupler.ports["E1"].y
    y2r = combiner.ports["E1"].y

    dl = abs(y2l - y1l)  # splitter ports distance
    dr = abs(y2r - y1r)  # combiner ports distance
    delta_length_combiner = dl - dr
    assert delta_length_combiner + L0 > 0, (
        f"coupler and combiner port height offset (delta_length)  {delta_length_combiner} +"
        f" {L0} >0"
    )

    l0r = waveguide_vertical(length=L0 + delta_length_combiner / 2)
    l1 = waveguide_vertical(length=DL / 2)
    l2 = waveguide_horizontal(length=L2)

    cin = c << coupler
    cout = c << combiner

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
    for port_name, port in cin.ports.items():
        if port.angle == 180:
            c.add_port(name=port_name, port=port)

    # east ports
    i = 0
    for port_name, port in cout.ports.items():
        if port.angle == 0:
            c.add_port(name=f"E{i}", port=port)
            i += 1

    return c


if __name__ == "__main__":
    DL = 116.8 / 2
    # print(DL)
    c = mzi(DL=DL, pins=True)
    # print(c.ports["E0"].midpoint[1])
    # c.plot_netlist()
    print(c.ports.keys())
    print(c.ports["E0"].midpoint)
    pp.show(c)
    pp.qp(c)
    # print(c.get_settings())
