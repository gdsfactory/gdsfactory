from typing import Callable

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.components.waveguide import waveguide
from pp.drc import assert_on_2nm_grid
from pp.netlist_to_gds import netlist_to_component


@cell
def ring_single_bus_deprecated(**kwargs):
    """ Ring single bus

    use single_bus instead as this one can have snaping issues that create gaps between waveguides of the ring

    Args:
        bend_radius=5
        length_x=1
        length_y=1
        gap=0.2
        bend90_factory=bend_circular
        coupler90_factory=coupler90
        straight_factory=waveguide
        cpl_straight_factory=coupler_straight
        wg_width=0.5


    .. code::

         --length_x-
         |         |
         |       length_y
         |         |
       ---===gap===---

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring_single_bus(gap=0.2, length_x=10, length_y=5, bend_radius=5)
      pp.plotgds(c)

    """
    components, connections, ports_map = ring_single_bus_netlist(**kwargs)
    component = netlist_to_component(components, connections, ports_map)
    return component


def ring_single_bus_netlist(
    bend_radius=5,
    length_x=1,
    length_y=1,
    gap=0.2,
    bend90_factory=bend_circular,
    coupler90_factory=coupler90,
    straight_factory=waveguide,
    cpl_straight_factory=coupler_straight,
    wg_width=0.5,
):
    """
    .. code::

         BL--H1--BR
         |       |
         VL      VR
         |       |
        -CL==CS==CR-
    """
    assert_on_2nm_grid(gap)

    bend = bend90_factory(width=wg_width, radius=bend_radius)
    cpl_bend = coupler90_factory(bend_radius=bend_radius, width=wg_width, gap=gap)
    cpl_straight = cpl_straight_factory(length=length_x, gap=gap, width=wg_width)
    h1 = straight_factory(length=length_x, width=wg_width)
    v = straight_factory(length=length_y, width=wg_width)

    components = {
        "CL": (cpl_bend, "mirror_y"),
        "CR": (cpl_bend, "None"),
        "BR": (bend, "mirror_x"),
        "BL": (bend, "R180"),
        "CS": (cpl_straight, "None"),
        "H1": (h1, "None"),
        "VL": (v, "R90"),
        "VR": (v, "R90"),
    }

    connections = [
        ("CL", "W0", "CS", "W0"),
        ("CS", "E0", "CR", "W0"),
        ("CR", "N0", "VR", "W0"),
        ("VR", "E0", "BR", "N0"),
        ("BR", "W0", "H1", "E0"),
        ("H1", "W0", "BL", "W0"),
        ("BL", "N0", "VL", "E0"),
        ("VL", "W0", "CL", "N0"),
    ]

    ports_map = {"W0": ("CL", "E0"), "E0": ("CR", "E0")}

    return components, connections, ports_map


@pp.cell
def ring_single_bus(
    coupler90_factory: Callable = coupler90,
    cpl_straight_factory: Callable = coupler_straight,
    straight_factory: Callable = waveguide,
    bend90_factory: Callable = bend_circular,
    length_y: float = 2.0,
    length_x: float = 4.0,
    gap: float = 0.2,
    wg_width: float = 0.5,
    bend_radius: float = 5.0,
) -> Component:
    """Single bus ring.

    .. code::

         ctl--wx--ctr
          |       |
         wyl     wgr
          |       |
        -cbl==CS==cbr-
    """
    c = pp.Component()
    assert_on_2nm_grid(gap)

    # define subcells
    coupler90 = pp.call_if_func(
        coupler90_factory, gap=gap, width=wg_width, bend_radius=bend_radius
    )
    waveguide_x = pp.call_if_func(straight_factory, length=length_x, width=wg_width)
    waveguide_y = pp.call_if_func(straight_factory, length=length_y, width=wg_width)
    bend = pp.call_if_func(bend90_factory, width=wg_width, radius=bend_radius)
    coupler_straight = pp.call_if_func(
        cpl_straight_factory, gap=gap, length=length_x, width=wg_width
    )

    # add references to subcells
    cbl = c << coupler90
    cbr = c << coupler90
    cs = c << coupler_straight
    wyl = c << waveguide_y
    wyr = c << waveguide_y
    wx = c << waveguide_x
    btl = c << bend
    btr = c << bend

    # connect references
    wyr.connect(port="E0", destination=cbr.ports["N0"])
    cs.connect(port="E0", destination=cbr.ports["W0"])

    cbl.reflect(p1=(0, coupler90.y), p2=(1, coupler90.y))
    cbl.connect(port="W0", destination=cs.ports["W0"])
    wyl.connect(port="E0", destination=cbl.ports["N0"])

    btl.connect(port="N0", destination=wyl.ports["W0"])
    btr.connect(port="W0", destination=wyr.ports["W0"])
    wx.connect(port="W0", destination=btl.ports["W0"])

    c.add_port("W0", port=cbl.ports["E0"])
    c.add_port("E0", port=cbr.ports["E0"])
    return c


def _compare_rings():

    c = pp.Component()
    c1 = ring_single_bus_deprecated(
        wg_width=0.45, gap=0.15, length_x=0.2, length_y=0.13
    )
    c2 = ring_single_bus(wg_width=0.45, gap=0.15, length_x=0.2, length_y=0.13)

    r1 = c << c1
    r2 = c << c2
    r1.ymin = 0
    r2.ymin = 0
    r1.xmin = 0
    r2.xmin = 0
    pp.show(c)


if __name__ == "__main__":
    # c = ring_single_bus(bend_radius=5.0, length_x=0.2, length_y=0.13, gap=0.15, wg_width=0.45)
    c = ring_single_bus(bend_radius=5.0, gap=0.3, wg_width=0.45)
    c = ring_single_bus(gap=0.3, wg_width=0.45)
    print(c.get_settings())
    print(c.name)
    pp.show(c)
    # c = ring_single_bus(bend_radius=5.0, length_x=2, length_y=4, gap=0.2, wg_width=0.4)
    # _compare_rings()
