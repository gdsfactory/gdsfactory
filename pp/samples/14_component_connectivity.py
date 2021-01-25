"""Lets define the references from a component and then connect them together.
"""

from typing import Callable

import pp
from pp.component import Component


@pp.cell
def test_ring_single_bus(
    coupler90_factory: Callable = pp.c.coupler90,
    cpl_straight_factory: Callable = pp.c.coupler_straight,
    straight_factory: Callable = pp.c.waveguide,
    bend90_factory: Callable = pp.c.bend_circular,
    length_y: float = 2.0,
    length_x: float = 4.0,
    gap: float = 0.2,
    wg_width: float = 0.5,
    bend_radius: int = 5,
) -> Component:
    """ single bus ring
    """
    c = pp.Component()

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
    assert c
    return c


if __name__ == "__main__":
    c = test_ring_single_bus(wg_width=0.45, gap=0.15, length_x=0.2, length_y=0.13)
    pp.show(c)
    # pp.write_gds(c, "ring.gds")
