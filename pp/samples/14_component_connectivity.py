"""Lets define the references from a component and then connect them together.
"""

from typing import Callable

import pp
from pp.component import Component
from pp.tech import TECH_SILICON_C, Tech


@pp.cell
def test_ring_single_bus(
    coupler90_factory: Callable = pp.c.coupler90,
    cpl_straight_factory: Callable = pp.c.coupler_straight,
    straight_factory: Callable = pp.c.waveguide,
    bend90_factory: Callable = pp.c.bend_circular,
    length_y: float = 2.0,
    length_x: float = 4.0,
    gap: float = 0.2,
    bend_radius: float = 5.0,
    tech: Tech = TECH_SILICON_C,
) -> Component:
    """single bus ring"""
    c = pp.Component()

    # define subcells
    coupler90 = pp.call_if_func(
        coupler90_factory, gap=gap, radius=bend_radius, tech=tech
    )
    waveguide_x = pp.call_if_func(straight_factory, length=length_x, tech=tech)
    waveguide_y = pp.call_if_func(straight_factory, length=length_y, tech=tech)
    bend = pp.call_if_func(bend90_factory, radius=bend_radius, tech=tech)
    coupler_straight = pp.call_if_func(
        cpl_straight_factory, gap=gap, length=length_x, tech=tech
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
    c = test_ring_single_bus(gap=0.15, length_x=0.2, length_y=0.13)
    c.show()
