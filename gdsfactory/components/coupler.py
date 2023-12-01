from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight import (
    coupler_straight as coupler_straight_function,
)
from gdsfactory.components.coupler_symmetric import (
    coupler_symmetric as coupler_symmetric_function,
)
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    coupler_symmetric: ComponentFactory = coupler_symmetric_function,
    coupler_straight: ComponentFactory = coupler_straight_function,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights in um.
        length: of coupling region in um.
        coupler_symmetric: spec for bend coupler.
        coupler_straight: spec for straight coupler.
        dy: port to port vertical spacing in um.
        dx: length of bend in x direction in um.
        cross_section: spec (CrossSection, string or dict).

    .. code::

               dx                                 dx
            |------|                           |------|
         o2 ________                           ______o3
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         o1                                          o4

                        coupler_straight  coupler_symmetric


    """
    length = gf.snap.snap_to_grid2x(length)
    gap = gf.snap.snap_to_grid2x(gap)
    c = Component()

    xs = gf.get_cross_section(cross_section)
    xs_no_pins = xs.copy(add_pins_function_name=None)

    sbend = coupler_symmetric(gap=gap, dy=dy, dx=dx, cross_section=xs_no_pins)

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight(length=length, gap=gap, cross_section=cross_section)
    sl.connect("o2", destination=cs.ports["o1"])
    sr.connect("o1", destination=cs.ports["o4"])

    c.add_port("o1", port=sl.ports["o3"])
    c.add_port("o2", port=sl.ports["o4"])
    c.add_port("o3", port=sr.ports["o3"])
    c.add_port("o4", port=sr.ports["o4"])

    c.absorb(sl)
    c.absorb(sr)
    c.absorb(cs)
    c.info["length"] = sbend.info["length"]
    c.info["min_bend_radius"] = sbend.info["min_bend_radius"]
    c.auto_rename_ports()

    x = gf.get_cross_section(cross_section)
    x.add_bbox(c)
    x.add_pins(c)
    return c


if __name__ == "__main__":
    c = coupler(gap=0.2, dx=7)
    c = gf.routing.add_fiber_array(c)
    c.show(show_ports=False)
