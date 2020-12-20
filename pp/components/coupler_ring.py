from typing import Callable

import pp
from pp.component import Component
from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.drc import assert_on_2nm_grid


@pp.cell
def coupler_ring(
    coupler90: Callable = coupler90,
    coupler: Callable = coupler_straight,
    length_x: float = 4.0,
    gap: float = 0.2,
    wg_width: float = 0.5,
    bend_radius: float = 5.0,
) -> Component:
    r"""Coupler for ring.

    .. code::

           N0            N1
           |             |
            \           /
             \         /
           ---=========---
        W0    length_x    E0

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_ring(length_x=20, bend_radius=5.0, gap=0.3, wg_width=0.45)
      pp.plotgds(c)

    """
    c = pp.Component()
    assert_on_2nm_grid(gap)

    # define subcells
    coupler90 = (
        coupler90(gap=gap, width=wg_width, bend_radius=bend_radius)
        if callable(coupler90)
        else coupler90
    )
    coupler_straight = (
        coupler(gap=gap, length=length_x, width=wg_width)
        if callable(coupler)
        else coupler
    )

    # add references to subcells
    cbl = c << coupler90
    cbr = c << coupler90
    cs = c << coupler_straight

    # connect references
    cs.connect(port="E0", destination=cbr.ports["W0"])
    cbl.reflect(p1=(0, coupler90.y), p2=(1, coupler90.y))
    cbl.connect(port="W0", destination=cs.ports["W0"])

    c.absorb(cbl)
    c.absorb(cbr)
    c.absorb(cs)

    c.add_port("W0", port=cbl.ports["E0"])
    c.add_port("N0", port=cbl.ports["N0"])
    c.add_port("E0", port=cbr.ports["E0"])
    c.add_port("N1", port=cbr.ports["N0"])
    return c


if __name__ == "__main__":
    c = coupler_ring(bend_radius=5.0, gap=0.3, wg_width=0.45)
    # c = coupler_ring(length_x=20, bend_radius=5.0, gap=0.3, wg_width=0.45)
    # print(c.get_settings())
    print(c.name)
    pp.show(c)
