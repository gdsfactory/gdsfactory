from typing import Optional

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.cross_section import CrossSectionFactory, strip
from pp.snap import assert_on_2nm_grid
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Layer


@cell
def coupler_ring(
    coupler90: ComponentFactory = coupler90,
    bend: Optional[ComponentFactory] = None,
    coupler: ComponentFactory = coupler_straight,
    length_x: float = 4.0,
    gap: float = 0.2,
    radius: float = 5.0,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
) -> Component:
    r"""Coupler for ring.

    Args:
        coupler90: straight waveguide coupled to a 90deg bend.
        coupler: two parallel coupled waveguides.
        length_x: length of the parallel coupled waveguides.
        gap: spacing between parallel coupled waveguides.
        wg_width: width of the waveguides
        radius: of the bends.
        tech: Technology

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

      c = pp.c.coupler_ring(length_x=20, radius=5.0, gap=0.3)
      c.plot()

    """
    bend = bend or bend_euler
    tech = tech or TECH_SILICON_C
    cross_section_factory = cross_section_factory or strip

    c = pp.Component()
    assert_on_2nm_grid(gap)

    # define subcells
    coupler90 = (
        coupler90(
            bend=bend,
            width=width,
            gap=gap,
            radius=radius,
            layer=layer,
            cross_section_factory=cross_section_factory,
            tech=tech,
        )
        if callable(coupler90)
        else coupler90
    )
    coupler_straight = (
        coupler(width=width, gap=gap, length=length_x, layer=layer, tech=tech)
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
    from pp.tech import TECH_METAL1

    c = coupler_ring(radius=5.0, gap=0.3, tech=TECH_METAL1)
    # c = coupler_ring(length_x=20, radius=5.0, gap=0.3)
    # print(c.get_settings())
    print(c.name)
    c.show()
