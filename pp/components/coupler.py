from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.coupler_straight import coupler_straight
from pp.components.coupler_symmetric import coupler_symmetric
from pp.snap import assert_on_1nm_grid
from pp.types import ComponentFactory, CrossSectionFactory


@cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    coupler_symmetric_factory: ComponentFactory = coupler_symmetric,
    coupler_straight_factory: ComponentFactory = coupler_straight,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights
        length: of coupling region
        coupler_symmetric_factory
        coupler_straight_factory
        dy: port to port vertical spacing
        dx: length of bend in x direction
        cross_section_factory: for straight and bend
        **cross_section_settings

    .. code::

               dx                                 dx
            |------|                           |------|
         W1 ________                           _______E1
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         W0                                           E0

              coupler_straight_factory  coupler_symmetric_factory


    """
    assert_on_1nm_grid(length)
    assert_on_1nm_grid(gap)
    c = Component()

    sbend = coupler_symmetric_factory(
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight_factory(
        length=length,
        gap=gap,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )
    sl.connect("W1", destination=cs.ports["W0"])
    sr.connect("W0", destination=cs.ports["E0"])

    c.add_port("W1", port=sl.ports["E0"])
    c.add_port("W0", port=sl.ports["E1"])
    c.add_port("E0", port=sr.ports["E0"])
    c.add_port("E1", port=sr.ports["E1"])

    c.absorb(sl)
    c.absorb(sr)
    c.absorb(cs)
    c.length = sbend.length
    c.min_bend_radius = sbend.min_bend_radius
    return c


if __name__ == "__main__":

    # c = pp.Component()
    # cp1 = c << coupler(gap=0.2)
    # cp2 = c << coupler(gap=0.5)
    # cp1.ymin = 0
    # cp2.ymin = 0

    c = coupler(gap=0.2)
    # c = coupler(width=0.9, length=1, dy=2, gap=0.2)
    # print(c.settings_changed)
    c.show()
