import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_straight import (
    coupler_straight as coupler_straight_function,
)
from gdsfactory.components.coupler_symmetric import (
    coupler_symmetric as coupler_symmetric_function,
)
from gdsfactory.cross_section import strip
from gdsfactory.snap import assert_on_2nm_grid, snap_to_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    coupler_symmetric: ComponentFactory = coupler_symmetric_function,
    coupler_straight: ComponentFactory = coupler_straight_function,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights
        length: of coupling region
        coupler_symmetric
        coupler_straight
        dy: port to port vertical spacing
        dx: length of bend in x direction
        cross_section: factory
        kwargs: cross_section settings

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
    length = snap_to_grid(length)
    assert_on_2nm_grid(gap)
    c = Component()

    sbend = coupler_symmetric(
        gap=gap, dy=dy, dx=dx, cross_section=cross_section, **kwargs
    )

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight(
        length=length, gap=gap, cross_section=cross_section, **kwargs
    )
    sl.connect("o2", destination=cs.ports["o1"])
    sr.connect("o1", destination=cs.ports["o4"])

    c.add_port("o1", port=sl.ports["o3"])
    c.add_port("o2", port=sl.ports["o4"])
    c.add_port("o3", port=sr.ports["o3"])
    c.add_port("o4", port=sr.ports["o4"])

    c.absorb(sl)
    c.absorb(sr)
    c.absorb(cs)
    c.info.length = sbend.info.length
    c.info.min_bend_radius = sbend.info.min_bend_radius
    c.auto_rename_ports()
    return c


if __name__ == "__main__":

    # c = gf.Component()
    # cp1 = c << coupler(gap=0.2)
    # cp2 = c << coupler(gap=0.5)
    # cp1.ymin = 0
    # cp2.ymin = 0

    # layer = (2, 0)
    # c = coupler(gap=0.300, layer=layer)
    c = coupler(cross_section=gf.cross_section.rib)
    c.show(show_subports=True)
