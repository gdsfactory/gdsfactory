from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler90 import coupler90 as coupler90function
from gdsfactory.components.coupler_straight import (
    coupler_straight as coupler_straight_function,
)
from gdsfactory.cross_section import strip
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    coupler90: ComponentFactory = coupler90function,
    bend: Optional[ComponentFactory] = None,
    coupler_straight: ComponentFactory = coupler_straight_function,
    cross_section: CrossSectionFactory = strip,
    bend_cross_section: Optional[CrossSectionFactory] = None,
    **kwargs
) -> Component:
    r"""Coupler for ring.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        coupler90: straight coupled to a 90deg bend.
        bend: factory for bend
        coupler_straight: two parallel coupled straight waveguides.
        cross_section:
        kwargs: cross_section settings

    .. code::

           2             3
           |             |
            \           /
             \         /
           ---=========---
         1    length_x    4


    """
    bend = bend or bend_euler

    c = Component()
    assert_on_2nm_grid(gap)

    # define subcells
    coupler90_component = (
        coupler90(
            gap=gap,
            radius=radius,
            bend=bend,
            cross_section=cross_section,
            bend_cross_section=bend_cross_section,
            **kwargs
        )
        if callable(coupler90)
        else coupler90
    )
    coupler_straight_component = (
        coupler_straight(
            gap=gap, length=length_x, cross_section=cross_section, **kwargs
        )
        if callable(coupler_straight)
        else coupler_straight
    )

    # add references to subcells
    cbl = c << coupler90_component
    cbr = c << coupler90_component
    cs = c << coupler_straight_component

    # connect references
    y = coupler90_component.y
    cs.connect(port="o4", destination=cbr.ports["o1"])
    cbl.reflect(p1=(0, y), p2=(1, y))
    cbl.connect(port="o2", destination=cs.ports["o2"])

    c.absorb(cbl)
    c.absorb(cbr)
    c.absorb(cs)

    c.add_port("o1", port=cbl.ports["o3"])
    c.add_port("o2", port=cbl.ports["o4"])
    c.add_port("o3", port=cbr.ports["o3"])
    c.add_port("o4", port=cbr.ports["o4"])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":

    c = coupler_ring(width=1, layer=(2, 0))
    c.show(show_subports=True)
