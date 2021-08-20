from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.config import call_if_func
from gdsfactory.cross_section import strip
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.010,
    coupler_ring: ComponentFactory = coupler_ring_function,
    straight: ComponentFactory = straight_function,
    bend: Optional[ComponentFactory] = None,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical straights (wl: left, wr: right)
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler_ring: ring coupler function
        straight: straight function
        bend: 90 degrees bend function
        cross_section:
        **kwargs: cross_section settings


    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    assert_on_2nm_grid(gap)

    coupler_ring_component = (
        coupler_ring(
            bend=bend,
            gap=gap,
            radius=radius,
            length_x=length_x,
            cross_section=cross_section,
            **kwargs
        )
        if callable(coupler_ring)
        else coupler_ring
    )
    straight_side = call_if_func(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )
    straight_top = call_if_func(
        straight, length=length_x, cross_section=cross_section, **kwargs
    )

    bend = bend or bend_euler
    bend_ref = (
        bend(radius=radius, cross_section=cross_section, **kwargs)
        if callable(bend)
        else bend
    )

    c = Component()
    cb = c << coupler_ring_component
    wl = c << straight_side
    wr = c << straight_side
    bl = c << bend_ref
    br = c << bend_ref
    wt = c << straight_top
    # wt.mirror(p1=(0, 0), p2=(1, 0))

    wl.connect(port="o2", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=wl.ports["o1"])

    wt.connect(port="o2", destination=bl.ports["o1"])
    br.connect(port="o2", destination=wt.ports["o1"])
    wr.connect(port="o1", destination=br.ports["o1"])
    wr.connect(port="o2", destination=cb.ports["N1"])  # just for netlist

    c.add_port("o2", port=cb.ports["o2"])
    c.add_port("o1", port=cb.ports["o1"])
    return c


if __name__ == "__main__":
    c = ring_single(width=2, gap=1, layer=(2, 0))
    print(c.ports)
    c.show()
