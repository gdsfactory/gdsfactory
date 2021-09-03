from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.config import call_if_func
from gdsfactory.cross_section import strip
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def ring_double(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentFactory = coupler_ring_function,
    straight: ComponentFactory = straight_function,
    bend: Optional[ComponentFactory] = None,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """Double bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler: ring coupler function
        straight: straight function
        bend: bend function
        **kwargs: cross_section settings

    .. code::

         --==ct==--
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    assert_on_2nm_grid(gap)

    coupler_component = (
        coupler_ring(
            gap=gap,
            radius=radius,
            length_x=length_x,
            bend=bend,
            cross_section=cross_section,
            **kwargs
        )
        if callable(coupler_ring)
        else coupler_ring
    )
    straight_component = call_if_func(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component)
    sl = c.add_ref(straight_component)
    sr = c.add_ref(straight_component)

    sl.connect(port="o1", destination=cb.ports["o2"])
    ct.connect(port="o3", destination=sl.ports["o2"])
    sr.connect(port="o2", destination=ct.ports["o2"])
    c.add_port("o1", port=cb.ports["o1"])
    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o3", port=ct.ports["o4"])
    c.add_port("o4", port=ct.ports["o1"])
    return c


if __name__ == "__main__":

    c = ring_double(width=1, layer=(2, 0), length_y=3)
    c.show(show_subports=False)
