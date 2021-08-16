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
        **waveguide_settings

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

    sl.connect(port=1, destination=cb.ports[2])
    ct.connect(port=3, destination=sl.ports[2])
    sr.connect(port=2, destination=ct.ports[2])
    c.add_port(1, port=cb.ports[1])
    c.add_port(2, port=cb.ports[4])
    c.add_port(3, port=ct.ports[1])
    c.add_port(4, port=ct.ports[4])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":

    c = ring_double(width=1, layer=(2, 0), length_y=3)
    c.show(show_subports=False)
