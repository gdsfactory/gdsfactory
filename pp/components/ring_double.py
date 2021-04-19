from typing import Optional

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.coupler_ring import coupler_ring as coupler_ring_function
from pp.components.straight import straight as straight_function
from pp.config import call_if_func
from pp.snap import assert_on_2nm_grid
from pp.types import ComponentFactory, CrossSectionFactory


@cell
def ring_double(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    coupler_ring: ComponentFactory = coupler_ring_function,
    straight: ComponentFactory = straight_function,
    bend: Optional[ComponentFactory] = None,
    pins: bool = False,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Double bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical straights (wyl: left, wyr: right)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler: ring coupler function
        straight: straight function
        bend: bend function
        pins: add pins
        cross_section_factory: for straights
        **cross_section_settings

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
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
            cross_section_factory=cross_section_factory,
            **cross_section_settings
        )
        if callable(coupler_ring)
        else coupler_ring
    )
    straight_component = call_if_func(
        straight,
        length=length_y,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component)
    wl = c.add_ref(straight_component)
    wr = c.add_ref(straight_component)

    wl.connect(port="E0", destination=cb.ports["N0"])
    ct.connect(port="N1", destination=wl.ports["W0"])
    wr.connect(port="W0", destination=ct.ports["N0"])
    cb.connect(port="N1", destination=wr.ports["E0"])
    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    c.add_port("E1", port=ct.ports["W0"])
    c.add_port("W1", port=ct.ports["E0"])
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":

    c = ring_double(width=1)
    c.show()
