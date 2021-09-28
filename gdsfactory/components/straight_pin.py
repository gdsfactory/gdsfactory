"""Straight Doped PIN waveguide."""
from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.via_stack import via_stack_slab
from gdsfactory.cross_section import pin, pn
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_pin(
    length: float = 500.0,
    cross_section: CrossSectionFactory = pin,
    via_stack: ComponentFactory = via_stack_slab,
    via_stack_width: float = 10.0,
    via_stack_spacing: float = 2,
    port_orientation_top: int = 0,
    port_orientation_bot: int = 180,
    taper: Optional[ComponentFactory] = taper_strip_to_ridge,
    **kwargs,
) -> Component:
    """Returns PIN with contacts

    https://doi.org/10.1364/OE.26.029983

    500um length from
    https://ieeexplore.ieee.org/document/8268112

    Args:
        length: of the waveguide
        cross_section: for the waveguide
        via_stack: for the contacts
        via_stack_size:
        via_stack_spacing: spacing between contacts
        port_orientation_top: for top contact
        port_orientation_bot: for bottom contact
        taper: optional taper
        kwargs: cross_section settings

    """
    c = Component()
    if taper:
        taper = taper() if callable(taper) else taper
        length -= 2 * taper.get_ports_xsize()

    wg = c << gf.c.straight(
        cross_section=cross_section,
        length=length,
        **kwargs,
    )

    if taper:
        t1 = c << taper
        t2 = c << taper
        t1.connect("o2", wg.ports["o1"])
        t2.connect("o2", wg.ports["o2"])
        c.add_port("o1", port=t1.ports["o1"])
        c.add_port("o2", port=t2.ports["o1"])

    else:
        c.add_ports(wg.get_ports_list())

    via_stack_length = length
    contact_top = c << via_stack(
        size=(via_stack_length, via_stack_width),
    )
    contact_bot = c << via_stack(
        size=(via_stack_length, via_stack_width),
    )

    contact_bot.xmin = wg.xmin
    contact_top.xmin = wg.xmin

    contact_top.ymin = +via_stack_spacing / 2
    contact_bot.ymax = -via_stack_spacing / 2

    c.add_port(
        "e1", port=contact_top.get_ports_list(orientation=port_orientation_top)[0]
    )
    c.add_port(
        "e2", port=contact_bot.get_ports_list(orientation=port_orientation_bot)[0]
    )
    return c


straight_pn = gf.partial(straight_pin, cross_section=pn)

if __name__ == "__main__":
    c = straight_pin(length=40)
    # print(c.ports.keys())
    c.show()
