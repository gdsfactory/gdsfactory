"""Straight Doped PIN waveguide."""
from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.contact import contact_slab_m3
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.cross_section import pin, pn
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_pin(
    length: float = 500.0,
    cross_section: CrossSectionFactory = pin,
    contact: ComponentFactory = contact_slab_m3,
    contact_width: float = 10.0,
    contact_spacing: float = 2,
    taper: Optional[ComponentFactory] = taper_strip_to_ridge,
    **kwargs,
) -> Component:
    """Returns PIN with contacts

    https://doi.org/10.1364/OE.26.029983

    500um length for PI phase shift
    https://ieeexplore.ieee.org/document/8268112

    to go beyond 2PI, you will need at least 1mm
    https://ieeexplore.ieee.org/document/8853396/

    Args:
        length: of the waveguide
        cross_section: for the waveguide
        contact: for the contacts
        contact_size:
        contact_spacing: spacing between contacts
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

    contact_length = length
    contact_top = c << contact(
        size=(contact_length, contact_width),
    )
    contact_bot = c << contact(
        size=(contact_length, contact_width),
    )

    contact_bot.xmin = wg.xmin
    contact_top.xmin = wg.xmin

    contact_top.ymin = +contact_spacing / 2
    contact_bot.ymax = -contact_spacing / 2

    c.add_ports(contact_bot.ports, prefix="bot_")
    c.add_ports(contact_top.ports, prefix="top_")
    return c


straight_pn = gf.partial(straight_pin, cross_section=pn)

if __name__ == "__main__":
    c = straight_pin(length=40)
    # print(c.ports.keys())
    c.show()
