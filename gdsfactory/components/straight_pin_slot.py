"""Straight Doped PIN waveguide."""
from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.contact import contact_m1_m3
from gdsfactory.components.contact_slot import contact_slot_slab_m1
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.cross_section import pin, pn
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_pin_slot(
    length: float = 500.0,
    cross_section: CrossSectionFactory = pin,
    contact: Optional[ComponentFactory] = contact_m1_m3,
    contact_width: float = 10.0,
    contact_slab: Optional[ComponentFactory] = contact_slot_slab_m1,
    contact_slab_top: Optional[ComponentFactory] = None,
    contact_slab_bot: Optional[ComponentFactory] = None,
    contact_slab_width: Optional[float] = None,
    contact_spacing: float = 3.0,
    contact_slab_spacing: float = 2.0,
    taper: Optional[ComponentFactory] = taper_strip_to_ridge,
    **kwargs,
) -> Component:
    """Returns a PIN straight waveguide with slotted via

    https://doi.org/10.1364/OE.26.029983

    500um length for PI phase shift
    https://ieeexplore.ieee.org/document/8268112

    to go beyond 2PI, you will need at least 1mm
    https://ieeexplore.ieee.org/document/8853396/

    Args:
        length: of the waveguide
        cross_section: for the waveguide
        contact: for contacting the metal
        contact_width:
        contact_slab: function for the component contacting the slab
        contact_slab_top: Optional, defaults to contact_slab
        contact_slab_bot: Optional, defaults to contact_slab
        contact_slab_width: defaults to contact_width
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

    contact_slab_width = contact_slab_width or contact_width
    contact_slab_spacing = contact_slab_spacing or contact_spacing

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

    if contact:
        contact_top = c << contact(
            size=(contact_length, contact_width),
        )
        contact_bot = c << contact(
            size=(contact_length, contact_width),
        )

        contact_bot.x = wg.x
        contact_top.x = wg.x

        contact_top.ymin = +contact_spacing / 2
        contact_bot.ymax = -contact_spacing / 2
        c.add_ports(contact_bot.ports, prefix="bot_")
        c.add_ports(contact_top.ports, prefix="top_")

    contact_slab_top = contact_slab_top or contact_slab
    contact_slab_bot = contact_slab_bot or contact_slab

    if contact_slab_top:
        slot_top = c << contact_slab_top(
            size=(contact_length, contact_slab_width),
        )
        slot_top.x = wg.x
        slot_top.ymin = +contact_slab_spacing / 2

    if contact_slab_bot:
        slot_bot = c << contact_slab_bot(
            size=(contact_length, contact_slab_width),
        )
        slot_bot.x = wg.x
        slot_bot.ymax = -contact_slab_spacing / 2

    return c


straight_pn_slot = gf.partial(straight_pin_slot, cross_section=pn)

if __name__ == "__main__":
    c = straight_pin_slot(contact_width=4, contact_slab_width=3, length=50)
    c.show()
