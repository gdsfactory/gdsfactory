"""
FIXME:

    this function works, see, straight_heater_doped_strip func
"""

from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.contact import contact_m1_m3 as contact_metal_function
from gdsfactory.components.contact import contact_slab_npp_m3
from gdsfactory.components.contact import contact_npp_m1
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import rib_heater_doped, strip_rib_tip
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_heater_doped_rib(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionFactory = strip_rib_tip,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    contact: Optional[ComponentFactory] = contact_slab_npp_m3,
    contact_metal: Optional[ComponentFactory] = contact_metal_function,
    contact_metal_size: Tuple[float, float] = (10.0, 10.0),
    contact_size: Tuple[float, float] = (10.0, 10.0),
    contact_yspacing: float = 2.0,
    taper: Optional[ComponentFactory] = taper_cross_section,
) -> Component:
    r"""Returns a doped thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        nsections: between contacts
        cross_section: for the input/output ports
        cross_section_heater: for the heater
        contact: function to connect the heated strip
        contact_metal: function to connect the metal area
        contact_metal_size:
        contact_size:
        contact_yspacing: spacing from waveguide to contact
        taper: optional taper function


    .. code::

                              length
          <-------------------------------------------->
                       length_section
             <--------------------------->
           length_contact
             <------->                             taper
             ______________________________________
           /|        |____________________|        |\
        __/ |contact_|                    |        | \___
        __  |        |                    |        |  ___cross_section
          \ | size   |____________________|        | /
           \|________|____________________|________|/

        taper         cross_section_heater



                                   |<------width------>|
                                    ____________________ heater_gap             slab_gap
                                   |                   |<---------->|               <-->
         ___ ______________________|                   |___________________________|___
        |   |            |               undoped Si                 |              |   |
        |   |layer_heater|               intrinsic region           |layer_heater  |   |
        |___|____________|__________________________________________|______________|___|
                                                                     <------------>
                                                                      heater_width
        <------------------------------------------------------------------------------>
                                       slab_width

    """
    c = Component()

    if taper:
        taper = taper()
        length -= taper.get_ports_xsize() * 2

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=snap_to_grid(length),
    )

    if taper:
        taper1 = c << taper
        taper2 = c << taper
        taper1.connect("o2", wg.ports["o1"])
        taper2.connect("o2", wg.ports["o2"])
        c.add_port("o1", port=taper1.ports["o1"])
        c.add_port("o2", port=taper2.ports["o1"])

    else:
        c.add_port("o1", port=wg.ports["o1"])
        c.add_port("o2", port=wg.ports["o2"])

    if contact_metal:
        contact_section = contact_metal(size=contact_metal_size)
    contacts = []
    length_contact = snap_to_grid(contact_size[1])
    length_section = snap_to_grid((length - length_contact) / nsections)
    x0 = contact_size[0] / 2
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i

        if contact_metal and contact:
            contact_center = c.add_ref(contact_section)
            contact_center.x = xi
            contact_ref = c << contact_section
            contact_ref.x = xi
            contact_ref.y = (
                +contact_metal_size[1] if i % 2 == 0 else -contact_metal_size[1]
            )
            contacts.append(contact_ref)

        if contact:
            contact_bot = c << contact(size=contact_size)
            contact_top = c << contact(size=contact_size)
            contact_top.x = xi
            contact_bot.x = xi
            contact_top.ymin = +contact_yspacing
            contact_bot.ymax = -contact_yspacing

    if contact_metal and contact:
        contact_length = length + contact_metal_size[0]
        contact_top = c << contact_metal(
            size=(contact_length, contact_metal_size[0]),
        )
        contact_bot = c << contact_metal(
            size=(contact_length, contact_metal_size[0]),
        )

        contact_bot.xmin = contacts[0].xmin
        contact_top.xmin = contacts[0].xmin

        contact_top.ymin = contacts[0].ymax
        contact_bot.ymax = contacts[1].ymin

        c.add_ports(contact_top.ports, prefix="top_")
        c.add_ports(contact_bot.ports, prefix="bot_")
    return c


straight_heater_doped_strip = gf.partial(
    straight_heater_doped_rib,
    cross_section_heater=gf.cross_section.strip_heater_doped,
    contact=contact_npp_m1,
)


if __name__ == "__main__":
    # c = straight_heater_doped_rib(length=80)
    c = straight_heater_doped_strip(length=80)
    c.show()
