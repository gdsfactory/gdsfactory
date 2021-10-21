from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.contact import contact_m1_m3 as contact_metal_function
from gdsfactory.components.contact import contact_slab_npp_m3
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import rib_heater_doped, strip_rib_tip
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Layer


@gf.cell
def straight_heater_doped_rib(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionFactory = strip_rib_tip,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    contact_contact: Optional[ComponentFactory] = contact_slab_npp_m3,
    contact_metal: Optional[ComponentFactory] = contact_metal_function,
    contact_metal_size: Tuple[float, float] = (10.0, 10.0),
    contact_contact_size: Tuple[float, float] = (10.0, 10.0),
    contact_contact_yspacing: float = 2.0,
    taper: Optional[ComponentFactory] = taper_cross_section,
    taper_length: float = 10.0,
    **kwargs,
) -> Component:
    r"""Returns a doped thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        nsections: between contacts
        cross_section_heater: for the heater
        contact_contact: function to connect the heated strip
        contact_metal: function to connect the metal area
        contact_metal_size:
        contact_contact_yspacing: spacing from waveguide to contact
        kwargs: cross_section settings
        taper: optional taper
        taper_length:

    .. code::

                              length
          <-------------------------------------------->
                       length_section
             <--------------------------->
           length_contact
             <------->                             taper
             ______________________________________
           /|        |____________________|        |\
        __/ |viastack|                    |        | \___
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
        taper = taper(
            cross_section1=cross_section,
            cross_section2=cross_section_heater,
            length=taper_length,
            **kwargs,
        )
        length -= taper_length * 2

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=snap_to_grid(length),
        **kwargs,
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
    length_contact = snap_to_grid(contact_contact_size[1])
    length_section = snap_to_grid((length - length_contact) / nsections)
    x0 = contact_contact_size[0] / 2
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i

        if contact_metal and contact_contact:
            contact_center = c.add_ref(contact_section)
            contact_center.x = xi
            contact = c << contact_section
            contact.x = xi
            contact.y = +contact_metal_size[1] if i % 2 == 0 else -contact_metal_size[1]
            contacts.append(contact)

        if contact_contact:
            contact_bot = c << contact_contact(size=contact_contact_size)
            contact_top = c << contact_contact(size=contact_contact_size)
            contact_top.x = xi
            contact_bot.x = xi
            contact_top.ymin = +contact_contact_yspacing
            contact_bot.ymax = -contact_contact_yspacing

    if contact_metal and contact_contact:
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


def test_straight_heater_doped_rib_ports() -> Component:
    c = straight_heater_doped_rib(length=100.0)
    assert c.get_ports_xsize(port_type="optical") == 100.0, c.get_ports_xsize(
        port_type="optical"
    )
    return c


@gf.cell
def straight_heater_doped_rib_south(
    contact_metal_size: Tuple[float, float] = (10.0, 10.0),
    layer_metal: Layer = (49, 0),
    **kwargs,
):
    c = gf.Component()
    s = c << straight_heater_doped_rib()
    contact_metal_size = (10.0, 10.0)
    lbend = c << gf.c.L(
        width=contact_metal_size[0], size=(10, s.ysize), layer=layer_metal
    )
    lbend.connect("e2", s.ports["top_e3"])
    c.add_ports(s.ports)
    c.ports.pop("top_e3")
    c.add_port("e1", port=lbend.ports["e1"])
    return c


if __name__ == "__main__":
    # c = straight_heater_doped_rib(length=80, contact_metal=None, contact_contact=None)
    c = straight_heater_doped_rib(length=80)
    # c = test_straight_heater_doped_rib_ports()

    # c = gf.Component()
    # s = c << straight_heater_doped_rib()
    # contact_metal_size = (10.0, 10.0)
    # lbend = c << gf.c.L(width=contact_metal_size[0], size=(10, s.ysize))
    # lbend.connect("o2", s.ports["e1"])
    # c = straight_heater_doped_rib_south()
    c.show()
