from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.components.via_stack import via_stack_metal as via_stack_metal_function
from gdsfactory.components.via_stack import via_stack_slab_npp
from gdsfactory.cross_section import rib_heater_doped, strip_rib_tip
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_heater_doped_rib(
    length: float = 320.0,
    nsections: int = 3,
    cross_section: CrossSectionFactory = strip_rib_tip,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    via_stack_contact: ComponentFactory = via_stack_slab_npp,
    via_stack_metal: ComponentFactory = via_stack_metal_function,
    via_stack_metal_size: Tuple[float, float] = (10.0, 10.0),
    via_stack_contact_size: Tuple[float, float] = (10.0, 10.0),
    via_stack_contact_yspacing: float = 2.0,
    port_orientation_top: int = 0,
    port_orientation_bot: int = 180,
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
        via_stack_contact: function to connect the heated strip
        via_stack_metal: function to connect the metal area
        via_stack_metal_size:
        via_stack_contact_yspacing: spacing from waveguide to contact
        port_orientation_top: for top contact
        port_orientation_bot: for bottom contact
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
                                      ____________________  heater_gap                slab_gap
                                     |                   |<----------->|               <-->
         ___ ________________________|                   |____________________________|___
        |   |            |                 undoped Si                  |              |   |
        |   |layer_heater|                 intrinsic region            |layer_heater  |   |
        |___|____________|_____________________________________________|______________|___|
                                                                        <------------>
                                                                         heater_width
        <--------------------------------------------------------------------------------->
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

    contact_section = via_stack_metal(size=via_stack_metal_size)
    contacts = []
    length_contact = snap_to_grid(via_stack_contact_size[1])
    length_section = snap_to_grid((length - length_contact) / nsections)
    x0 = via_stack_contact_size[0] / 2
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i
        contact_center = c.add_ref(contact_section)
        contact_center.x = xi

        contact = c << contact_section
        contact.x = xi
        contact.y = +via_stack_metal_size[1] if i % 2 == 0 else -via_stack_metal_size[1]
        contacts.append(contact)

        contact_bot = c << via_stack_contact(size=via_stack_contact_size)
        contact_top = c << via_stack_contact(size=via_stack_contact_size)
        contact_top.x = xi
        contact_bot.x = xi
        contact_top.ymin = +via_stack_contact_yspacing
        contact_bot.ymax = -via_stack_contact_yspacing

    via_stack_length = length + via_stack_metal_size[0]
    contact_top = c << via_stack_metal(
        size=(via_stack_length, via_stack_metal_size[0]),
    )
    contact_bot = c << via_stack_metal(
        size=(via_stack_length, via_stack_metal_size[0]),
    )

    contact_bot.xmin = contacts[0].xmin
    contact_top.xmin = contacts[0].xmin

    contact_top.ymin = contacts[0].ymax
    contact_bot.ymax = contacts[1].ymin

    c.add_port(
        "e1", port=contact_top.get_ports_list(orientation=port_orientation_top)[0]
    )
    c.add_port(
        "e2", port=contact_bot.get_ports_list(orientation=port_orientation_bot)[0]
    )
    return c


def test_straight_heater_doped_rib_ports() -> Component:
    c = straight_heater_doped_rib(length=100.0)
    assert c.get_ports_xsize(port_type="optical") == 100.0, c.get_ports_xsize(
        port_type="optical"
    )
    return c


if __name__ == "__main__":
    # c = straight_heater_doped(length=80)
    c = test_straight_heater_doped_rib_ports()
    c.show()
