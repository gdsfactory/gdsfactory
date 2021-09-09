from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.via_stack import via_stack_metal
from gdsfactory.cross_section import rib_heater_doped, rib_heater_doped_contact
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_heater_doped(
    length: float = 320.0,
    nsections: int = 3,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    cross_section_contact: CrossSectionFactory = rib_heater_doped_contact,
    via_stack: ComponentFactory = via_stack_metal,
    via_stack_size: Tuple[float, float] = (10.0, 10.0),
    port_orientation_top: int = 0,
    port_orientation_bot: int = 180,
    taper: ComponentFactory = taper_strip_to_ridge,
    inclusion_contact: float = 0.2,
    **kwargs,
) -> Component:
    r"""Returns a doped thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        nsections: between contacts
        cross_section_heater: for heated sections
        via_stack: for connecting the metal
        via_stack_size:
        port_orientation_top: for top contact
        port_orientation_bot: for bottom contact
        taper: optional taper
        kwargs: cross_section settings

    .. code::

                              length
         <-------------------------------------------->
                       length_section
             <-------------------------------->
           length_contact
             <------->
             ______________________________________
           /|        |____________________|        |
          / |viastack|                    |        |
          \ | size   |____________________|        |
           \|________|____________________|________|

        taper         cross_section_heater cross_section_contact
    """
    c = Component()

    taper = taper()
    length -= 2 * taper.get_ports_east_west_distance()

    length_contact = via_stack_size[1] - 2 * inclusion_contact
    length_section = (length - length_contact) / nsections
    length_heater = length_section - length_contact

    if length_heater < 0:
        raise ValueError("You need to increase the length")

    wg0 = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=inclusion_contact,
        **kwargs,
    )

    wg1 = c << gf.c.straight(
        cross_section=cross_section_contact,
        length=length_contact - inclusion_contact,
        **kwargs,
    )
    wg1.connect("o1", wg0.ports["o2"])
    x0 = wg0.get_ports_list()[0].x + via_stack_size[0] / 2

    for _ in range(nsections):
        wg2 = c << gf.c.straight(
            cross_section=cross_section_heater,
            length=length_heater,
            **kwargs,
        )
        wg2.connect("o1", wg1.ports["o2"])
        wg3 = c << gf.c.straight(
            cross_section=cross_section_contact,
            length=length_contact,
            **kwargs,
        )
        wg3.connect("o1", wg2.ports["o2"])
        wg1 = wg3

    contact_section = via_stack(size=via_stack_size)
    contacts = []
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i - inclusion_contact
        contact_i_center = c.add_ref(contact_section)
        contact_i_center.x = xi

        contact_i = c << contact_section
        contact_i.x = xi
        contact_i.y = +via_stack_size[1] if i % 2 == 0 else -via_stack_size[1]
        contacts.append(contact_i)

    t1 = c << taper
    t2 = c << taper
    t1.connect("o2", wg0.ports["o1"])
    t2.connect("o2", wg1.ports["o2"])
    c.add_port("o1", port=t1.ports["o1"])
    c.add_port("o2", port=t2.ports["o1"])

    via_stack_length = length + via_stack_size[0]
    contact_top = c << via_stack(
        size=(via_stack_length, via_stack_size[0]),
    )
    contact_bot = c << via_stack(
        size=(via_stack_length, via_stack_size[0]),
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


def test_straight_heater_doped_ports() -> Component:
    c = straight_heater_doped(length=100.0)
    assert (
        c.get_ports_east_west_distance(port_type="optical") == 100.0
    ), c.get_ports_east_west_distance(port_type="optical")
    return c


if __name__ == "__main__":
    c = test_straight_heater_doped_ports()
    # c = straight_heater_doped(length=80)
    c.show()
