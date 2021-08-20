from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.via_stack import via_stack_slab
from gdsfactory.cross_section import rib_heater_doped
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@cell
def straight_heater_doped(
    length: float = 320.0,
    nsections: int = 3,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    via_stack: ComponentFactory = via_stack_slab,
    contact_size: Tuple[float, float] = [10.0, 10.0],
    contact_spacing: float = 5,
    port_orientation_top: int = 0,
    port_orientation_bot: int = 180,
    taper: Optional[ComponentFactory] = taper_strip_to_ridge,
    **kwargs,
) -> Component:
    """Returns a doped thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        nsections: between contacts
        cross_section_heater: for heated sections
        via_stack: for the contacts
        contact_size:
        contact_spacing: spacing between contacts
        port_orientation_bot: for bottom contact
        port_orientation_top: for top contact
        taper: optional taper
        kwargs: cross_section settings
    """
    c = Component()

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=length,
        **kwargs,
    )

    x0 = wg.get_ports_list()[0].x

    if taper:
        taper = taper() if callable(taper) else taper
        t1 = c << taper
        t2 = c << taper
        t1.connect(2, wg.ports[1])
        t2.connect(2, wg.ports[2])
        c.add_port(1, port=t1.ports[1])
        c.add_port(2, port=t2.ports[1])

    else:
        c.add_ports(wg.get_ports_list())

    length_section = length / nsections

    contact = via_stack(size=contact_size)
    contacts = []
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i
        contact_i_center = c.add_ref(contact)
        contact_i_center.x = xi

        contact_i = c << contact
        contact_i.x = xi
        contact_i.y = +contact_spacing if i % 2 == 0 else -contact_spacing
        contacts.append(contact_i)

    length_contact = length + contact_size[0]
    contact_top = c << via_stack(
        size=(length_contact, contact_size[0]),
        port_orientation=port_orientation_top,
    )
    contact_bot = c << via_stack(
        size=(length_contact, contact_size[0]),
        port_orientation=port_orientation_bot,
    )

    contact_bot.xmin = contacts[0].xmin
    contact_top.xmin = contacts[0].xmin

    contact_top.ymin = contacts[0].ymax
    contact_bot.ymax = contacts[1].ymin

    c.add_port("eW", port=contact_top.get_ports_list()[0])
    c.add_port("eE", port=contact_bot.get_ports_list()[0])
    return c


if __name__ == "__main__":
    c = straight_heater_doped()
    c.show()
