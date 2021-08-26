from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.via_stack import via_stack, via_stack_slab
from gdsfactory.cross_section import rib_heater_doped
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def straight_heater_doped(
    length: float = 320.0,
    nsections: int = 3,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    via_stack_slab: ComponentFactory = via_stack_slab,
    via_stack_metal: ComponentFactory = via_stack,
    via_stack_size: Tuple[float, float] = (10.0, 10.0),
    via_stack_spacing: float = 5,
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
        via_stack_size:
        via_stack_spacing: spacing between contacts
        port_orientation_top: for top contact
        port_orientation_bot: for bottom contact
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
        t1.connect("o2", wg.ports["o1"])
        t2.connect("o2", wg.ports["o2"])
        c.add_port("o1", port=t1.ports["o1"])
        c.add_port("o2", port=t2.ports["o1"])

    else:
        c.add_ports(wg.get_ports_list())

    length_section = length / nsections

    contact_section = via_stack_slab(size=via_stack_size)
    contacts = []
    for i in range(0, nsections + 1):
        xi = x0 + length_section * i
        contact_i_center = c.add_ref(contact_section)
        contact_i_center.x = xi

        contact_i = c << contact_section
        contact_i.x = xi
        contact_i.y = +via_stack_spacing if i % 2 == 0 else -via_stack_spacing
        contacts.append(contact_i)

    via_stack_length = length + via_stack_size[0]
    contact_top = c << via_stack_metal(
        size=(via_stack_length, via_stack_size[0]),
        port_orientation=port_orientation_top,
    )
    contact_bot = c << via_stack_metal(
        size=(via_stack_length, via_stack_size[0]),
        port_orientation=port_orientation_bot,
    )

    contact_bot.xmin = contacts[0].xmin
    contact_top.xmin = contacts[0].xmin

    contact_top.ymin = contacts[0].ymax
    contact_bot.ymax = contacts[1].ymin

    c.add_port("e1", port=contact_top.get_ports_list()[0])
    c.add_port("e2", port=contact_bot.get_ports_list()[0])
    return c


if __name__ == "__main__":
    c = straight_heater_doped()
    c.show()
