from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.contact import contact_m1_m3 as contact_metal_function
from gdsfactory.components.contact import contact_slab_npp_m3
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import rib_heater_doped, strip_rib_tip
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, ComponentOrFactory, CrossSectionFactory


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
    taper: Optional[ComponentOrFactory] = taper_cross_section,
    with_taper1: bool = True,
    with_taper2: bool = True,
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    contact_gap: float = 0.0,
    width: float = 0.5,
    with_top_contact: bool = True,
    with_bot_contact: bool = True,
    **kwargs
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
        taper: optional taper function
        heater_width:
        heater_gap:
        contact_gap: from edge of contact to waveguide
        width: waveguide width on the ridge
        kwargs: cross_section settings

    .. code::

                              length
        |<--------------------------------------------->|
        |              length_section                   |
        |    <--------------------------->              |
        |  length_contact                               |
        |    <------->                             taper|
        |    _________                    _________     |
        |   |        |                    |        |    |
        |   | contact|____________________|        |    |
        |   |  size  |    heater width    |        |    |
        |  /|________|____________________|________|\   |
        | / |             heater_gap               | \  |
        |/  |______________________________________|  \ |
         \  |_______________width__________________|  /
          \ |                                      | /
           \|_____________heater_gap______________ |/
            |        |                    |        |
            |        |____heater_width____|        |
            |        |                    |        |
            |________|                    |________|

        taper         cross_section_heater



                                   |<------width------>|
                                    ____________________ heater_gap             slab_gap
             top_contact           |                   |<---------->| bot_contact   <-->
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
    cross_section_heater = gf.partial(
        cross_section_heater,
        heater_width=heater_width,
        heater_gap=heater_gap,
        width=width,
        **kwargs
    )

    if taper:
        taper = (
            taper(cross_section1=cross_section, cross_section2=cross_section_heater)
            if callable(taper)
            else taper
        )
        length -= taper.get_ports_xsize() * 2

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=snap_to_grid(length),
    )

    if taper:
        if with_taper1:
            taper1 = c << taper
            taper1.connect("o2", wg.ports["o1"])
            c.add_port("o1", port=taper1.ports["o1"])
        else:
            c.add_port("o1", port=wg.ports["o1"])

        if with_taper2:
            taper2 = c << taper
            taper2.mirror()
            taper2.connect("o2", wg.ports["o2"])
            c.add_port("o2", port=taper2.ports["o1"])

        else:
            c.add_port("o2", port=wg.ports["o2"])
    else:
        c.add_port("o2", port=wg.ports["o2"])
        c.add_port("o1", port=wg.ports["o1"])

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
            if with_top_contact:
                contact_top = c << contact(size=contact_size)
                contact_top.x = xi
                contact_top.ymin = +(heater_gap + width / 2 + contact_gap)

            if with_bot_contact:
                contact_bot = c << contact(size=contact_size)
                contact_bot.x = xi
                contact_bot.ymax = -(heater_gap + width / 2 + contact_gap)

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


def test_straight_heater_doped_rib_ports() -> Component:
    c = straight_heater_doped_rib(length=100.0)
    assert c.get_ports_xsize(port_type="optical") == 100.0, c.get_ports_xsize(
        port_type="optical"
    )
    return c


if __name__ == "__main__":
    # c = straight_heater_doped_rib(with_top_heater=False, with_top_contact=False)
    c = straight_heater_doped_rib(with_taper1=False)
    # c = straight_heater_doped_rib()
    c.show()
