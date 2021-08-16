import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.via_stack import via_stack_slab
from gdsfactory.cross_section import rib_heater_doped
from gdsfactory.types import ComponentFactory, CrossSectionFactory

via_stack_silicon_tall = gf.partial(via_stack_slab, height=20, width=5)


@cell
def straight_heater_doped(
    length: float = 320.0,
    length_section: float = 100.0,
    cross_section_heater: CrossSectionFactory = rib_heater_doped,
    via_stack: ComponentFactory = via_stack_silicon_tall,
    **kwargs,
) -> Component:
    """Returns a doped thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        length_section: between contacts
        cross_section_heater: for heated sections
        via_stack:
        kwargs: cross_section common settings
    """
    n = int(length / length_section)
    c = Component()

    wg = c << gf.c.straight(
        cross_section=cross_section_heater,
        length=length,
        **kwargs,
    )

    contact = via_stack()
    contact_west = c << contact
    contact_east = c << contact
    contact_west.connect(contact_west.get_ports_list()[0].name, wg.ports[1])
    contact_east.connect(contact_east.get_ports_list()[0].name, wg.ports[2])

    x0 = wg.get_ports_list()[0].x
    c.add_ports(wg.get_ports_list())

    if n > 1:
        for i in range(1, n):
            xi = x0 + length_section * i
            contact_i = c << contact
            contact_i.x = xi
            c.add_port(f"M{i}", port=contact_i.get_ports_list()[0])

    c.add_port("MW", port=contact_west.get_ports_list()[0])
    c.add_port("ME", port=contact_east.get_ports_list()[0])
    gf.port.auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = straight_heater_doped()
    c.show()
