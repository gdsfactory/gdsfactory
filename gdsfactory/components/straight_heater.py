from typing import Optional

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.via_stack import via_stack_heater
from gdsfactory.cross_section import strip_heater_metal, strip_heater_metal_undercut
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@cell
def straight_heater_metal_undercut(
    length: float = 320.0,
    length_undercut_spacing: float = 6.0,
    length_undercut: float = 30.0,
    length_straight_input: float = 15.0,
    cross_section_heater: CrossSectionFactory = strip_heater_metal,
    cross_section_heater_undercut: CrossSectionFactory = strip_heater_metal_undercut,
    with_undercut: bool = True,
    via_stack: Optional[ComponentFactory] = via_stack_heater,
    port_orientation1: int = 180,
    port_orientation2: int = 0,
    **kwargs,
) -> Component:
    """Returns a thermal phase shifter.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide
        length_undercut_spacing: from undercut regions
        length_straight_input: from input port to where trenches start
        cross_section_heater: for heated sections
        cross_section_heater_undercut: for heated sections with undercut
        with_undercut: isolation trenches for higher efficiency
        via_stack: via stack
        port_orientation1: left via stack port orientation
        port_orientation2: right via stack port orientation
        kwargs: cross_section common settings
    """
    period = length_undercut + length_undercut_spacing
    n = int((length - 2 * length_straight_input) // period)

    length_straight_input = (length - n * period) / 2

    s_si = gf.c.straight(
        cross_section=cross_section_heater,
        length=length_straight_input,
        **kwargs,
    )
    cross_section_undercut = (
        cross_section_heater_undercut if with_undercut else cross_section_heater
    )
    s_uc = gf.c.straight(
        cross_section=cross_section_undercut, length=length_undercut, **kwargs
    )
    s_spacing = gf.c.straight(
        cross_section=cross_section_heater, length=length_undercut_spacing, **kwargs
    )
    symbol_to_component = {
        "-": (s_si, "o1", "o2"),
        "U": (s_uc, "o1", "o2"),
        "H": (s_spacing, "o1", "o2"),
    }

    # Each character in the sequence represents a component
    sequence = "-" + n * "UH" + "-"

    c = Component()
    sequence = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    c.add_ref(sequence)
    c.add_ports(sequence.ports)

    if via_stack:
        contactw = via_stack()
        contacte = via_stack()
        contact_west_midpoint = sequence.aliases["-1"].size_info.cw
        contact_east_midpoint = sequence.aliases["-2"].size_info.ce

        contact_west = c << contactw
        contact_east = c << contacte
        contact_west.move(contact_west_midpoint)
        contact_east.move(contact_east_midpoint)
        c.add_port(
            "e1", port=contact_west.get_ports_list(orientation=port_orientation1)[0]
        )
        c.add_port(
            "e2", port=contact_east.get_ports_list(orientation=port_orientation2)[0]
        )
    return c


straight_heater_metal = gf.partial(
    straight_heater_metal_undercut,
    with_undercut=False,
)
straight_heater_metal_90_90 = gf.partial(
    straight_heater_metal_undercut,
    with_undercut=False,
    port_orientation1=90,
    port_orientation2=90,
)
straight_heater_metal_undercut_90_90 = gf.partial(
    straight_heater_metal_undercut,
    with_undercut=False,
    port_orientation1=90,
    port_orientation2=90,
)


def test_ports() -> Component:
    c = straight_heater_metal(length=50.0)
    assert c.ports["o2"].midpoint[0] == 50.0, c.ports["o2"].midpoint[0]
    return c


if __name__ == "__main__":
    # c = test_ports()
    # c = straight_heater_metal_undercut()
    # print(c.ports['o2'].midpoint[0])
    # c.pprint_ports
    c = straight_heater_metal(length=50.0)
    c.show()
    scene = gf.to_trimesh(c, layer_set=gf.layers.LAYER_SET)
    scene.show()
