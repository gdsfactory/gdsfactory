from __future__ import annotations

from typing import List

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Floats, LayerSpec, Optional
from gdsfactory.cross_section import Section
from gdsfactory.components.via_stack import via_stack
from gdsfactory.components.via import via

via_stack = gf.partial(
    via_stack,
    size=(1.5, 1.5),
    layers=("M1", "M2"),
    vias=(
        gf.partial(
            via,
            layer="VIAC",
            size=(0.1, 0.1),
            spacing=(0.2, 0.2),
            enclosure=0.1,
        ),
        gf.partial(
            via,
            layer="VIA1",
            size=(0.1, 0.1),
            spacing=(0.2, 0.2),
            enclosure=0.1,
        ),
    ),
)


@gf.cell
def straight_heater_meander_doped(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: gf.typings.CrossSectionSpec = "strip",
    heater_width: float = 1.5,
    extension_length: float = 15.0,
    layers_doping: List[LayerSpec] = ("P", "PP", "PPP"),
    radius: float = 5.0,
    via_stack: Optional[ComponentSpec] = via_stack,
    port_orientation1: int = 180,
    port_orientation2: int = 0,
    straight_widths: Floats = (0.8, 0.9, 0.8),
    taper_length: float = 10,
) -> Component:
    """Returns a meander based heater.

    based on SungWon Chung, Makoto Nakai, and Hossein Hashemi,
    Low-power thermo-optic silicon modulator for large-scale photonic integrated systems
    Opt. Express 27, 13430-13459 (2019)
    https://www.osapublishing.org/oe/abstract.cfm?URI=oe-27-9-13430

    Args:
        length: total length of the optical path.
        spacing: waveguide spacing (center to center).
        cross_section: for waveguide.
        heater_width: for heater.
        extension_length: of input and output optical ports.
        layers_doping: doping layers to be used for heater.
        radius: for the meander bends.
        via_stack: for the heater to via_stack metal.
        port_orientation1: in degrees.
        port_orientation2: in degrees.
        straight_width: width of the straight section.
        taper_length: from the cross_section.
    """
    rows = len(straight_widths)
    c = gf.Component()
    x = gf.get_cross_section(cross_section)
    p1 = gf.Port(
        name="p1",
        center=(0, 0),
        orientation=0,
        cross_section=x,
        layer=x.layer,
        width=x.width,
    )
    p2 = gf.Port(
        name="p2",
        center=(0, spacing),
        orientation=0,
        cross_section=x,
        layer=x.layer,
        width=x.width,
    )
    route = gf.routing.get_route(p1, p2, radius=radius)

    cross_section2 = cross_section

    straight_length = gf.snap.snap_to_grid(
        (length - (rows - 1) * route.length) / rows, nm=2
    )
    ports = {}

    """
    Straights
    """
    for row, straight_width in enumerate(straight_widths):
        cross_section1 = gf.get_cross_section(cross_section, width=straight_width)
        straight = gf.c.straight(
            length=straight_length - 2 * taper_length, cross_section=cross_section1
        )

        taper = gf.partial(
            gf.c.taper_cross_section_linear,
            cross_section1=cross_section1,
            cross_section2=cross_section2,
            length=taper_length,
        )

        straight_with_tapers = gf.c.extend_ports(straight, extension=taper)

        straight_ref = c << straight_with_tapers
        if row < len(straight_widths) // 2:
            straight_ref.y = row * spacing
        else:
            straight_ref.y = (row + 1) * spacing
        ports[f"o1_{row+1}"] = straight_ref.ports["o1"]
        ports[f"o2_{row+1}"] = straight_ref.ports["o2"]
    """
    Loopbacks
    """
    for row in range(1, rows, 2):
        extra_length = 3 * (rows - row - 1) / 2 * radius
        extra_straight1 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight1.connect("o1", ports[f"o1_{row+1}"])
        extra_straight2 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight2.connect("o1", ports[f"o1_{row+2}"])

        route = gf.routing.get_route(
            extra_straight1.ports["o2"],
            extra_straight2.ports["o2"],
            radius=radius,
            cross_section=cross_section,
        )
        c.add(route.references)

        extra_length = 3 * (row - 1) / 2 * radius
        extra_straight1 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight1.connect("o1", ports[f"o2_{row+1}"])
        extra_straight2 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight2.connect("o1", ports[f"o2_{row}"])

        route = gf.routing.get_route(
            extra_straight1.ports["o2"],
            extra_straight2.ports["o2"],
            radius=radius,
            cross_section=cross_section,
        )
        c.add(route.references)

    straight1 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight2 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight1.connect("o2", ports["o1_1"])
    straight2.connect("o1", ports[f"o2_{rows}"])

    c.add_port("o1", port=straight1.ports["o1"])
    c.add_port("o2", port=straight2.ports["o2"])

    if layers_doping:
        sectionlist = ()
        for doping_layer in layers_doping:
            sectionlist += (Section(layer=doping_layer, width=heater_width, offset=0),)
        heater_cross_section = gf.partial(
            gf.cross_section.cross_section,
            width=heater_width,
            layer="WG",
            sections=sectionlist,
        )

        heater = c << gf.c.straight(
            length=straight_length,
            cross_section=heater_cross_section,
        )
        heater.movey(spacing * (rows // 2))

    if layers_doping and via_stack:
        via_stacke = via_stackw = gf.get_component(via_stack)
        dx = via_stackw.get_ports_xsize() / 2 or 0
        via_stack_west_center = heater.size_info.cw + (dx, 0)
        via_stack_east_center = heater.size_info.ce - (dx, 0)

        via_stack_east_center = gf.snap.snap_to_grid(via_stack_east_center, nm=10)
        via_stack_west_center = gf.snap.snap_to_grid(via_stack_west_center, nm=10)

        via_stack_west = c << via_stackw
        via_stack_east = c << via_stacke
        via_stack_west.move(via_stack_west_center)
        via_stack_east.move(via_stack_east_center)
        c.add_port(
            "e1", port=via_stack_west.get_ports_list(orientation=port_orientation1)[0]
        )
        c.add_port(
            "e2", port=via_stack_east.get_ports_list(orientation=port_orientation2)[0]
        )
    return c


if __name__ == "__main__":
    # rows = 3
    # length = 300.0
    # spacing = 3

    # c = gf.Component()
    # p1 = gf.Port(center=(0, 0), orientation=0)
    # p2 = gf.Port(center=(0, spacing), orientation=0)
    # route = gf.routing.get_route(p1, p2)
    # straight_length = gf.snap.snap_to_grid((length - (rows - 1) * route.length) / rows)
    # straight_array = c << gf.components.array(spacing=(0, spacing), columns=1, rows=rows)

    # for row in range(1, rows, 2):
    #     route = gf.routing.get_route(
    #         straight_array.ports[f"o2_{row+1}_1"], straight_array.ports[f"o2_{row}_1"]
    #     )
    #     c.add(route.references)

    #     route = gf.routing.get_route(
    #         straight_array.ports[f"o1_{row+1}_1"], straight_array.ports[f"o1_{row+2}_1"]
    #     )
    #     c.add(route.references)

    # c.add_port("o1", port=straight_array.ports["o1_1_1"])
    # c.add_port("o2", port=straight_array.ports[f"o2_{rows}_1"])

    c = straight_heater_meander_doped(
        straight_widths=(0.5,) * 7,
        taper_length=10,
        # taper_length=10,
        length=1000,
        # cross_section=gf.partial(gf.cross_section.strip, width=0.8),
    )
    c.show(show_ports=True)
    # scene = c.to_3d()
    # scene.show()
