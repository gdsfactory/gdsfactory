import gdsfactory as gf
from gdsfactory.components.via_stack import via_stack_heater
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Layer, Optional


@gf.cell
def straight_heater_meander(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: gf.types.CrossSectionFactory = gf.cross_section.strip,
    heater_width: float = 2.5,
    extension_length: float = 15.0,
    layer_heater: Optional[Layer] = LAYER.HEATER,
    radius: float = 5.0,
    via_stack: Optional[ComponentFactory] = via_stack_heater,
    port_orientation1: int = 180,
    port_orientation2: int = 0,
    taper_length: Optional[float] = 10.0,
):
    """Returns a meander based heater
    based on SungWon Chung, Makoto Nakai, and Hossein Hashemi,
    Low-power thermo-optic silicon modulator for large-scale photonic integrated systems
    Opt. Express 27, 13430-13459 (2019)
    https://www.osapublishing.org/oe/abstract.cfm?URI=oe-27-9-13430

    FIXME: only works for 3 rows.

    Args:
        length: total length of the optical path
        spacing: waveguide spacing (center to center)
        cross_section: for waveguide
        heater_width: for heater
        extension_length: of input and output optical ports
        layer_heater: for top heater, if None, it does not add a heater
        radius: for the meander bends
        via_stack: for the heater to contact metal
        port_orientation1:
        port_orientation2:
        taper_length: minimizes current concentrations from heater to contact
    """
    rows = 3
    c = gf.Component()
    p1 = gf.Port(midpoint=(0, 0), orientation=0)
    p2 = gf.Port(midpoint=(0, spacing), orientation=0)
    route = gf.routing.get_route(p1, p2, radius=radius)

    straight_length = gf.snap.snap_to_grid((length - (rows - 1) * route.length) / rows)
    straight = gf.c.straight(length=straight_length, cross_section=cross_section)
    straight_array = c << gf.c.array(
        straight, spacing=(0, spacing), columns=1, rows=rows
    )

    for row in range(1, rows, 2):
        route = gf.routing.get_route(
            straight_array.ports[f"o2_{row+1}_1"],
            straight_array.ports[f"o2_{row}_1"],
            radius=radius,
            cross_section=cross_section,
        )
        c.add(route.references)

        route = gf.routing.get_route(
            straight_array.ports[f"o1_{row+1}_1"],
            straight_array.ports[f"o1_{row+2}_1"],
            radius=radius,
            cross_section=cross_section,
        )
        c.add(route.references)

    straight1 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight2 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight1.connect("o2", straight_array.ports["o1_1_1"])
    straight2.connect("o1", straight_array.ports[f"o2_{rows}_1"])

    c.add_port("o1", port=straight1.ports["o1"])
    c.add_port("o2", port=straight2.ports["o2"])

    if layer_heater:
        heater_cross_section = gf.partial(
            gf.cross_section.cross_section, width=heater_width, layer=layer_heater
        )

        heater = c << gf.c.straight(
            length=straight_length + 2 * extension_length,
            cross_section=heater_cross_section,
        )
        heater.movex(-extension_length)
        heater.movey(spacing * (rows // 2))

    if layer_heater and via_stack:
        contactw = via_stack()
        contacte = via_stack()
        contact_west_midpoint = heater.size_info.cw - (contactw.xsize / 2, 0)
        contact_east_midpoint = heater.size_info.ce + (contacte.xsize / 2, 0)

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

        if taper_length:
            taper = gf.c.taper(
                cross_section=heater_cross_section,
                width1=contactw.ysize,
                width2=heater_width,
                length=taper_length,
            )
            taper1 = c << taper
            taper2 = c << taper
            taper1.connect("o1", contact_west.ports["e3"])
            taper2.connect("o1", contact_east.ports["e1"])
    return c


if __name__ == "__main__":
    # rows = 3
    # length = 300.0
    # spacing = 3

    # c = gf.Component()
    # p1 = gf.Port(midpoint=(0, 0), orientation=0)
    # p2 = gf.Port(midpoint=(0, spacing), orientation=0)
    # route = gf.routing.get_route(p1, p2)
    # straight_length = gf.snap.snap_to_grid((length - (rows - 1) * route.length) / rows)
    # straight_array = c << gf.c.array(spacing=(0, spacing), columns=1, rows=rows)

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

    c = straight_heater_meander()
    c.show()
