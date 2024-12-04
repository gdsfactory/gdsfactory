from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpec, Port


@gf.cell
def straight_heater_meander(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: CrossSectionSpec = "strip",
    heater_width: float = 2.5,
    extension_length: float = 15.0,
    layer_heater: LayerSpec = "HEATER",
    radius: float | None = None,
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: float | None = None,
    port_orientation2: float | None = None,
    heater_taper_length: float = 10.0,
    straight_widths: Floats = (0.8, 0.9, 0.8),
    taper_length: float = 10,
    n: int | None = None,
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
        layer_heater: for top heater, if None, it does not add a heater.
        radius: for the meander bends. Defaults to cross_section radius.
        via_stack: for the heater to via_stack metal.
        port_orientation1: in degrees. None adds all orientations.
        port_orientation2: in degrees. None adds all orientations.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        straight_widths: widths of the straight sections.
        taper_length: from the cross_section.
        n: number of straight sections.
    """
    if n and straight_widths:
        raise ValueError("n and straight_widths are mutually exclusive")

    rows = n or len(straight_widths)
    c = gf.Component()
    cross_section2 = cross_section

    straight_length = gf.snap.snap_to_grid(length / rows, grid_factor=2)
    ports: dict[str, Port] = {}

    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius

    assert radius is not None

    if n and not straight_widths:
        if n % 2 == 0:
            raise ValueError(f"n={n} should be odd")
        straight_widths = [x.width] * n

    ##############
    # Straights
    ##############

    for row, straight_width in enumerate(straight_widths):
        cross_section1 = gf.get_cross_section(cross_section, width=straight_width)
        _straight = gf.c.straight(
            length=straight_length - 2 * taper_length,
            cross_section=cross_section,
            width=straight_width,
        )

        taper = gf.c.taper_cross_section_linear(
            cross_section1=cross_section1,
            cross_section2=cross_section2,
            length=taper_length,
        )
        straight_with_tapers = gf.c.extend_ports(component=_straight, extension=taper)

        straight_ref = c << straight_with_tapers
        straight_ref.dy = row * spacing
        ports[f"o1_{row + 1}"] = straight_ref.ports["o1"]
        ports[f"o2_{row + 1}"] = straight_ref.ports["o2"]

    ##############
    # loopbacks
    ##############
    for row in range(1, rows, 2):
        extra_length = 3 * (rows - row - 1) / 2 * radius
        extra_straight1 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight1.connect("o1", ports[f"o1_{row + 1}"])
        extra_straight2 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight2.connect("o1", ports[f"o1_{row + 2}"])

        gf.routing.route_single(
            c,
            extra_straight2.ports["o2"],
            extra_straight1.ports["o2"],
            radius=radius,
            cross_section=cross_section,
        )

        extra_length = 3 * (row - 1) / 2 * radius
        extra_straight1 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight1.connect("o1", ports[f"o2_{row + 1}"])
        extra_straight2 = c << gf.c.straight(
            length=extra_length, cross_section=cross_section
        )
        extra_straight2.connect("o1", ports[f"o2_{row}"])

        gf.routing.route_single(
            c,
            extra_straight2.ports["o2"],
            extra_straight1.ports["o2"],
            radius=radius,
            cross_section=cross_section,
        )

    straight1 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight2 = c << gf.c.straight(length=extension_length, cross_section=cross_section)
    straight1.connect("o2", ports["o1_1"])
    straight2.connect("o1", ports[f"o2_{rows}"])

    c.add_port("o1", port=straight1.ports["o1"])
    c.add_port("o2", port=straight2.ports["o2"])

    heater: ComponentReference | None = None
    heater_cross_section: CrossSectionSpec | None = None
    if layer_heater:
        heater_cross_section = partial(
            gf.cross_section.cross_section, width=heater_width, layer=layer_heater
        )

        heater = c << gf.c.straight(
            length=straight_length,
            cross_section=heater_cross_section,
        )
        heater.dmovey(spacing * (rows // 2))

    if layer_heater and via_stack and heater:
        via = gf.get_component(via_stack)
        dx = via.dxsize / 2 + heater_taper_length or 0
        via_stack_west_center = (heater.dbbox().left - dx, 0)
        via_stack_east_center = (heater.dbbox().right + dx, 0)

        via_stack_west = c << via
        via_stack_east = c << via
        via_stack_west.dmove(via_stack_west_center)
        via_stack_east.dmove(via_stack_east_center)

        valid_orientations = {p.orientation for p in via.ports}
        p1 = via_stack_west.ports.filter(orientation=port_orientation1)
        p2 = via_stack_east.ports.filter(orientation=port_orientation2)
        c.add_ports(p1, prefix="l_")
        c.add_ports(p2, prefix="r_")

        if not p1:
            raise ValueError(
                f"No ports for port_orientation1 {port_orientation1} in {valid_orientations}"
            )
        if not p2:
            raise ValueError(
                f"No ports for port_orientation2 {port_orientation2} in {valid_orientations}"
            )

        if heater_taper_length and heater_cross_section:
            taper = gf.c.taper(
                cross_section=heater_cross_section,
                width1=via.ports["e1"].dwidth,
                width2=heater_width,
                length=heater_taper_length,
            )
            taper1 = c << taper
            taper2 = c << taper

            taper1.connect("o2", heater.ports["o1"])
            taper2.connect("o2", heater.ports["o2"])

            via_stack_west.connect(
                "e3",
                taper1.ports["o1"],
                allow_width_mismatch=True,
                allow_layer_mismatch=True,
                allow_type_mismatch=True,
            )
            via_stack_east.connect(
                "e1",
                taper2.ports["o1"],
                allow_width_mismatch=True,
                allow_layer_mismatch=True,
                allow_type_mismatch=True,
            )
    c.flatten()
    return c


if __name__ == "__main__":
    c = straight_heater_meander(
        # heater_taper_length=0,
        # straight_widths=(0.5,) * 7,
        # taper_length=10,
        # taper_length=10,
        # length=1000,
        # port_orientation1=0
        # cross_section=partial(gf.cross_section.strip, width=0.8),
    )
    c.show()
