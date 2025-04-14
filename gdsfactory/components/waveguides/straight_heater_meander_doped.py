"""Straight heater meander doped."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.components.vias.via import via
from gdsfactory.components.vias.via_stack import via_stack
from gdsfactory.cross_section import Section
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Floats, LayerSpecs, Port

_via_stack = partial(
    via_stack,
    size=(1.5, 1.5),
    layers=("M1", "M2"),
    vias=(
        partial(
            via,
            layer="VIAC",
            size=(0.1, 0.1),
            pitch=0.2,
            enclosure=0.1,
        ),
        partial(
            via,
            layer="VIA1",
            size=(0.1, 0.1),
            pitch=0.2,
            enclosure=0.1,
        ),
    ),
)


@gf.cell_with_module_name
def straight_heater_meander_doped(
    length: float = 300.0,
    spacing: float = 2.0,
    cross_section: CrossSectionSpec = "strip",
    heater_width: float = 1.5,
    extension_length: float = 15.0,
    layers_doping: LayerSpecs = ("P", "PP", "PPP"),
    radius: float = 5.0,
    via_stack: ComponentSpec | None = _via_stack,
    port_orientation1: float | None = None,
    port_orientation2: float | None = None,
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
        port_orientation1: in degrees. None adds all orientations.
        port_orientation2: in degrees. None adds all orientations.
        straight_widths: width of the straight sections.
        taper_length: from the cross_section.
    """
    from gdsfactory.pdk import get_layer

    rows = len(straight_widths)
    c = gf.Component()
    x = gf.get_cross_section(cross_section)
    layer = get_layer(x.layer)

    temp_component = Component()
    p1 = temp_component.add_port(
        name="p1",
        center=(0, 0),
        orientation=0,
        cross_section=x,
        layer=layer,
        width=x.width,
    )
    p2 = temp_component.add_port(
        name="p2",
        center=(0, spacing),
        orientation=0,
        cross_section=x,
        layer=layer,
        width=x.width,
    )

    dummy = gf.Component()
    route = gf.routing.route_single(
        dummy, p1, p2, radius=radius, cross_section=cross_section
    )
    cross_section2 = cross_section

    straight_length = gf.snap.snap_to_grid2x(
        (length - (rows - 1) * c.kcl.dbu * route.length) / rows,
    )
    ports: dict[str, Port] = {}

    if straight_length - 2 * taper_length <= 0:
        raise ValueError("straight_length - 2 * taper_length <= 0")

    # Straights
    for row, straight_width in enumerate(straight_widths):
        cross_section1 = gf.get_cross_section(cross_section, width=straight_width)
        straight = gf.c.straight(
            length=straight_length - 2 * taper_length, cross_section=cross_section1
        )

        taper = partial(
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
        ports[f"o1_{row + 1}"] = straight_ref["o1"]
        ports[f"o2_{row + 1}"] = straight_ref["o2"]

    # Loopbacks
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

        route = gf.routing.route_single(
            c,
            extra_straight2["o2"],
            extra_straight1["o2"],
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

        route = gf.routing.route_single(
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

    if layers_doping:
        sections: tuple[Section, ...] = ()
        for doping_layer in layers_doping:
            sections += (Section(layer=doping_layer, width=heater_width, offset=0),)
        heater_cross_section = partial(
            gf.cross_section.cross_section,
            width=heater_width,
            layer="WG",
            sections=sections,
            port_names=("e1", "e2"),
            port_types=("electrical", "electrical"),
        )

        heater = c << gf.c.straight(
            length=straight_length,
            cross_section=heater_cross_section,
        )
        heater.movey(spacing * (rows // 2))

    if layers_doping and via_stack and heater is not None:
        via = via_stacke = via_stackw = gf.get_component(via_stack)
        via_stack_west = c << via_stackw
        via_stack_east = c << via_stacke
        via_stack_west.connect(
            "e3", heater["e1"], allow_layer_mismatch=True, allow_width_mismatch=True
        )
        via_stack_east.connect(
            "e1", heater["e2"], allow_layer_mismatch=True, allow_width_mismatch=True
        )

        valid_orientations = {p.orientation for p in via.ports}
        ports1: Iterable[Port] = []
        ports2: Iterable[Port] = []
        if port_orientation1 is None:
            ports1 = via_stack_west.ports
        else:
            ports1 = via_stack_west.ports.filter(orientation=port_orientation1)

        if port_orientation2 is None:
            ports2 = via_stack_east.ports
        else:
            ports2 = via_stack_east.ports.filter(orientation=port_orientation2)

        if not ports1:
            raise ValueError(
                f"No ports for port_orientation1 {port_orientation1} in {valid_orientations}"
            )
        if not ports2:
            raise ValueError(
                f"No ports for port_orientation2 {port_orientation2} in {valid_orientations}"
            )

        c.add_ports(ports1, prefix="l_")
        c.add_ports(ports2, prefix="r_")

    # delete any straights with zero length
    for inst in list(c.insts):
        if inst.cell.settings.get("length") == 0.0:
            del c.insts[inst]
    c.flatten()
    return c


if __name__ == "__main__":
    # rows = 3
    # length = 300.0
    # spacing = 3

    # c = gf.Component()
    # p1 = gf.Port(center=(0, 0), orientation=0)
    # p2 = gf.Port(center=(0, spacing), orientation=0)
    # route = gf.routing.route_single(p1, p2)
    # straight_length = gf.snap.snap_to_grid((length - (rows - 1) * route.length) / rows)
    # straight_array = c << gf.components.array(spacing=(0, spacing), columns=1, rows=rows)

    # for row in range(1, rows, 2):
    #     route = gf.routing.route_single(
    #         straight_array.ports[f"o2_{row+1}_1"], straight_array.ports[f"o2_{row}_1"]
    #     )
    #     c.add(route.references)

    #     route = gf.routing.route_single(
    #         straight_array.ports[f"o1_{row+1}_1"], straight_array.ports[f"o1_{row+2}_1"]
    #     )
    #     c.add(route.references)

    # c.add_port("o1", port=straight_array.ports["o1_1_1"])
    # c.add_port("o2", port=straight_array.ports[f"o2_{rows}_1"])

    c = straight_heater_meander_doped(
        # straight_widths=(0.5,) * 7,
        taper_length=10,
        # taper_length=10,
        length=2000,
        # cross_section=partial(gf.cross_section.strip, width=0.8),
    )
    c.show()
    # scene = c.to_3d()
    # scene.show()
