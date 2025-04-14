from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bends.bend_euler import bend_euler
from gdsfactory.components.bends.bend_s import get_min_sbend_size
from gdsfactory.components.waveguides.straight import straight
from gdsfactory.routing.route_single import route_single
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Floats,
    Port,
)


@gf.cell_with_module_name
def spiral_racetrack(
    min_radius: float | None = None,
    straight_length: float = 20.0,
    spacings: Floats = (2, 2, 3, 3, 2, 2),
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
    extra_90_deg_bend: bool = False,
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns Racetrack-Spiral.

    Args:
        min_radius: smallest radius in um.
        straight_length: length of the straight segments in um.
        spacings: space between the center of neighboring waveguides in um.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
        extra_90_deg_bend: if True, we add an additional straight + 90 degree bent at the output, so the output port is looking down.
        allow_min_radius_violation: if True, will allow the s-bend to have a smaller radius than the minimum radius.
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    min_radius = min_radius or xs.radius
    assert min_radius

    _bend_s = gf.get_component(
        bend_s,
        size=(straight_length, -min_radius * 2 + 1 * spacings[0]),
        cross_section=cross_section_s or cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )
    bend_s_ref = c << _bend_s
    c.info["length"] = _bend_s.info["length"]

    ports: list[Port] = []
    for port in bend_s_ref.ports:
        for i in range(len(spacings)):
            _bend = gf.get_component(
                bend,
                angle=180,
                radius=min_radius + np.sum(spacings[:i]),
                cross_section=cross_section,
            )
            bend_ref = c << _bend
            bend_ref.connect("o1", port)

            _straight = gf.get_component(
                straight, length=straight_length, cross_section=cross_section
            )
            straight_ref = c << _straight
            straight_ref.connect("o1", bend_ref.ports["o2"])
            port = straight_ref.ports["o2"]

            c.info["length"] += _bend.info["length"] + _straight.info["length"]
        ports.append(port)

    c.add_port("o1", port=ports[0])

    if extra_90_deg_bend:
        bend_ref = c << gf.get_component(
            bend,
            angle=90,
            radius=min_radius + np.sum(spacings),
            cross_section=cross_section,
        )
        bend_ref.connect("o1", ports[1])
        c.add_port("o2", port=bend_ref.ports["o2"])

    else:
        c.add_port("o2", port=ports[1])
    return c


@gf.cell_with_module_name
def spiral_racetrack_fixed_length(
    length: float = 1000,
    in_out_port_spacing: float = 150,
    n_straight_sections: int = 8,
    min_radius: float | None = None,
    min_spacing: float = 5.0,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = "bend_circular",
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
) -> Component:
    """Returns Racetrack-Spiral with a specified total length.

    The input and output ports are aligned in y. This class is meant to
    be used for generating interferometers with long waveguide lengths, where
    the most important parameter is the length difference between the arms.

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        n_straight_sections: total number of straight sections for the racetrack spiral. Has to be even.
        min_radius: smallest radius in um.
        min_spacing: minimum center-center spacing between adjacent waveguides.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
    """
    c = gf.Component()

    xs_s_bend = cross_section_s or cross_section
    xs = gf.get_cross_section(xs_s_bend)
    min_radius = min_radius or xs.radius

    if np.mod(n_straight_sections, 2) != 0:
        raise ValueError("The number of straight sections has to be even!")

    # get the length of the straight sections to achieve the required length
    spacings = (min_spacing,) * (n_straight_sections // 2)

    straight_length = _req_straight_len(
        length=length,
        in_out_port_spacing=in_out_port_spacing,
        min_radius=min_radius,
        spacings=spacings,
        bend=bend,
        bend_s=bend_s,
        cross_section_s_bend=xs_s_bend,
        cross_section=cross_section,
    )

    _spiral = spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=spacings,
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=cross_section,
        cross_section_s=cross_section_s,
        extra_90_deg_bend=True,
    )

    spiral = c << _spiral
    c.info["length"] = _spiral.info["length"]
    c.info["straight_length"] = straight_length

    if spiral.ports["o1"].x > spiral.ports["o2"].x:
        spiral.mirror_x()

    # add a bit more to the spiral racetrack to make the in and out ports be aligned in y
    in_wg = c << gf.get_component(
        straight,
        length=spiral.ports["o1"].x - spiral.xmin,
        cross_section=cross_section,
    )
    if np.mod(n_straight_sections // 2, 2) == 1:
        in_wg.mirror_y()
    in_wg.connect("o1", spiral.ports["o1"])

    c.info["length"] += spiral.ports["o1"].x - spiral.xmin

    temp_component = Component()

    o2_temp = temp_component.add_port(
        name="o2_temp",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=180,
        cross_section=gf.get_cross_section(xs_s_bend),
    )

    route = route_single(
        c,
        spiral.ports["o2"],
        o2_temp,
        straight=straight,
        bend=bend,
        cross_section=xs_s_bend,
        radius=min_radius,
    )

    c.add_port(
        "o2",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=0,
        cross_section=gf.get_cross_section(xs_s_bend),
    )
    c.add_port("o1", port=in_wg.ports["o2"])
    c.info["length"] += c.kcl.dbu * route.length
    return c


def _req_straight_len(
    length: float = 1000,
    in_out_port_spacing: float = 100,
    min_radius: float | None = None,
    spacings: Floats = (1.0, 1.0),
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    cross_section: CrossSectionSpec = "strip",
    cross_section_s_bend: CrossSectionSpec = "strip",
) -> float:
    """Returns geometrical parameters to make a spiral of a given length.

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        min_radius: smallest radius in um. Defaults to the radius of the cross-section.
        spacings: spacings between adjacent waveguides.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s_bend: s bend cross section
    """
    from scipy.interpolate import interp1d

    xs = gf.get_cross_section(cross_section)
    min_radius = min_radius or xs.radius
    assert min_radius

    # "Brute force" approach - sweep length and save total length
    lens: list[float] = []

    # Figure out the min straight for the spiral so that the inner
    # s bend has min radius within the bend radius of the waveguide
    min_straigth_length = get_min_sbend_size(
        (None, -min_radius * 2 + 1 * spacings[0]), cross_section_s_bend
    )

    if min_straigth_length > 0.8 * in_out_port_spacing:
        raise ValueError(
            "The maximum straight length makes the inner s bend too tight. Increase the in-out port spacing."
        )

    straight_lengths = np.linspace(min_straigth_length, 0.9 * in_out_port_spacing, 100)

    for str_len in straight_lengths:
        c = gf.Component()

        _spiral = spiral_racetrack(
            min_radius=min_radius,
            straight_length=str_len,
            spacings=spacings,
            straight=straight,
            bend=bend,
            bend_s=bend_s,
            cross_section=cross_section,
            cross_section_s=cross_section_s_bend,
            extra_90_deg_bend=True,
        )
        spiral = c << _spiral
        c.info["length"] = _spiral.info["length"]

        if spiral.ports["o1"].x > spiral.ports["o2"].x:
            spiral.mirror_x()

        c.info["length"] += spiral.ports["o1"].x - spiral.xmin

        c.add_port(
            "o2",
            center=(
                spiral.ports["o1"].x + in_out_port_spacing,
                spiral.ports["o1"].y,
            ),
            orientation=180,
            cross_section=gf.get_cross_section(cross_section_s_bend),
        )
        route = route_single(
            c,
            spiral.ports["o2"],
            c.ports["o2"],
            straight=straight,
            bend=bend,
            cross_section=cross_section_s_bend,
            radius=min_radius,
        )
        c.info["length"] += c.kcl.dbu * route.length
        lens.append(c.info["length"])

    # get the required spacing to achieve the required length (interpolate)
    f = interp1d(lens, straight_lengths)
    return float(f(length))


@gf.cell_with_module_name
def spiral_racetrack_heater_metal(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "heater_metal",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
) -> Component:
    """Returns spiral racetrack with a heater above.

    based on https://doi.org/10.1364/OL.400230 .

    Args:
        min_radius: smallest radius. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments.
        spacing: space between the center of neighboring waveguides.
        num: number of loops.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
        via_stack: via stack to connect the heater to the metal layer.
    """
    c = gf.Component()
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius or xs.radius or 0

    spiral = c << spiral_racetrack(
        min_radius,
        straight_length,
        (spacing,) * num,
        straight,
        bend,
        bend_s,
        waveguide_cross_section,
    )

    heater_top = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_top.connect(
        "e1",
        spiral.ports["o1"].copy().copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_top.movey(spacing * num // 2)
    heater_bot = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_bot.connect(
        "e1",
        spiral.ports["o2"].copy().copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_bot.movey(-spacing * num // 2)

    heater_bend = c << gf.get_component(
        bend,
        angle=180,
        radius=min_radius + spacing * (num // 2 + 1),
        cross_section=heater_cross_section,
    )
    heater_bend.y = spiral.y
    heater_bend.x = spiral.x + min_radius + spacing * (num // 2 + 1)
    heater_top.connect("e1", heater_bend.ports["e1"])
    heater_bot.connect("e1", heater_bend.ports["e2"])

    c.add_ports(spiral.ports)

    if via_stack:
        via_stack = gf.get_component(via_stack)
        via_stack_top = c << via_stack
        via_stack_bot = c << via_stack
        via_stack_top.connect(
            "e3",
            heater_bot.ports["e2"],
            allow_layer_mismatch=True,
            allow_width_mismatch=True,
        )
        via_stack_bot.connect(
            "e3",
            heater_top.ports["e2"],
            allow_layer_mismatch=True,
            allow_width_mismatch=True,
        )

        p1 = via_stack_top.ports
        p2 = via_stack_bot.ports
        c.add_ports(p1, prefix="top_")
        c.add_ports(p2, prefix="bot_")

    else:
        c.add_port("e1", port=heater_bot["e2"])
        c.add_port("e2", port=heater_top["e2"])
    return c


@gf.cell_with_module_name
def spiral_racetrack_heater_doped(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight: ComponentSpec = straight,
    bend: ComponentSpec = bend_euler,
    bend_s: ComponentSpec = "bend_s",
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "npp",
) -> Component:
    """Returns spiral racetrack with a heater between the loops.

    based on https://doi.org/10.1364/OL.400230 but with the heater between the loops.

    Args:
        min_radius: smallest radius in um. Defaults to the radius of the cross-section.
        straight_length: length of the straight segments in um.
        spacing: space between the center of neighboring waveguides in um.
        num: number.
        straight: factory to generate the straight segments.
        bend: factory to generate the bend segments.
        bend_s: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
    """
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius or xs.radius or 0

    c = gf.Component()

    spiral = c << spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=(spacing,) * (num // 2)
        + (spacing + 1,) * 2
        + (spacing,) * (num // 2 - 2),
        straight=straight,
        bend=bend,
        bend_s=bend_s,
        cross_section=waveguide_cross_section,
    )

    heater_straight = gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )

    heater_top = c << heater_straight
    heater_bot = c << heater_straight

    heater_bot.connect(
        "e1",
        spiral.ports["o1"].copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_bot.movey(-spacing * (num // 2 - 1))
    heater_top.connect(
        "e1",
        spiral.ports["o2"].copy_polar(),
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
        allow_type_mismatch=True,
    )
    heater_top.movey(spacing * (num // 2 - 1))

    c.add_ports(spiral.ports)
    c.add_ports(prefix="top_", ports=heater_top.ports)
    c.add_ports(prefix="bot_", ports=heater_bot.ports)
    return c


if __name__ == "__main__":
    # test_length_spiral_racetrack()

    # c = spiral_racetrack(cross_section="rib")
    # c = spiral_racetrack(cross_section="nitride", straight_length=54)
    # c = spiral_racetrack()
    # c = spiral_racetrack_fixed_length()
    c = spiral_racetrack_heater_metal(
        waveguide_cross_section="nitride", straight_length=54
    )
    # c = spiral_racetrack_heater_doped()
    c.show()
