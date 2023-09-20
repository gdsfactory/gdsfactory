from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_s import bend_s, get_min_sbend_size
from gdsfactory.components.straight import straight
from gdsfactory.routing.get_route import get_route
from gdsfactory.typings import ComponentFactory, CrossSectionSpec, Floats


@gf.cell
def spiral_racetrack(
    min_radius: float = 5,
    straight_length: float = 10.0,
    spacings: Floats = (2, 2, 3, 3, 2, 2),
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
    n_bend_points: int | None = None,
    with_inner_ports: bool = False,
    extra_90_deg_bend: bool = False,
) -> Component:
    """Returns Racetrack-Spiral.

    Args:
        min_radius: smallest radius in um.
        straight_length: length of the straight segments in um.
        spacings: space between the center of neighboring waveguides in um.
        straight_factory: factory to generate the straight segments.
        bend_factory: factory to generate the bend segments.
        bend_s_factory: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        n_bend_points: optional bend points.
        with_inner_ports: if True, will build the spiral, but expose the inner ports where the S-bend would be.
        extra_90_deg_bend: if True, we add an additional straight + 90 degree bent at the output, so the
            output port is looking down.
    """
    c = gf.Component()

    if with_inner_ports:
        bend_s_component = bend_s_factory(
            (straight_length, -min_radius * 2 + 1 * spacings[0]),
            cross_section=cross_section_s or cross_section,
            **({"npoints": n_bend_points} if n_bend_points else {}),
        )
        bend_s = type("obj", (object,), {"ports": bend_s_component.ports})
        c.info["length"] = 0
        c.add_port(
            "o3",
            center=bend_s.ports["o1"].center,
            orientation=0,
            cross_section=bend_s.ports["o1"].cross_section,
        )
        c.add_port(
            "o4",
            center=bend_s.ports["o2"].center,
            orientation=180,
            cross_section=bend_s.ports["o2"].cross_section,
        )
    else:
        bend_s = c << bend_s_factory(
            (straight_length, -min_radius * 2 + 1 * spacings[0]),
            cross_section=cross_section_s or cross_section,
            **({"npoints": n_bend_points} if n_bend_points else {}),
        )
        c.info["length"] = bend_s.info["length"]

    ports = []
    for port in bend_s.ports.values():
        for i in range(len(spacings)):
            bend = c << bend_factory(
                angle=180,
                radius=min_radius + np.sum(spacings[:i]),
                p=0,
                cross_section=cross_section,
                **({"npoints": n_bend_points} if n_bend_points else {}),
            )
            bend.connect("o1", port)

            straight = c << straight_factory(
                straight_length, cross_section=cross_section
            )
            straight.connect("o1", bend.ports["o2"])
            port = straight.ports["o2"]

            c.info["length"] += bend.info["length"] + straight.info["length"]
        ports.append(port)

    c.add_port("o1", port=ports[0])

    if extra_90_deg_bend:
        bend = c << bend_factory(
            angle=90,
            radius=min_radius + np.sum(spacings),
            p=0,
            cross_section=cross_section,
            **({"npoints": n_bend_points} if n_bend_points else {}),
        )
        bend.connect("o1", ports[1])
        c.add_port("o2", port=bend.ports["o2"])

    else:
        c.add_port("o2", port=ports[1])
    return c


@gf.cell
def spiral_racetrack_fixed_length(
    length: float = 1000,
    in_out_port_spacing: float = 150,
    n_straight_sections: int = 8,
    min_radius: float = 5,
    min_spacing: float = 5.0,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    cross_section: CrossSectionSpec = "strip",
    cross_section_s: CrossSectionSpec | None = None,
    n_bend_points: int | None = None,
    with_inner_ports: bool = False,
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
        straight_factory: factory to generate the straight segments.
        bend_factory: factory to generate the bend segments.
        bend_s_factory: factory to generate the s-bend segments.
        cross_section: cross-section of the waveguides.
        cross_section_s: cross-section of the s bend waveguide (optional).
        n_bend_points: optional bend points.
        with_inner_ports: if True, will build the spiral, but expose the inner ports where the S-bend would be.
    """

    c = gf.Component()

    xs_s_bend = cross_section_s or cross_section

    if np.mod(n_straight_sections, 2) != 0:
        raise ValueError("The number of straight sections has to be even!")

    # get the length of the straight sections to achieve the required length
    spacings = (min_spacing,) * (n_straight_sections // 2)

    straight_length = _req_straight_len(
        length=length,
        in_out_port_spacing=in_out_port_spacing,
        min_radius=min_radius,
        spacings=spacings,
        bend_factory=bend_factory,
        bend_s_factory=bend_s_factory,
        cross_section_s_bend=xs_s_bend,
        cross_section=cross_section,
    )

    spiral = c << spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=spacings,
        straight_factory=straight_factory,
        bend_factory=bend_factory,
        bend_s_factory=bend_s_factory,
        cross_section=cross_section,
        cross_section_s=cross_section_s,
        n_bend_points=n_bend_points,
        with_inner_ports=with_inner_ports,
        extra_90_deg_bend=True,
    )

    c.info["length"] = spiral.info["length"]
    c.info["straight_length"] = straight_length
    c.info["spiral_center"] = spiral.center

    if spiral.ports["o1"].x > spiral.ports["o2"].x:
        spiral.mirror_x()

    # add a bit more to the spiral racetrack to make the in and out ports be aligned in y
    in_wg = c << straight_factory(
        spiral.ports["o1"].x - spiral.xmin, cross_section=cross_section
    )
    if np.mod(n_straight_sections // 2, 2) == 1:
        in_wg.mirror_y()
    in_wg.connect("o1", spiral.ports["o1"])

    c.info["length"] += spiral.ports["o1"].x - spiral.xmin

    c.add_port(
        "o2_temp",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=180,
        cross_section=gf.get_cross_section(xs_s_bend),
    )

    route = get_route(
        spiral.ports["o2"],
        c.ports["o2_temp"],
        straight=straight,
        bend=bend_factory,
        cross_section=xs_s_bend,
    )
    c.add(route.references)

    c.ports.pop("o2_temp")

    c.add_port(
        "o2",
        center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
        orientation=0,
        cross_section=gf.get_cross_section(xs_s_bend),
    )

    c.info["length"] += np.sum([r.info["length"] for r in route.references])
    c.add_port("o1", port=in_wg.ports["o2"])
    return c


def _req_straight_len(
    length: float = 1000,
    in_out_port_spacing: float = 100,
    min_radius: float = 5,
    spacings: Floats = (1.0, 1.0),
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    cross_section: CrossSectionSpec = "strip",
    cross_section_s_bend: CrossSectionSpec = "strip",
) -> float:
    """Returns geometrical parameters to make a spiral of a given length.

    Args:
        length: total length of the spiral from input to output ports in um.
        in_out_port_spacing: spacing between input and output ports of the spiral in um.
        min_radius: smallest radius in um.
        spacings: spacings between adjacent waveguides.
        bend_factory: factory to generate the bend segments.
        bend_s_factory: factory to generate the s-bend segments.
        cross_section_s_bend: s bend cross section
    """
    from scipy.interpolate import interp1d

    # "Brute force" approach - sweep length and save total length

    lens = []

    # Figure out the min straight for the spiral so that the inner
    # s bend has min radius within the bend radius of the waveguide
    min_straigth_length = get_min_sbend_size(
        [None, -min_radius * 2 + 1 * spacings[0]], cross_section_s_bend
    )

    if min_straigth_length > 0.8 * in_out_port_spacing:
        raise ValueError(
            "The maximum straight length makes the inner s bend too tight. Increase the in-out port spacing."
        )

    straight_lengths = np.linspace(min_straigth_length, 0.9 * in_out_port_spacing, 100)

    for str_len in straight_lengths:
        c = gf.Component()

        spiral = c << spiral_racetrack(
            min_radius=min_radius,
            straight_length=str_len,
            spacings=spacings,
            straight_factory=straight,
            bend_factory=bend_factory,
            bend_s_factory=bend_s_factory,
            cross_section=cross_section,
            cross_section_s=cross_section_s_bend,
            extra_90_deg_bend=True,
        )

        c.info["length"] = spiral.info["length"]

        if spiral.ports["o1"].x > spiral.ports["o2"].x:
            spiral.mirror_x()

        c.info["length"] += spiral.ports["o1"].x - spiral.xmin

        c.add_port(
            "o2",
            center=(spiral.ports["o1"].x + in_out_port_spacing, spiral.ports["o1"].y),
            orientation=180,
            cross_section=gf.get_cross_section(cross_section_s_bend),
        )

        route = get_route(
            spiral.ports["o2"],
            c.ports["o2"],
            straight=straight,
            bend=bend_factory,
            cross_section=cross_section_s_bend,
        )
        c.add(route.references)

        c.info["length"] += np.sum([r.info["length"] for r in route.references])

        lens.append(c.info["length"])

    # get the required spacing to achieve the required length (interpolate)
    f = interp1d(lens, straight_lengths)
    return f(length)


@gf.cell
def spiral_racetrack_heater_metal(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "heater_metal",
) -> Component:
    """Returns spiral racetrack with a heater above.

    based on https://doi.org/10.1364/OL.400230 .

    Args:
        min_radius: smallest radius.
        straight_length: length of the straight segments.
        spacing: space between the center of neighboring waveguides.
        num: number.
        straight_factory: factory to generate the straight segments.
        bend_factory: factory to generate the bend segments.
        bend_s_factory: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
    """
    c = gf.Component()
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius or xs.radius

    spiral = c << spiral_racetrack(
        min_radius,
        straight_length,
        (spacing,) * num,
        straight_factory,
        bend_factory,
        bend_s_factory,
        waveguide_cross_section,
    )

    heater_top = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_top.connect("e1", spiral.ports["o1"].copy().rotate(180)).movey(
        spacing * num // 2
    )
    heater_bot = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_bot.connect("e1", spiral.ports["o2"].copy().rotate(180)).movey(
        -spacing * num // 2
    )

    heater_bend = c << gf.components.bend_circular(
        angle=180,
        radius=min_radius + spacing * (num // 2 + 1),
        cross_section=heater_cross_section,
    )
    heater_bend.y = spiral.y
    heater_bend.x = spiral.x + min_radius + spacing * (num // 2 + 1)
    heater_top.connect("e1", heater_bend.ports["e1"])
    heater_bot.connect("e1", heater_bend.ports["e2"])

    c.add_ports(spiral.ports)
    c.add_port("e1", port=heater_bot["e2"])
    c.add_port("e2", port=heater_top["e2"])
    return c


@gf.cell
def spiral_racetrack_heater_doped(
    min_radius: float | None = None,
    straight_length: float = 30,
    spacing: float = 2,
    num: int = 8,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    waveguide_cross_section: CrossSectionSpec = "strip",
    heater_cross_section: CrossSectionSpec = "npp",
) -> Component:
    """Returns spiral racetrack with a heater between the loops.

    based on https://doi.org/10.1364/OL.400230 but with the heater between the loops.

    Args:
        min_radius: smallest radius in um.
        straight_length: length of the straight segments in um.
        spacing: space between the center of neighboring waveguides in um.
        num: number.
        straight_factory: factory to generate the straight segments.
        bend_factory: factory to generate the bend segments.
        bend_s_factory: factory to generate the s-bend segments.
        waveguide_cross_section: cross-section of the waveguides.
        heater_cross_section: cross-section of the heater.
    """
    xs = gf.get_cross_section(waveguide_cross_section)
    min_radius = min_radius or xs.radius

    c = gf.Component()

    spiral = c << spiral_racetrack(
        min_radius=min_radius,
        straight_length=straight_length,
        spacings=(spacing,) * (num // 2)
        + (spacing + 1,) * 2
        + (spacing,) * (num // 2 - 2),
        straight_factory=straight_factory,
        bend_factory=bend_factory,
        bend_s_factory=bend_s_factory,
        cross_section=waveguide_cross_section,
    )

    heater_straight = gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )

    heater_top = c << heater_straight
    heater_bot = c << heater_straight

    heater_bot.connect("e1", spiral.ports["o1"].copy().rotate(180)).movey(
        -spacing * (num // 2 - 1)
    )

    heater_top.connect("e1", spiral.ports["o2"].copy().rotate(180)).movey(
        spacing * (num // 2 - 1)
    )

    c.add_ports(spiral.ports)
    c.add_ports(prefix="top_", ports=heater_top.ports)
    c.add_ports(prefix="bot_", ports=heater_bot.ports)
    return c


def test_length_spiral_racetrack() -> None:
    import numpy as np

    length = 1000
    c = spiral_racetrack_fixed_length(length=length, cross_section="strip_no_pins")
    length_computed = c.area() / 0.5
    np.isclose(length, length_computed)


@gf.cell
def spiral_slab(cross_section="strip", layer_slab=(3, 0), cladding_offset=3, **kwargs):
    xs = gf.get_cross_section(cross_section)
    xs_slab = gf.CrossSection(layer=layer_slab, width=xs.width + cladding_offset)

    c = gf.Component()

    o = 2.5
    s1 = spiral_racetrack(cross_section=cross_section, **kwargs)
    s2 = (
        spiral_racetrack(cross_section=xs_slab, **kwargs)
        .offset(+o)
        .offset(-o, layer=(2, 0))
    )

    ref = c << s1
    c << s2

    c.copy_child_info(s1)
    c.add_ports(ref.ports)
    return c


if __name__ == "__main__":
    # c = spiral_racetrack(cross_section="rib_conformal", cache=False)

    c = spiral_slab(cache=False)
    c.show(show_ports=True)
