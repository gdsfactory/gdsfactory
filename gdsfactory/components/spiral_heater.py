from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentFactory, CrossSectionSpec, Floats, Optional


@gf.cell
def spiral_racetrack(
    min_radius: float = 5,
    straight_length: float = 10.0,
    spacings: Floats = (2, 2, 3, 3, 2, 2),
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    cross_section: CrossSectionSpec = "strip",
    n_bend_points: Optional[int] = None,
    with_inner_ports: bool = False,
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
    """
    c = gf.Component()

    if with_inner_ports:
        bend_s_component = bend_s_factory(
            (straight_length, -min_radius * 2 + 1 * spacings[0]),
            cross_section=cross_section,
            **({"nb_points": n_bend_points} if n_bend_points else {}),
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
            cross_section=cross_section,
            **({"nb_points": n_bend_points} if n_bend_points else {}),
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
    c.add_port("o2", port=ports[1])
    return c


@gf.cell
def spiral_racetrack_heater_metal(
    min_radius: Optional[float] = None,
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
    min_radius: Optional[float] = None,
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


if __name__ == "__main__":
    # heater = spiral_racetrack(
    #     min_radius=3.0, straight_length=30.0, spacings=(2, 2, 3, 3, 2, 2)
    # )
    # heater.show()

    # heater = spiral_racetrack_heater_metal(3, 30, 2, 5)
    # heater.show()

    # heater = spiral_racetrack_heater_doped(
    #     min_radius=3, straight_length=30, spacing=2, num=8
    # )
    # heater.show()
    c = spiral_racetrack(with_inner_ports=True)
    # c = spiral_racetrack_heater_doped()
    c.show(show_ports=True)
