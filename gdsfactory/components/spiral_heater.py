from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight
from gdsfactory.types import ComponentFactory, CrossSectionSpec


@gf.cell
def spiral_racetrack(
    min_radius: float,
    straight_length: float,
    spacings: Tuple[float, ...],
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    cross_section: CrossSectionSpec = gf.cross_section.strip,
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

    """
    c = gf.Component()

    bend_s = c << bend_s_factory(
        (straight_length, -min_radius * 2 - spacings[0]), cross_section=cross_section
    )

    ports = []
    for port in bend_s.ports.values():
        for i in range(len(spacings)):
            bend = c << bend_factory(
                angle=180,
                radius=min_radius + np.sum(spacings[: i + 1]),
                p=0,
                cross_section=cross_section,
            )
            bend.connect("o1", port)

            straight = c << straight_factory(
                straight_length, cross_section=cross_section
            )
            straight.connect("o1", bend.ports["o2"])
            port = straight.ports["o2"]
        ports.append(port)

    c.add_port("o1", port=ports[0])
    c.add_port("o2", port=ports[1])
    return c


@gf.cell
def spiral_racetrack_heater_metal(
    min_radius: float,
    straight_length: float,
    spacing: float,
    num: int,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    waveguide_cross_section: CrossSectionSpec = gf.cross_section.strip,
    heater_cross_section: CrossSectionSpec = gf.cross_section.heater_metal,
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

    spiral = c << spiral_racetrack(
        min_radius,
        straight_length,
        (spacing,) * num,
        straight_factory,
        bend_factory,
        bend_s_factory,
        waveguide_cross_section,
    )

    heater_straight = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_straight.connect("e1", spiral.ports["o1"].copy().rotate(180)).movey(
        spacing * num // 2
    )

    heater_straight = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_straight.connect("e1", spiral.ports["o2"].copy().rotate(180)).movey(
        -spacing * num // 2
    )

    heater_bend = c << gf.components.bend_euler(
        angle=180,
        radius=min_radius + spacing * (num // 2 + 1),
        cross_section=heater_cross_section,
        p=0,
    )
    heater_bend.connect("e1", heater_straight.ports["e1"])

    return c


@gf.cell
def spiral_racetrack_heater_doped(
    min_radius: float,
    straight_length: float,
    spacing: float,
    num: int,
    straight_factory: ComponentFactory = straight,
    bend_factory: ComponentFactory = bend_euler,
    bend_s_factory: ComponentFactory = bend_s,
    waveguide_cross_section: CrossSectionSpec = gf.cross_section.strip,
    heater_cross_section: CrossSectionSpec = gf.cross_section.npp,
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

    heater_straight = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_straight.connect("o1", spiral.ports["o1"].copy().rotate(180)).movey(
        -spacing * (num // 2)
    )

    heater_straight = c << gf.components.straight(
        straight_length, cross_section=heater_cross_section
    )
    heater_straight.connect("o1", spiral.ports["o2"].copy().rotate(180)).movey(
        spacing * (num // 2)
    )

    return c


if __name__ == "__main__":
    # heater = spiral_racetrack(
    #     min_radius=3.0, straight_length=30.0, spacings=(2, 2, 3, 3, 2, 2)
    # )
    # heater.show()

    # heater = spiral_racetrack_heater_metal(3, 30, 2, 5)
    # heater.show()

    heater = spiral_racetrack_heater_doped(
        min_radius=3, straight_length=30, spacing=2, num=8
    )
    heater.show()
