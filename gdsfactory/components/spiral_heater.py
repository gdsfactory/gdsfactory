from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.types import ComponentFactory


@gf.cell
def spiral_heater(
    min_radius: float,
    straight_length: float,
    spacings: Tuple[float, ...],
    straight_factory: ComponentFactory = gf.components.straight,
    bend_factory: ComponentFactory = gf.components.bend_euler,
    bend_s_factory: ComponentFactory = gf.components.bend_s,
):
    c = gf.Component()

    bend_s = c << bend_s_factory((straight_length, -min_radius * 2 - spacings[0]))

    for port in bend_s.ports.values():
        for i in range(len(spacings)):
            bend = c << bend_factory(
                angle=180, radius=min_radius + np.sum(spacings[: i + 1]), p=0
            )
            bend.connect("o1", port)

            straight = c << straight_factory(straight_length)
            straight.connect("o1", bend.ports["o2"])
            port = straight.ports["o2"]

    return c


if __name__ == "__main__":
    heater = spiral_heater(3.0, 30.0, (2, 2, 3, 3, 2, 2))
    heater.flatten().show()
