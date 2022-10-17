import numpy as np

import gdsfactory as gf


@gf.cell
def spiral_heater(
    min_radius,
    straight_length,
    spacing,
    num,
    straight_factory=gf.components.straight,
    bend_factory=gf.components.bend_euler,
):
    c = gf.Component()

    center_straight = c << straight_factory(straight_length)
    angle = 187
    center_straight.rotate(180 - angle)

    for port in center_straight.ports.values():
        bend = c << bend_factory(angle=angle, radius=min_radius, p=0)
        bend.connect("o1", port)

        straight = c << straight_factory(1.5 * straight_length)
        straight.connect("o1", bend.ports["o2"])

        inner_loop_width = (
            4 * min_radius
            - straight_length
            * np.sin(np.deg2rad(-angle))  # width of the area of the inner straight
            - 2
            * min_radius
            * (
                1 - np.cos(np.deg2rad(180 - angle))
            )  # width of the part of the bend >180
        )

        print(inner_loop_width)
        print(
            min_radius * (1 - np.cos(np.deg2rad(180 - angle))),
            np.cos(np.deg2rad(180 - angle)),
        )

        for i in range(num):
            bend = c << bend_factory(
                angle=180, radius=min_radius + (i + 1) * spacing, p=0
            )
            bend.connect("o1", straight.ports["o2"])

            straight = c << straight_factory(2 * straight_length)
            straight.connect("o1", bend.ports["o2"])

    return c


if __name__ == "__main__":
    heater = spiral_heater(10, 100, 5, 10)
    heater.flatten().show()
