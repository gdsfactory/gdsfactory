import gdsfactory as gf
from gdsfactory.types import ComponentFactory


@gf.cell
def spiral_heater(
    min_radius: float,
    straight_length: float,
    spacing: float,
    num: int,
    straight_factory: ComponentFactory = gf.components.straight,
    bend_factory: ComponentFactory = gf.components.bend_euler,
    bend_s_factory: ComponentFactory = gf.components.bend_s,
):
    c = gf.Component()

    bend_s = c << bend_s_factory((straight_length, -min_radius * 2 - spacing))

    for port in bend_s.ports.values():
        for i in range(num):
            bend = c << bend_factory(
                angle=180, radius=min_radius + (i + 1) * spacing, p=0
            )
            bend.connect("o1", port)

            straight = c << straight_factory(straight_length)
            straight.connect("o1", bend.ports["o2"])
            port = straight.ports["o2"]

    return c


if __name__ == "__main__":
    heater = spiral_heater(3.0, 30.0, 2.0, 10)
    heater.flatten().show()
