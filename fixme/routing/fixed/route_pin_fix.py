"""Problem.

- bends in routes flip the polarity in some cases

temporary Solution / hack:

- create a smooth path and extrude it

Problems:

- extruding path removes the route elements complicating simulation.

"""

import gdsfactory as gf


if __name__ == "__main__":
    c = gf.Component()

    length = 50
    s1 = c << gf.components.straight(length=length)
    s2 = c << gf.components.straight(length=length)

    xs_pin = gf.partial(gf.cross_section.pin, via_stack_width=3)

    s2.move((100, 60))
    radius = 20
    points = gf.routing.manhattan.generate_manhattan_waypoints(
        s1.ports["o2"], s2.ports["o1"], radius=radius
    )
    path = gf.path.smooth(points, radius=radius - 10)
    route = c << gf.path.extrude(path, xs_pin)
    c.show(show_ports=True)
