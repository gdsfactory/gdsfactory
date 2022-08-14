import gdsfactory as gf


def test_route_error2():
    """Ensures that an impossible route raises value Error."""
    c = gf.Component("pads_route_from_steps")
    pt = c << gf.components.pad_array(orientation=270, columns=3)
    pb = c << gf.components.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps(
        pt.ports["e11"],
        pb.ports["e11"],
        steps=[
            {"y": 100},
        ],
        cross_section=gf.cross_section.metal3,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    return c


if __name__ == "__main__":
    c = test_route_error2()
    c.show(show_ports=True)
