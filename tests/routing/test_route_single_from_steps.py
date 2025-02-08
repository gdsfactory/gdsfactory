import gdsfactory as gf


def test_route_from_steps() -> None:
    """Test route_single with steps."""
    c = gf.Component()
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.dmove((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.dymin = 40
    obstacle2.dxmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    gf.routing.route_single(
        c,
        cross_section="strip",
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20},
            {"y": 20},
            {"x": 120},
            {"y": 80},
        ],
    )


def test_route_waypoints() -> None:
    """Test route_single with waypoints."""
    c = gf.Component(name="electrical")
    w = gf.components.wire_straight()
    left = c << w
    right = c << w
    right.dmove((100, 80))
    obstacle = gf.components.rectangle(size=(100, 10))
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.dymin = 40
    obstacle2.dxmin = 25

    p0 = left.ports["e2"]
    p1 = right.ports["e2"]
    p0x, p0y = left.ports["e2"].center
    p1x, p1y = right.ports["e2"].center
    o = 10  # vertical offset to overcome bottom obstacle
    ytop = 20

    gf.routing.route_single(
        c,
        p0,
        p1,
        cross_section="metal_routing",
        waypoints=[
            (p0x + o, p0y),
            (p0x + o, ytop),
            (p1x + o, ytop),
            (p1x + o, p1y),
        ],
    )


def test_route_waypoints_numpy() -> None:
    """Test route_single with waypoints."""
    c = gf.Component()
    w = gf.components.wire_straight()
    left = c << w
    right = c << w
    right.dmove((100, 80))
    obstacle = gf.components.rectangle(size=(100, 10))
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.dymin = 40
    obstacle2.dxmin = 25

    p0 = left.ports["e2"]
    p1 = right.ports["e2"]
    p0x, p0y = left.ports["e2"].center
    p1x, p1y = right.ports["e2"].center
    o = 10  # vertical offset to overcome bottom obstacle
    ytop = 20

    gf.routing.route_single(
        c,
        p0,
        p1,
        cross_section="metal_routing",
        waypoints=[
            (p0x + o, p0y),
            (p0x + o, ytop),
            (p1x + o, ytop),
            (p1x + o, p1y),
        ],
    )


if __name__ == "__main__":
    test_route_from_steps()
    test_route_waypoints()
    test_route_waypoints_numpy()
