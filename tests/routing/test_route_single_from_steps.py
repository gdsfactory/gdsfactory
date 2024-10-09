import gdsfactory as gf


def test_route_from_steps() -> None:
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
        port1=p1,
        port2=p2,
        steps=[
            {"x": 20},
            {"y": 20},
            {"x": 120},
            {"y": 80},
        ],
    )


if __name__ == "__main__":
    test_route_from_steps()
