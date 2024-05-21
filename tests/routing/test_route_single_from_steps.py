import gdsfactory as gf


def test_route_from_steps():
    c = gf.Component()
    w = gf.components.straight()
    left = c << w
    right = c << w
    right.d.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 10), port_type=None)
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.d.ymin = 40
    obstacle2.d.xmin = 25

    p1 = left.ports["o2"]
    p2 = right.ports["o2"]
    gf.routing.route_single_from_steps(
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
