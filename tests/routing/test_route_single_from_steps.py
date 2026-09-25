import pytest

import gdsfactory as gf


def test_route_from_steps() -> None:
    """Test route_bundle with steps."""
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
    gf.routing.route_bundle(
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
    """Test route_bundle with waypoints."""
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

    gf.routing.route_bundle(
        c,
        p0,
        p1,
        cross_section="metal_routing",
        bend="wire_corner",
        waypoints=[
            (p0x + o, p0y),
            (p0x + o, ytop),
            (p1x + o, ytop),
            (p1x + o, p1y),
        ],
    )


def test_route_waypoints_numpy() -> None:
    """Test route_bundle with waypoints."""
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

    gf.routing.route_bundle(
        c,
        p0,
        p1,
        bend="wire_corner",
        cross_section="metal_routing",
        waypoints=[
            (p0x + o, p0y),
            (p0x + o, ytop),
            (p1x + o, ytop),
            (p1x + o, p1y),
        ],
    )


def _two_straights(cross_section: str) -> tuple[gf.Component, gf.Port, gf.Port]:
    c = gf.Component()
    straight = gf.components.straight(cross_section=cross_section)
    left = c << straight
    right = c << straight
    right.move((300, 200))
    return c, left.ports[1], right.ports[0]


def _bends(c: gf.Component) -> list[str]:
    return [
        inst.cell.name
        for inst in c.insts
        if inst.cell.name.startswith(("bend_euler", "wire_corner"))
    ]


@pytest.mark.parametrize(
    ("cross_section", "bend"),
    [
        ("strip", "bend_euler"),
        # Optical, even though it has an electrical heater section.
        ("strip_heater_metal", "bend_euler"),
        ("metal_routing", "wire_corner"),
    ],
)
def test_route_single_default_bend(cross_section: str, bend: str) -> None:
    c, port1, port2 = _two_straights(cross_section)
    gf.routing.route_single(c, port1, port2, cross_section=cross_section)
    bends = _bends(c)
    assert bends
    assert all(name.startswith(bend) for name in bends)


@pytest.mark.parametrize("with_waypoints", [False, True])
def test_route_single_incompatible_bend(with_waypoints: bool) -> None:
    c, port1, port2 = _two_straights("strip")
    waypoints = (
        [(port1.x + 50, port1.y), (port1.x + 50, port2.y)] if with_waypoints else None
    )
    with pytest.raises(ValueError, match="0 'optical' ports"):
        gf.routing.route_single(
            c,
            port1,
            port2,
            cross_section="strip",
            bend="wire_corner",
            waypoints=waypoints,
        )
