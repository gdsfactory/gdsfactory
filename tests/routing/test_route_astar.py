import importlib
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

import gdsfactory as gf

route_astar_module = importlib.import_module("gdsfactory.routing.route_astar")


def segment_crosses_box(
    p1: tuple[float, float],
    p2: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    xmin, ymin, xmax, ymax = box
    if x1 == x2:
        return xmin <= x1 <= xmax and max(min(y1, y2), ymin) <= min(max(y1, y2), ymax)
    if y1 == y2:
        return ymin <= y1 <= ymax and max(min(x1, x2), xmin) <= min(max(x1, x2), xmax)
    raise ValueError("route_astar backbone segments should be Manhattan")


def expanded_dbbox(
    ref: gf.ComponentReference,
    distance: float,
) -> tuple[float, float, float, float]:
    box = ref.dbbox()
    return (
        box.left - distance,
        box.bottom - distance,
        box.right + distance,
        box.top + distance,
    )


def test_route_astar_obstacle_route() -> None:
    c = gf.Component()
    cross_section = gf.get_cross_section("strip", radius=5)
    straight = gf.components.straight(cross_section=cross_section)
    left = c << straight
    right = c << straight
    right.rotate(90)
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M2")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacles = [obstacle1, obstacle2, obstacle3, obstacle4]
    obstacle4.rotate(90)
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23
    obstacle4.xmin = 200
    obstacle4.ymin = 55

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o1"],
        port2=right.ports["o2"],
        cross_section=cross_section,
        resolution=15,
        distance=12,
        avoid_layers=("M2",),
        bend=gf.components.bend_euler,
    )

    assert route.length > 0
    route_backbone = [(point.x / 1000, point.y / 1000) for point in route.backbone]
    expanded_obstacles = [expanded_dbbox(ref, distance=12) for ref in obstacles]
    assert not any(
        segment_crosses_box(p1, p2, box)
        for p1, p2 in pairwise(route_backbone)
        for box in expanded_obstacles
    )


def test_route_astar_accepts_cross_section_kwargs() -> None:
    c = gf.Component()
    straight = gf.components.straight(width=1)
    left = c << straight
    right = c << straight
    right.move((80, 0))

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o2"],
        port2=right.ports["o1"],
        cross_section="strip",
        width=1,
        resolution=10,
        distance=5,
        bend=gf.components.bend_euler,
    )

    assert route.length > 0


def test_route_astar_forwards_route_bundle_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_raise_on_error: list[bool | None] = []
    route_bundle = route_astar_module.gf.routing.route_bundle

    def wrapped_route_bundle(*args: object, **kwargs: object) -> object:
        raise_on_error = kwargs.get("raise_on_error")
        captured_raise_on_error.append(
            raise_on_error if isinstance(raise_on_error, bool) else None
        )
        return route_bundle(*args, **kwargs)

    monkeypatch.setattr(
        route_astar_module.gf.routing, "route_bundle", wrapped_route_bundle
    )

    c = gf.Component()
    straight = gf.components.straight(width=1)
    left = c << straight
    right = c << straight
    right.move((80, 0))

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o2"],
        port2=right.ports["o1"],
        cross_section="strip",
        width=1,
        raise_on_error=False,
        resolution=10,
        distance=5,
        bend=gf.components.bend_euler,
    )

    assert route.length > 0
    assert captured_raise_on_error[-1] is False


def test_route_astar_does_not_copy_input_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = gf.Component()
    straight = gf.components.straight()
    left = c << straight
    right = c << straight
    right.move((80, 0))

    def fail_copy() -> None:
        raise AssertionError("route_astar should not copy the full input component")

    monkeypatch.setattr(c, "copy", fail_copy)

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o2"],
        port2=right.ports["o1"],
        cross_section="strip",
        resolution=10,
        distance=5,
        bend=gf.components.bend_euler,
    )

    assert route.length > 0


def test_count_bends_skips_duplicate_points_and_handles_non_manhattan() -> None:
    waypoints = [
        gf.kdb.DPoint(0, 0),
        gf.kdb.DPoint(0, 0),
        gf.kdb.DPoint(1, 1),
        gf.kdb.DPoint(2, 2),
        gf.kdb.DPoint(2, 3),
    ]

    assert route_astar_module.count_bends(waypoints) == 1


def test_route_astar_selects_fewest_actual_bends() -> None:
    c = gf.Component()
    cross_section = gf.get_cross_section("strip", radius=5)
    straight = gf.components.straight(cross_section=cross_section)
    left = c << straight
    right = c << straight
    right.rotate(90)
    right.move((20, -50))

    route = gf.routing.route_astar(
        component=c,
        port1=left.ports["o2"],
        port2=right.ports["o1"],
        cross_section=cross_section,
        resolution=10,
        distance=8,
        bend=gf.components.bend_euler,
    )

    assert route.n_bend90 == 3


def test_route_astar_single_preserves_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_route_bundle(
        component: gf.Component,
        ports1: object = None,
        ports2: object = None,
        waypoints: object = None,
        cross_section: object = None,
        bend: object = "bend_euler",
        raise_on_error: bool | None = None,
        **kwargs: object,
    ) -> list[SimpleNamespace]:
        captured_kwargs.update(
            {
                "component": component,
                "ports1": ports1,
                "ports2": ports2,
                "waypoints": waypoints,
                "cross_section": cross_section,
                "bend": bend,
                "raise_on_error": raise_on_error,
                **kwargs,
            }
        )
        return [SimpleNamespace(length=1)]

    monkeypatch.setattr(
        route_astar_module.gf.routing, "route_bundle", fake_route_bundle
    )

    c = gf.Component()
    straight = gf.components.straight(width=1)
    left = c << straight
    right = c << straight
    right.move((80, 0))
    route = route_astar_module.route_astar_single(
        component=gf.Component(),
        port1=left.ports["o2"],
        port2=right.ports["o1"],
        cross_section="strip",
        width=1,
        layer="M2",
        radius=7,
        raise_on_error=True,
        resolution=10,
        blocked_grid=np.zeros((20, 5), dtype=bool),
        x=np.arange(0, 200, 10, dtype=float),
        y=np.arange(-20, 30, 10, dtype=float),
        start_node=(1, 2),
        end_node=(8, 2),
    )

    assert route.length == 1
    assert captured_kwargs["cross_section"].width == 1
    assert captured_kwargs["cross_section"].layer == "M2"
    assert captured_kwargs["cross_section"].radius == 7
    assert captured_kwargs["raise_on_error"] is True
    assert "width" not in captured_kwargs
    assert "layer" not in captured_kwargs
    assert "radius" not in captured_kwargs
