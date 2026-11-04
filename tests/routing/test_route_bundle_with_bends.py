from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.routing.route_bundle_with_bends import route_bundle_with_bends


def test_route_bundle_with_bends_restarts_sequence_for_each_route() -> None:
    c = gf.Component()
    sources = [c << gf.components.straight(length=10) for _ in range(2)]
    targets = [c << gf.components.straight(length=10) for _ in range(2)]
    for i, source in enumerate(sources):
        source.dmovey(i * 50)
    for i, target in enumerate(targets):
        target.dmove((300, 100 + i * 50))
    bend_specs = [
        partial(gf.components.bend_euler, radius=5),
        partial(gf.components.bend_euler, radius=10),
    ]
    bend_names_expected = [
        gf.get_component(spec, cross_section="strip").name for spec in bend_specs
    ]

    routes = route_bundle_with_bends(
        c,
        [source.ports["o2"] for source in sources],
        [target.ports["o1"] for target in targets],
        bends=bend_specs,
        cross_section="strip",
        separation=5,
    )

    assert len(routes) == 2
    for route in routes:
        bend_names = [
            inst.cell.name
            for inst in route.instances
            if inst.cell.name.startswith("bend_")
        ]
        assert len(bend_names) >= 2
        assert bend_names[:2] == bend_names_expected


def test_route_bundle_with_bends_waypoints() -> None:
    c = gf.Component()
    source = c << gf.components.straight(length=10)
    target = c << gf.components.straight(length=10)
    target.dmove((400, 200))
    bend = gf.components.bend_euler(radius=10)

    routes = route_bundle_with_bends(
        c,
        [source.ports["o2"]],
        [target.ports["o1"]],
        bends=[bend],
        cross_section="strip",
        waypoints=[(100, 0), (100, 200), (300, 200)],
    )

    assert len(routes) == 1
    assert routes[0].n_bend90 >= 2
    assert routes[0].length > 0
