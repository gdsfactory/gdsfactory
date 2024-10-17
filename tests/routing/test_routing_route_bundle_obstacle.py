from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_obstacle(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    lengths = {}
    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    ptop.dmovex(300)
    ptop.dmovey(300)

    obstacle = c << gf.c.rectangle(size=(100, 100), layer="M3")
    obstacle.dymin = pbot.dymax
    obstacle.dxmin = pbot.dxmax + 10

    routes = gf.routing.route_bundle_electrical(
        c,
        reversed(pbot.ports),
        ptop.ports,
        start_straight_length=100,
        separation=20,
        bboxes=[obstacle.bbox()],  # can avoid obstacles
        cross_section=gf.cross_section.metal_routing,
    )

    for i, route in enumerate(routes):
        lengths[i] = route.length

    if check:
        data_regression.check(lengths)


if __name__ == "__main__":
    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    ptop.dmovex(300)
    ptop.dmovey(300)

    obstacle = c << gf.c.rectangle(size=(100, 100), layer="M3")
    obstacle.dymin = pbot.dymax
    obstacle.dxmin = pbot.dxmax + 10

    routes = gf.routing.route_bundle_electrical(
        c,
        reversed(pbot.ports),
        ptop.ports,
        start_straight_length=100,
        separation=20,
        bboxes=[obstacle.bbox()],  # can avoid obstacles
        cross_section=gf.cross_section.metal_routing,
    )
    c.show()
