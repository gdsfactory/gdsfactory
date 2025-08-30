from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Component, Port
from gdsfactory.typings import AngleInDegrees, Delta, Layer


def test_connect_bundle_u_indirect(
    data_regression: DataRegressionFixture,
    dy: Delta = -200,
    orientation: AngleInDegrees = 180,
    layer: Layer = (1, 0),
    check: bool = True,
) -> None:
    """Test routing a bundle of ports with indirect connection.

    Args:
        data_regression: regression test fixture.
        dy: vertical offset.
        orientation: orientation of the ports.
        layer: layer of the ports.
        check: check the output.
    """
    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]
    axis = "X" if orientation in [0, 180] else "Y"
    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = orientation
    a2 = a1 + 180

    if axis == "X":
        ports1 = [
            Port(
                name=f"top_{i}",
                center=(0, xs1[i]),
                width=0.5,
                orientation=a1,
                layer=gf.kcl.layout.layer(*layer),
            )
            for i in range(N)
        ]

        ports2 = [
            Port(
                name=f"bot_{i}",
                center=(dy, xs2[i]),
                width=0.5,
                orientation=a2,
                layer=gf.kcl.layout.layer(*layer),
            )
            for i in range(N)
        ]

    else:
        ports1 = [
            Port(
                name=f"top_{i}",
                center=(xs1[i], 0),
                width=0.5,
                orientation=a1,
                layer=gf.kcl.layout.layer(*layer),
            )
            for i in range(N)
        ]

        ports2 = [
            Port(
                name=f"bot_{i}",
                center=(xs2[i], dy),
                width=0.5,
                orientation=a2,
                layer=gf.kcl.layout.layer(*layer),
            )
            for i in range(N)
        ]

    c = Component()
    routes = gf.routing.route_bundle(
        c,
        ports1,
        ports2,
        bend=gf.components.bend_euler,
        radius=5,
        sort_ports=True,
        cross_section=gf.cross_section.strip,
    )
    lengths = {i: route.length for i, route in enumerate(routes)}
    if check:
        data_regression.check(lengths)
