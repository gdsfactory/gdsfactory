from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_single(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((100, 50))
    route = gf.routing.route_single(
        c,
        mmi1.ports["o3"],
        mmi2.ports["o1"],
        cross_section=gf.cross_section.strip,
    )
    if check:
        lengths = dict(length=route.length)
        data_regression.check(lengths)
