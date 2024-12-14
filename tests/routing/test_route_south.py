from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_south(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()
    cr = c << gf.components.mmi2x2()
    routes = gf.routing.route_south(c, cr)

    lengths: dict[int, int] = {}
    for i, route in enumerate(routes):
        lengths[i] = route.length
    if check:
        data_regression.check(lengths)  # type: ignore


if __name__ == "__main__":
    test_route_south(None, check=False)  # type: ignore
