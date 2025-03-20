from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf


def test_route_bundle_all_angle(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component()

    mmi = gf.components.mmi2x2(width_mmi=10, gap_mmi=3)
    mmi1 = c << mmi
    mmi2 = c << mmi

    mmi2.dmove((100, 10))
    mmi2.rotate(30)

    routes = gf.routing.route_bundle_all_angle(
        c,
        mmi1.ports.filter(orientation=0),
        [mmi2.ports["o2"], mmi2.ports["o1"]],
    )
    if check:
        lengths = {i: int(route.length) for i, route in enumerate(routes)}
        data_regression.check(lengths)


if __name__ == "__main__":
    c = gf.Component(name="demo")

    mmi = gf.components.mmi2x2(width_mmi=10, gap_mmi=3)
    mmi1 = c << mmi
    mmi2 = c << mmi

    mmi2.dmove((100, 10))
    mmi2.rotate(30)

    routes = gf.routing.route_bundle_all_angle(
        c,
        mmi1.ports.filter(orientation=0),
        [mmi2.ports["o2"], mmi2.ports["o1"]],
    )
    lengths = {i: int(route.length) for i, route in enumerate(routes)}
    print(lengths)
    c.show()
