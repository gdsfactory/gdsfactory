from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.difftest import difftest


def test_get_route(data_regression: DataRegressionFixture, check: bool = True) -> None:
    c = gf.Component("sample_connect")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.move((100, 50))
    route = gf.routing.get_route(
        mmi1.ports["o3"],
        mmi2.ports["o1"],
        cross_section=gf.cross_section.metal1,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    if check:
        lengths = dict(length=route.length)

        data_regression.check(lengths)
        difftest(c)


def test_get_route_electrical_multilayer(
    data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = gf.Component("multi-layer")
    ptop = c << gf.components.pad_array()
    pbot = c << gf.components.pad_array(orientation=90)

    ptop.movex(300)
    ptop.movey(300)
    route = gf.routing.get_route_electrical_multilayer(
        ptop.ports["e11"],
        pbot.ports["e11"],
        end_straight_length=100,
    )
    c.add(route.references)
    if check:
        lengths = dict(length=route.length)

        data_regression.check(lengths)
        difftest(c)


if __name__ == "__main__":
    # c = gf.Component("sample_connect")
    # mmi1 = c << gf.components.mmi1x2()
    # mmi2 = c << gf.components.mmi1x2()
    # mmi2.move((100, 50))
    # route = gf.routing.get_route(
    #     mmi1.ports["o3"],
    #     mmi2.ports["o1"],
    #     cross_section=gf.cross_section.metal1,
    #     bend=gf.components.wire_corner,
    # )
    # c.add(route.references)
    # c.show(show_ports=True)

    c = gf.Component("multi-layer")
    ptop = c << gf.components.pad_array()
    pbot = c << gf.components.pad_array(orientation=90)

    ptop.movex(300)
    ptop.movey(300)
    route = gf.routing.get_route_electrical_multilayer(
        ptop.ports["e11"],
        pbot.ports["e11"],
        end_straight_length=100,
    )
    c.add(route.references)
    lengths = dict(length=route.length)
    c.show(show_ports=True)
