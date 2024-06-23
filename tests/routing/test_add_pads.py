from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.difftest import difftest


@gf.cell
def add_pads0() -> Component:
    c = gf.components.straight_heater_metal(length=101.0)
    c = gf.routing.add_pads_bot(
        component=c, port_names=("l_e4", "r_e4"), fanout_length=80
    )
    return c


components = [add_pads0]


@pytest.fixture(params=components, scope="function")
def component(request) -> Component:
    return request.param()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


if __name__ == "__main__":
    c = add_pads0()
    c.show()
