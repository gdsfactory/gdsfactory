import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory.sp as sparameters
from gdsfactory.components import component_factory

component_types = [
    "straight",
    "bend_circular",
    # "bend_euler",
    "coupler",
    "mmi1x2",
    "mmi2x2",
]


@pytest.mark.parametrize("component_type", component_types)
def test_sp_load(
    component_type: str, data_regression: DataRegressionFixture, check: bool = True
) -> None:
    c = component_factory[component_type]()
    sp = sparameters.read_sparameters_component(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    if check:
        data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


if __name__ == "__main__":
    # c = gf.components.straight(layer=(2, 0))
    # print(c.get_sparameters_path())
    test_sp_load("straight", None, False)
