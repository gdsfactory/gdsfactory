import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.containers import containers
from gdsfactory.difftest import difftest

component = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")

skip_test = {
    "pack_doe",
    "pack_doe_grid",
}
container_names = set(containers.keys()) - skip_test


@pytest.mark.parametrize("container_type", container_names)
def test_settings(container_type: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    c = containers[container_type](component=component)
    data_regression.check(c.to_dict())


@pytest.mark.parametrize("container_type", container_names)
def test_gds(container_type: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    c = containers[container_type](component=component)
    difftest(c)


# Special test cases for exotic components
# def test_add_gratings_and_loopback(data_regression: DataRegressionFixture) -> None:
#     """This container requires all ports to face the same direction."""
#     c = add_gratings_and_loopback(component=spiral_inner_io())
#     data_regression.check(c.settings)


if __name__ == "__main__":
    component.show()
    # for i in container_names:
    #     print(i)
