from __future__ import annotations

from functools import partial

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.difftest import difftest
from gdsfactory.generic_tech.containers import containers

component = gf.components.mzi2x2_2x2(straight_x_top="straight_heater_metal")

skip_test = {"add_fiber_array", "add_termination"}

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


add_gratings_with_loopback = (
    gf.components.add_grating_couplers_with_loopback_fiber_array
)
add_gratings = gf.components.add_grating_couplers

spiral = partial(
    gf.c.spiral_inner_io,
    decorator=add_gratings,
)

spiral_loopback = partial(
    gf.c.spiral_inner_io,
    decorator=add_gratings_with_loopback,
)

# def test_container_double_decorator(test=True) -> None:
#     """Avoid regressions when exporting settings."""
#     c = spiral()
#     if test:
#         difftest(c)

#     c = spiral_loopback()
#     if test:
#         difftest(c)


if __name__ == "__main__":
    c = spiral()
    c.show()
    # for i in container_names:
    #     print(i)
