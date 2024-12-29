import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import gdsfactory as gf


@pytest.mark.parametrize("component_type", ["bend_circular", "bend_euler"])
@given(
    padding=st.decimals(min_value=5, max_value=100, places=4).map(float),
)
@settings(deadline=None)
def test_add_padding_size(padding: float, component_type: str) -> None:
    c = gf.get_component(component_type)
    xsize_before = c.xsize
    dxsize_before = c.dxsize
    ysize_before = c.ysize
    dysize_before = c.dysize

    c = gf.add_padding(c, default=padding)

    # Check that padding was added
    assert c.xsize > xsize_before
    assert c.dxsize > dxsize_before
    assert c.ysize > ysize_before
    assert c.dysize > dysize_before

    # Check that padding was added symmetrically on both sides
    assert math.isclose(c.xsize - xsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dxsize - dxsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.ysize - ysize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dysize - dysize_before, 2 * padding, abs_tol=0.01)


@pytest.mark.parametrize("component_type", ["bend_circular", "bend_euler"])
@given(
    xsize=st.decimals(min_value=5, max_value=100, places=4).map(float),
    ysize=st.decimals(min_value=5, max_value=100, places=4).map(float),
)
@settings(deadline=None)
def test_add_padding_to_size(xsize: float, ysize: float, component_type: str) -> None:
    c = gf.get_component(component_type)
    xsize += c.xsize
    ysize += c.ysize

    c = gf.add_padding_to_size(c, xsize=xsize, ysize=ysize)

    assert math.isclose(c.xsize, xsize, abs_tol=0.01)
    assert math.isclose(c.dxsize, xsize, abs_tol=0.01)
    assert math.isclose(c.ysize, ysize, abs_tol=0.01)
    assert math.isclose(c.dysize, ysize, abs_tol=0.01)


if __name__ == "__main__":
    test_add_padding_size()
    test_add_padding_to_size()
