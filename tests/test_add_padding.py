import math

from hypothesis import given
from hypothesis import strategies as st

import gdsfactory as gf
from gdsfactory.components.bends.bend_circular import bend_circular


@given(
    padding=st.floats(
        min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
    )
)
def test_add_padding_size(padding: float) -> None:
    c = bend_circular().copy()
    xsize_before = c.xsize
    dxsize_before = c.dxsize
    ysize_before = c.ysize
    dysize_before = c.dysize

    c = gf.add_padding(c, default=padding)

    assert c.xsize > xsize_before
    assert c.dxsize > dxsize_before
    assert c.ysize > ysize_before
    assert c.dysize > dysize_before

    assert math.isclose(c.xsize - xsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dxsize - dxsize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.ysize - ysize_before, 2 * padding, abs_tol=0.01)
    assert math.isclose(c.dysize - dysize_before, 2 * padding, abs_tol=0.01)


@given(
    xsize_extra=st.floats(
        min_value=1, max_value=100, allow_nan=False, allow_infinity=False
    ),
    ysize_extra=st.floats(
        min_value=1, max_value=100, allow_nan=False, allow_infinity=False
    ),
)
def test_add_padding_to_size(xsize_extra: float, ysize_extra: float) -> None:
    c = bend_circular().copy()
    xsize = xsize_extra + c.xsize
    ysize = ysize_extra + c.ysize

    c = gf.add_padding_to_size(c, xsize=xsize, ysize=ysize)

    assert math.isclose(c.xsize, xsize, abs_tol=0.01)
    assert math.isclose(c.dxsize, xsize, abs_tol=0.01)
    assert math.isclose(c.ysize, ysize, abs_tol=0.01)
    assert math.isclose(c.dysize, ysize, abs_tol=0.01)
