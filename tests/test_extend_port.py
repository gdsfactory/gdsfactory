import numpy as np
from numpy.testing import assert_array_almost_equal

from gdsfactory.port import Port


def test_custom_length() -> None:
    port = Port(
        name="o1", center=np.array([0, 0]), orientation=45, width=1, layer=(1, 0)
    )
    expected = np.array([0.70710678, 0.70710678])
    result = port.get_extended_center(length=1)
    assert_array_almost_equal(result, expected, decimal=7)


def test_changed_orientation() -> None:
    port = Port(
        name="o1", center=np.array([0, 0]), orientation=90, width=1, layer=(1, 0)
    )
    expected = np.array([0, 1])
    result = port.get_extended_center()
    assert_array_almost_equal(result, expected, decimal=7)


def test_zero_length() -> None:
    port = Port(
        name="o1", center=np.array([0, 0]), orientation=90, width=1, layer=(1, 0)
    )
    expected = np.array([0, 0])
    result = port.get_extended_center(length=0)
    assert_array_almost_equal(result, expected, decimal=7)


if __name__ == "__main__":
    # test_custom_length()
    # test_changed_orientation()
    test_zero_length()
