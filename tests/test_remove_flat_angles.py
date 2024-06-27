import numpy as np
import pytest

from gdsfactory.functions import remove_flat_angles


def angles_deg(points):
    return np.array([0, 90, 180])


def test_remove_flat_angles_no_flat_angles():
    points = np.array([[0, 0], [1, 1], [2, 2]])
    result = remove_flat_angles(points)
    expected = np.array([[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(result, expected)


def test_remove_flat_angles_with_flat_angles():
    points = np.array([[0, 0], [1, 0], [2, 0], [2, 1]])
    result = remove_flat_angles(points)
    expected = np.array([[0, 0], [2, 0], [2, 1]])
    np.testing.assert_array_equal(result, expected)


def test_remove_flat_angles_with_angles_at_edges():
    points = np.array([[0, 0], [1, 0], [1, 1]])
    result = remove_flat_angles(points)
    expected = np.array([[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_remove_flat_angles_list_input():
    points = [[0, 0], [1, 0], [2, 0], [2, 1]]
    result = remove_flat_angles(points)
    expected = [[0, 0], [2, 0], [2, 1]]
    assert result == expected


if __name__ == "__main__":
    pytest.main()
