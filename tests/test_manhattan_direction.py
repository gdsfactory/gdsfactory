import numpy as np

from gdsfactory.functions import manhattan_direction


def test_manhattan_direction_positive_x():
    p0 = np.array([0, 0])
    p1 = np.array([1, 0])
    result = manhattan_direction(p0, p1)
    expected = np.array([1, 0])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_negative_x():
    p0 = np.array([1, 0])
    p1 = np.array([0, 0])
    result = manhattan_direction(p0, p1)
    expected = np.array([-1, 0])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_positive_y():
    p0 = np.array([0, 0])
    p1 = np.array([0, 1])
    result = manhattan_direction(p0, p1)
    expected = np.array([0, 1])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_negative_y():
    p0 = np.array([0, 1])
    p1 = np.array([0, 0])
    result = manhattan_direction(p0, p1)
    expected = np.array([0, -1])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_zero_x():
    p0 = np.array([0, 0])
    p1 = np.array([0, 0.000001])
    result = manhattan_direction(p0, p1)
    expected = np.array([0, 1])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_zero_y():
    p0 = np.array([0, 0])
    p1 = np.array([0.000001, 0])
    result = manhattan_direction(p0, p1)
    expected = np.array([1, 0])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_small_tol():
    p0 = np.array([0, 0])
    p1 = np.array([1e-6, 0])
    result = manhattan_direction(p0, p1, tol=1e-7)
    expected = np.array([1, 0])
    np.testing.assert_array_equal(result, expected)


def test_manhattan_direction_large_tol():
    p0 = np.array([0, 0])
    p1 = np.array([1e-6, 0])
    result = manhattan_direction(p0, p1, tol=1e-5)
    expected = np.array([0, 0])
    np.testing.assert_array_equal(result, expected)
