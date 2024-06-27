import numpy as np
import pytest

from gdsfactory.functions import polygon_grow


def test_polygon_grow_single_point():
    single_point_polygon = np.array([[1.0, 1.0]])
    offset = 0.5
    result = polygon_grow(single_point_polygon, offset)
    expected = np.array([[1.0, 1.0]])
    np.testing.assert_array_equal(result, expected)


def test_polygon_grow_empty_polygon():
    empty_polygon = np.array([])
    offset = 0.5
    result = polygon_grow(empty_polygon, offset)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_polygon_grow_valid_polygon():
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    offset = 0.5
    result = polygon_grow(polygon, offset)

    assert result.shape == polygon.shape


def test_polygon_grow_triangle():
    polygon = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    offset = 0.1
    result = polygon_grow(polygon, offset)
    assert result.shape == polygon.shape


def test_polygon_grow_line():
    polygon = np.array([[0, 0], [1, 0]])
    offset = 0.2
    result = polygon_grow(polygon, offset)
    assert result.shape == polygon.shape


if __name__ == "__main__":
    pytest.main()
