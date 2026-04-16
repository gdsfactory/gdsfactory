"""Tests for gdsfactory/component_layout.py geometry helpers."""

import numpy as np
import pytest

from gdsfactory.component_layout import (
    parse_coordinate,
    parse_move,
    reflect_points,
    rotate_points,
)


def test_rotate_points_zero_angle_is_noop() -> None:
    pts = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = rotate_points(pts, angle=0)
    np.testing.assert_array_equal(out, pts)


def test_rotate_points_90_around_origin() -> None:
    pts = np.array([[1.0, 0.0]])
    out = rotate_points(pts, angle=90)
    np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-12)


def test_rotate_points_around_center() -> None:
    pts = np.array([[2.0, 1.0]])
    out = rotate_points(pts, angle=180, center=(1.0, 1.0))
    np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-12)


def test_rotate_points_single_1d() -> None:
    pt = np.array([1.0, 0.0])
    out = rotate_points(pt, angle=90)
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_rotate_points_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="array-like"):
        rotate_points(np.zeros((2, 2, 2)), angle=10)


def test_reflect_points_across_x_axis() -> None:
    pts = np.array([[1.0, 2.0], [3.0, -4.0]])
    out = reflect_points(pts, p1=(0, 0), p2=(1, 0))
    np.testing.assert_allclose(out, [[1.0, -2.0], [3.0, 4.0]], atol=1e-12)


def test_reflect_points_across_y_axis() -> None:
    pts = np.array([[1.0, 2.0]])
    out = reflect_points(pts, p1=(0, 0), p2=(0, 1))
    # single-row input: function returns the row, not a 2D array
    np.testing.assert_allclose(out, [-1.0, 2.0], atol=1e-12)


def test_parse_coordinate_tuple() -> None:
    assert parse_coordinate((1.5, 2.5)) == (1.5, 2.5)


def test_parse_coordinate_list() -> None:
    assert parse_coordinate([1.0, 2.0]) == (1.0, 2.0)


def test_parse_coordinate_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Could not parse coordinate"):
        parse_coordinate([1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_parse_move_origin_only_treats_as_destination() -> None:
    dx, dy = parse_move((3.0, 4.0), destination=None)
    assert (dx, dy) == (3.0, 4.0)


def test_parse_move_with_destination() -> None:
    dx, dy = parse_move((1.0, 1.0), (4.0, 5.0))
    assert (dx, dy) == (3.0, 4.0)


def test_parse_move_axis_x_zeroes_y() -> None:
    dx, dy = parse_move((0.0, 0.0), (3.0, 4.0), axis="x")
    assert (dx, dy) == (3.0, 0.0)


def test_parse_move_axis_y_zeroes_x() -> None:
    dx, dy = parse_move((0.0, 0.0), (3.0, 4.0), axis="y")
    assert (dx, dy) == (0.0, 4.0)
