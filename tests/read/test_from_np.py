"""Tests for gdsfactory/read/from_np.py."""

import numpy as np
import pytest

from gdsfactory.read.from_np import compute_area_signed, from_np


def test_compute_area_signed_counterclockwise_positive() -> None:
    # CCW unit square: area > 0
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    assert compute_area_signed(ring) > 0


def test_compute_area_signed_clockwise_negative() -> None:
    ring = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    assert compute_area_signed(ring) < 0


def test_from_np_returns_component_with_polygons() -> None:
    # build a 20x20 array with a filled 10x10 block in the middle
    arr = np.zeros((20, 20))
    arr[5:15, 5:15] = 1.0
    c = from_np(arr, threshold=0.5, invert=True)
    assert c is not None
    polys = c.get_polygons()
    assert sum(len(v) for v in polys.values()) >= 1


def test_from_np_preserves_island_inside_hole() -> None:
    array = np.zeros((15, 15))
    array[1:14, 1:14] = 1
    array[4:11, 4:11] = 0
    array[6:9, 6:9] = 1

    component = from_np(array, nm_per_pixel=1_000, threshold=0.5)

    # Marching squares places the contours halfway between pixels: the expected
    # area is outer contour - hole + the nested island.
    assert component.area((1, 0)) == pytest.approx(168.5 - 48.5 + 8.5)


def test_from_np_no_contours_raises() -> None:
    # all zeros below threshold -> no contours found
    arr = np.zeros((10, 10))
    with pytest.raises(AssertionError, match="no contours found"):
        from_np(arr, threshold=0.5)
