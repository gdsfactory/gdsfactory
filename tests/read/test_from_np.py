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


def test_from_np_no_contours_raises() -> None:
    # all zeros below threshold -> no contours found
    arr = np.zeros((10, 10))
    with pytest.raises(AssertionError, match="no contours found"):
        from_np(arr, threshold=0.5)
