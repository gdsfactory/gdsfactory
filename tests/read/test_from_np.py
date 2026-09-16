"""Tests for gdsfactory/read/from_np.py."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

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
    # area is outer contour - hole + the nested island. Absolute tolerance so
    # small geometry differences across skimage/KLayout versions don't flake.
    assert component.area((1, 0)) == pytest.approx(168.5 - 48.5 + 8.5, abs=1e-3)


def test_from_np_no_contours_raises() -> None:
    # all zeros below threshold -> no contours found
    arr = np.zeros((10, 10))
    with pytest.raises(AssertionError, match="no contours found"):
        from_np(arr, threshold=0.5)


def _concentric_array(widths: list[int]) -> np.ndarray:
    n = widths[0] + 2
    array = np.zeros((n, n))
    for depth, width in enumerate(widths):
        start = (n - width) // 2
        array[start : start + width, start : start + width] = (
            1.0 if depth % 2 == 0 else 0.0
        )
    return array


def _even_odd_area(widths: list[int]) -> float:
    # expected area in um^2 at 1 um per pixel: each contour of a width-wide
    # block encloses width^2 - 0.5 pixels, alternating solid/hole by depth
    return sum(
        (width * width - 0.5) if depth % 2 == 0 else -(width * width - 0.5)
        for depth, width in enumerate(widths)
    )


@pytest.mark.parametrize(
    "widths",
    [[19], [19, 13], [19, 13, 9], [19, 13, 9, 5], [19, 13, 9, 5, 3]],
    ids=["solid", "hole", "island", "island-with-hole", "double-island"],
)
def test_from_np_even_odd_nesting_preserved(widths: list[int]) -> None:
    component = from_np(_concentric_array(widths), nm_per_pixel=1_000, threshold=0.5)

    assert component.area((1, 0)) == pytest.approx(_even_odd_area(widths), abs=1e-3)
    assert len(component.insts) == 0


def test_from_np_no_invert_keeps_only_holes() -> None:
    widths = [19, 13, 9, 5, 3]
    component = from_np(
        _concentric_array(widths), nm_per_pixel=1_000, threshold=0.5, invert=False
    )

    # invert=False keeps only the hole contours, and each hole contour encloses
    # all deeper nesting, so the result is just the outermost hole's enclosure
    assert component.area((1, 0)) == pytest.approx(widths[1] ** 2 - 0.5, abs=1e-3)
    assert len(component.insts) == 0


def test_from_np_multiple_disjoint_blobs() -> None:
    array = np.zeros((150, 150))
    for row in range(3):
        for col in range(3):
            y0, x0 = row * 50, col * 50
            array[y0 + 5 : y0 + 45, x0 + 5 : x0 + 45] = 1
            array[y0 + 15 : y0 + 35, x0 + 15 : x0 + 35] = 0
            array[y0 + 22 : y0 + 28, x0 + 22 : x0 + 28] = 1

    component = from_np(array, threshold=0.5)

    blob_area = (40**2 - 0.5) - (20**2 - 0.5) + (6**2 - 0.5)
    assert component.area((1, 0)) == pytest.approx(
        9 * blob_area * (20 * 1e-3) ** 2, abs=1e-3
    )
    assert len(component.insts) == 0


@settings(deadline=None)
@given(
    outer=st.integers(min_value=8, max_value=60),
    gaps=st.lists(st.integers(min_value=4, max_value=12), max_size=4),
)
def test_from_np_random_nesting_matches_even_odd_area(
    outer: int, gaps: list[int]
) -> None:
    widths = [outer]
    for gap in gaps:
        widths.append(widths[-1] - gap)
    assume(widths[-1] >= 3)

    component = from_np(_concentric_array(widths), nm_per_pixel=1_000, threshold=0.5)

    assert component.area((1, 0)) == pytest.approx(_even_odd_area(widths), abs=1e-3)
