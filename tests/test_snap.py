from __future__ import annotations

import pytest

import gdsfactory as gf


def test_snap_to_grid() -> None:
    assert gf.snap.snap_to_grid(1.1e-3) == 0.001


def test_snap_to_2nm_grid() -> None:
    assert gf.snap.snap_to_grid2x(1.1e-3) == 0.002
    assert gf.snap.snap_to_grid2x(3.1e-3) == 0.004


def test_is_on_1x_grid() -> None:
    assert not gf.snap.is_on_grid(0.1e-3)
    assert gf.snap.is_on_grid(1e-3)


def test_is_on_2x_grid() -> None:
    assert not gf.snap.is_on_grid(1.1e-3, nm=2)
    assert not gf.snap.is_on_grid(1e-3, nm=2)
    assert gf.snap.is_on_grid(2e-3, nm=2)


def test_point_is_on_grid() -> None:
    assert gf.snap.is_on_grid([0.5555, 0]) is False
    assert gf.snap.is_on_grid([0.555, 0]) is True


def test_point_is_on_2x_grid() -> None:
    assert gf.snap.is_on_grid([0.555, 0], grid_factor=2) is False
    assert gf.snap.is_on_grid([0.556, 0], grid_factor=2) is True


def test_snap_polygon() -> None:
    c1 = gf.Component()
    c1.add_polygon(
        [(-8.0005, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0), snap_to_grid=True
    )
    gf.snap.assert_on_1nm_grid(c1.polygons[0].points[0][0])

    c2 = gf.Component()
    p = c2.add_polygon(
        [(-8.0005, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0), snap_to_grid=False
    )
    p = p.snap()
    gf.snap.assert_on_1nm_grid(p.points[0][0])


def test_no_snap_polygon() -> None:
    c1 = gf.Component()
    c1.add_polygon(
        [(-8.0005, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0), snap_to_grid=False
    )
    with pytest.raises(ValueError):
        gf.snap.assert_on_1nm_grid(c1.polygons[0].points[0][0])


if __name__ == "__main__":
    test_snap_to_grid()
    # test_no_snap_polygon()
    # c2 = gf.Component()
    # p = c2.add_polygon(
    #     [(-8.0005, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0), snap_to_grid=False
    # )
    # p = p.snap()
    # gf.snap.assert_on_1nm_grid(p.points[0][0])
