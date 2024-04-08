from __future__ import annotations

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
