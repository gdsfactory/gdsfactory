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


def test_snap_to_grid_with_explicit_nm() -> None:
    assert gf.snap.snap_to_grid(0.0015, nm=1) == 0.002
    assert gf.snap.snap_to_grid(0.0015, nm=1, grid_factor=2) == 0.002
    assert gf.snap.snap_to_grid(0.0015, nm=2, grid_factor=1) == 0.002

    assert gf.snap.snap_to_grid(-0.0015, nm=1) == -0.002
    assert gf.snap.snap_to_grid(-0.0015, nm=1, grid_factor=2) == -0.002
    assert gf.snap.snap_to_grid(-0.0015, nm=2, grid_factor=1) == -0.002


def test_snap_to_grid_sub_nm_dbu() -> None:
    assert gf.snap.snap_to_grid(0.00074, nm=0.5) == 0.0005
    assert gf.snap.snap_to_grid(0.00076, nm=0.5) == 0.001
    assert gf.snap.snap_to_grid(0.00126, nm=0.5) == 0.0015

    assert gf.snap.snap_to_grid(0.00014, nm=0.1) == 0.0001
    assert gf.snap.snap_to_grid(0.00016, nm=0.1) == 0.0002
