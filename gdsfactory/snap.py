"""snaps values and coordinates to the GDS grid in nm."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias, TypeVar, overload

import kfactory as kf
import numpy as np
import numpy.typing as npt

Value: TypeAlias = float | Sequence[float] | npt.NDArray[np.floating[Any]]


def is_on_grid(
    x: Value,
    nm: int | None = None,
    grid_factor: int = 1,
) -> bool:
    return np.array_equal(
        snap_to_grid(x, grid_factor=grid_factor, nm=nm), np.round(x, 6)
    )


def warn_if_not_on_grid(x: Value) -> None:
    if not is_on_grid(x):
        warnings.warn(f"{x} is not on grid", stacklevel=3)


def assert_on_grid(
    x: Value,
    nm: int | None = None,
    grid_factor: int = 1,
) -> None:
    x_grid = snap_to_grid(x, nm=nm, grid_factor=grid_factor)
    if not np.isclose(x_grid, x, rtol=0).all():
        raise ValueError(f"{x} needs to be on 1nm grid and should be {x_grid}")


assert_on_1nm_grid = partial(assert_on_grid, nm=1)
assert_on_2nm_grid = partial(assert_on_grid, nm=2)


def assert_on_2x_grid(x: float) -> None:
    x_grid = snap_to_grid(x, grid_factor=2)
    if not np.isclose(x_grid, x, rtol=0).all():
        raise ValueError(f"{x} needs to be on 2x grid and should be {x_grid}")


_T = TypeVar("_T", bound=npt.NDArray[np.floating[Any]])


def snap_to_grid(
    x: _T,
    nm: int | None = None,
    grid_factor: int = 1,
) -> _T:
    """Snap x to grid.

    Args:
        x: value to snap.
        nm: Optional grid size in nm. If None, it will use the default grid size from PDK multiplied by grid_factor.
        grid_factor: snap to grid_factor * grid_size.
    """
    grid_size = kf.kcl.dbu
    nm = nm or round(grid_size * 1000 * grid_factor)

    # Fast path for scalar float or floating-point numpy scalar
    if isinstance(x, (float, np.floating)):
        snapped = nm * round(float(x) * 1e3 / nm) / 1e3
        return float(snapped)

    # Convert Sequence[float] and np.ndarray to np.ndarray and operate in bulk
    arr = np.asarray(x, dtype=float)
    snapped = nm * np.round(arr * 1e3 / nm) / 1e3
    return snapped


def snap_to_grid(
    x: Sequence[float],
    nm: int | None = None,
    grid_factor: int = 1,
) -> npt.NDArray[np.floating[Any]]:
    """Snap x to grid.

    Args:
        x: value to snap.
        nm: Optional grid size in nm. If None, it will use the default grid size from PDK multiplied by grid_factor.
        grid_factor: snap to grid_factor * grid_size.
    """
    grid_size = kf.kcl.dbu
    nm = nm or round(grid_size * 1000 * grid_factor)

    # Fast path for scalar float or floating-point numpy scalar
    if isinstance(x, (float, np.floating)):
        snapped = nm * round(float(x) * 1e3 / nm) / 1e3
        return float(snapped)

    # Convert Sequence[float] and np.ndarray to np.ndarray and operate in bulk
    arr = np.asarray(x, dtype=float)
    snapped = nm * np.round(arr * 1e3 / nm) / 1e3
    return snapped


def snap_to_grid(
    x: float,
    nm: int | None = None,
    grid_factor: int = 1,
) -> float:
    """Snap x to grid.

    Args:
        x: value to snap.
        nm: Optional grid size in nm. If None, it will use the default grid size from PDK multiplied by grid_factor.
        grid_factor: snap to grid_factor * grid_size.
    """
    grid_size = kf.kcl.dbu
    nm = nm or round(grid_size * 1000 * grid_factor)

    # Fast path for scalar float or floating-point numpy scalar
    if isinstance(x, (float, np.floating)):
        snapped = nm * round(float(x) * 1e3 / nm) / 1e3
        return float(snapped)

    # Convert Sequence[float] and np.ndarray to np.ndarray and operate in bulk
    arr = np.asarray(x, dtype=float)
    snapped = nm * np.round(arr * 1e3 / nm) / 1e3
    return snapped


def snap_to_grid(
    x: float | Sequence[float] | _T,
    nm: int | None = None,
    grid_factor: int = 1,
) -> _T | float:
    """Snap x to grid.

    Args:
        x: value to snap.
        nm: Optional grid size in nm. If None, it will use the default grid size from PDK multiplied by grid_factor.
        grid_factor: snap to grid_factor * grid_size.
    """
    grid_size = kf.kcl.dbu
    nm = nm or round(grid_size * 1000 * grid_factor)

    # Fast path for scalar float or floating-point numpy scalar
    if isinstance(x, (float, np.floating)):
        snapped = nm * round(float(x) * 1e3 / nm) / 1e3
        return float(snapped)

    # Convert Sequence[float] and np.ndarray to np.ndarray and operate in bulk
    arr = np.asarray(x, dtype=float)
    snapped = nm * np.round(arr * 1e3 / nm) / 1e3
    return snapped


@overload
def snap_to_grid2x(
    x: Sequence[float] | npt.NDArray[np.floating[Any]],
    nm: int | None = None,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def snap_to_grid2x(
    x: float,
    nm: int | None = None,
) -> float: ...
def snap_to_grid2x(
    x: Value,
    nm: int | None = None,
) -> npt.NDArray[np.floating[Any]] | float:
    return snap_to_grid(x, nm, grid_factor=2)


if __name__ == "__main__":
    assert_on_grid(1.1e-3)
    # print(snap_to_grid(1.1e-3))
    # print(snap_to_2nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(3.1e-3))

    # print(on_1nm_grid(1.1e-3))
    # print(on_1nm_grid(1e-3))

    # print(on_2nm_grid(1.1e-3))
    # print(on_2nm_grid(1e-3))
    # print(on_2nm_grid(2e-3))
