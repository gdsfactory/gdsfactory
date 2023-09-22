"""snaps values and coordinates to the GDS grid in nm."""
from __future__ import annotations

from functools import partial

import numpy as np


def is_on_grid(x: float, grid_factor: int = 1, nm: int = 1) -> bool:
    return np.array_equal(
        snap_to_grid(x, grid_factor=grid_factor, nm=nm), np.round(x, 6)
    )


assert_on_1nm_grid = partial(is_on_grid, nm=1)
assert_on_2nm_grid = partial(is_on_grid, nm=2)


def assert_on_grid(x: float) -> None:
    x_grid = snap_to_grid(x)
    if not np.isclose(x_grid, x):
        raise ValueError(f"{x} needs to be on 1nm grid and should be {x_grid}")


def assert_on_2x_grid(x: float) -> None:
    x_grid = snap_to_grid(x, grid_factor=2)
    if not np.isclose(x_grid, x):
        raise ValueError(f"{x} needs to be on 2x grid and should be {x_grid}")


def snap_to_grid(
    x: float | tuple | np.ndarray, grid_factor: int = 1, nm: int | None = None
) -> float | tuple | np.ndarray:
    """snap x to grid_sizes

    Args:
        x: value to snap.
        grid_factor: snap to grid_factor * grid_size.
        nm: Optional grid size in nm. If None, it will use the default grid size from PDK multiplied by grid_factor.
    """
    from gdsfactory.pdk import get_grid_size

    nm = nm or int(get_grid_size() * 1000 * grid_factor)
    y = nm * np.round(np.asarray(x, dtype=float) * 1e3 / nm) / 1e3

    if isinstance(x, tuple):
        return tuple(y)
    elif isinstance(x, int | float | str | np.float_):
        return float(y)
    return y


snap_to_grid2x = partial(snap_to_grid, grid_factor=2)


if __name__ == "__main__":
    print(assert_on_grid(1.1e-3))
    # print(snap_to_grid(1.1e-3))
    # print(snap_to_2nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(3.1e-3))

    # print(on_1nm_grid(1.1e-3))
    # print(on_1nm_grid(1e-3))

    # print(on_2nm_grid(1.1e-3))
    # print(on_2nm_grid(1e-3))
    # print(on_2nm_grid(2e-3))
