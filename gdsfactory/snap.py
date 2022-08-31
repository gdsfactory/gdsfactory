"""snaps values and coordinates to the GDS grid in nm."""
from typing import Tuple, Union

import numpy as np


def is_on_grid(x: float, nm: int = 1) -> bool:
    return np.isclose(snap_to_grid(x, nm=nm), x)


def assert_on_1nm_grid(x: float) -> None:
    x_grid = snap_to_grid(x)
    if not np.isclose(x_grid, x):
        raise ValueError(f"{x} needs to be on 1nm grid, try {x_grid}")


def assert_on_2nm_grid(x: float) -> None:
    x_grid = snap_to_2nm_grid(x)
    if not np.isclose(x_grid, x):
        raise ValueError(f"{x} needs to be on 2nm grid, try {x_grid}")


def snap_to_grid(
    x: Union[float, Tuple, np.ndarray], nm: int = 1
) -> Union[float, Tuple, np.ndarray]:
    if nm == 0:
        return x
    elif nm < 0:
        raise ValueError("nm must be an integer tolerance value greater than zero")
    elif nm == 1:
        y = np.round(np.asarray(x, dtype=float), 3)
    else:
        y = nm * np.round(np.asarray(x, dtype=float) * 1e3 / nm) / 1e3
    if isinstance(x, tuple):
        return tuple(y)
    elif isinstance(x, (int, float, str, np.float_)):
        return float(y)
    return y


def snap_to_2nm_grid(x: float) -> float:
    return snap_to_grid(x, nm=2)


def snap_to_5nm_grid(x: float) -> float:
    return snap_to_grid(x, nm=5)


if __name__ == "__main__":
    print(snap_to_grid(1.1e-3))
    # print(snap_to_2nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(3.1e-3))

    # print(on_1nm_grid(1.1e-3))
    # print(on_1nm_grid(1e-3))

    # print(on_2nm_grid(1.1e-3))
    # print(on_2nm_grid(1e-3))
    # print(on_2nm_grid(2e-3))
