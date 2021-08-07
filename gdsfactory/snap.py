from typing import Tuple, Union

import numpy as np


def is_on_grid(x: float, nm: int = 1) -> bool:
    return np.isclose(snap_to_grid(x, nm=nm), x)


def assert_on_1nm_grid(x: float) -> None:
    assert np.isclose(snap_to_grid(x), x), f"{x} needs to be on 1nm grid"


def assert_on_2nm_grid(x: float) -> None:
    assert np.isclose(snap_to_2nm_grid(x), x), f"{x} needs to be on 1nm grid"


def snap_to_grid(
    x: Union[float, Tuple[float, float], np.array], nm: int = 1
) -> Union[float, np.array, Tuple[float, float]]:
    y = nm * np.round(np.array(x, dtype=float) * 1e3 / nm) / 1e3
    if isinstance(x, tuple):
        return tuple(y)
    elif type(x) in [int, float, str, np.float64]:
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
