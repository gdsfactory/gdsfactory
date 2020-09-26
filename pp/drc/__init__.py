""" design rule checking """
import numpy as np
from numpy import bool_, float64
from typing import Union


def on_grid(x: float64, nm: int = 1) -> bool_:
    return np.isclose(snap_to_grid(x, nm=nm), x)


def on_1nm_grid(x):
    return np.isclose(snap_to_1nm_grid(x), x)


def on_2nm_grid(x):
    return np.isclose(snap_to_2nm_grid(x), x)


def assert_on_1nm_grid(x: float) -> None:
    assert np.isclose(snap_to_1nm_grid(x), x), f"{x} needs to be on 1nm grid"


def assert_on_2nm_grid(x: float) -> None:
    assert np.isclose(snap_to_2nm_grid(x), x), f"{x} needs to be on 1nm grid"


def snap_to_grid(x: Union[float64, float], nm: int = 1) -> float64:
    return nm * np.round(np.array(x) * 1e3 / nm) / 1e3


def snap_to_1nm_grid(x: float) -> float64:
    return snap_to_grid(x, nm=1)


def snap_to_2nm_grid(x: float) -> float64:
    return snap_to_grid(x, nm=2)


def snap_to_5nm_grid(x: float64) -> float64:
    return snap_to_grid(x, nm=5)


def test_snap_to_1nm_grid():
    assert snap_to_1nm_grid(1.1e-3) == 0.001


def test_snap_to_2nm_grid():
    assert snap_to_2nm_grid(1.1e-3) == 0.002
    assert snap_to_2nm_grid(3.1e-3) == 0.004


def test_on_1nm_grid():
    assert not on_1nm_grid(1.1e-3)
    assert on_1nm_grid(1e-3)


def test_on_2nm_grid():
    assert not on_2nm_grid(1.1e-3)
    assert not on_2nm_grid(1e-3)
    assert on_2nm_grid(2e-3)


if __name__ == "__main__":
    # print(snap_to_1nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(3.1e-3))

    # print(on_1nm_grid(1.1e-3))
    # print(on_1nm_grid(1e-3))

    print(on_2nm_grid(1.1e-3))
    print(on_2nm_grid(1e-3))
    print(on_2nm_grid(2e-3))
