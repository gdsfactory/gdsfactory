""" design rule checking """
import numpy as np


def on_1nm_grid(x):
    return snap_to_1nm_grid(x) == x


def on_2nm_grid(x):
    return snap_to_2nm_grid(x) == x


def assert_on_1nm_grid(x):
    if snap_to_1nm_grid(x) != x:
        raise ValueError("{} needs to be on 1nm grid".format(x))


def assert_on_2nm_grid(x):
    if snap_to_2nm_grid(x) != x:
        raise ValueError("{} needs to be on 2nm grid".format(x))


def snap_to_1nm_grid(x):
    return np.round(x * 1e3) / 1e3


def snap_to_2nm_grid(x):
    return 2 * np.round(x * 1e3 / 2) / 1e3


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
