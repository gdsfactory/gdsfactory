import gdsfactory


def test_snap_to_grid() -> None:
    assert gdsfactory.snap.snap_to_grid(1.1e-3) == 0.001


def test_snap_to_2nm_grid() -> None:
    assert gdsfactory.snap.snap_to_2nm_grid(1.1e-3) == 0.002
    assert gdsfactory.snap.snap_to_2nm_grid(3.1e-3) == 0.004


def test_is_on_1nm_grid() -> None:
    assert not gdsfactory.snap.is_on_grid(0.1e-3)
    assert gdsfactory.snap.is_on_grid(1e-3)


def test_is_on_2nm_grid() -> None:
    assert not gdsfactory.snap.is_on_grid(1.1e-3, 2)
    assert not gdsfactory.snap.is_on_grid(1e-3, 2)
    assert gdsfactory.snap.is_on_grid(2e-3, 2)


if __name__ == "__main__":
    test_is_on_2nm_grid()
    # print(snap_to_grid(1.1e-3))
    # print(snap_to_2nm_grid(1.1e-3))
    # print(snap_to_2nm_grid(3.1e-3))

    # print(on_1nm_grid(1.1e-3))
    # print(on_1nm_grid(1e-3))

    # print(on_2nm_grid(1.1e-3))
    # print(on_2nm_grid(1e-3))
    # print(on_2nm_grid(2e-3))
