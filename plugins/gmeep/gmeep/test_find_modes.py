import numpy as np

from gmeep.find_modes import find_modes


def test_find_modes():
    r = find_modes(wg_width=0.45)
    neff = r["neff"]
    ng = r["ng"]
    print(neff, ng)
    assert np.isclose(neff, 2.31169419861862)
    assert np.isclose(ng, 4.076561522948648)


if __name__ == "__main__":
    test_find_modes()
