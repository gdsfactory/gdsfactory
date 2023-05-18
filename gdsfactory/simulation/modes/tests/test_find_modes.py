from __future__ import annotations

import numpy as np

from gdsfactory.simulation.modes.find_modes import find_modes_waveguide


def test_find_modes_waveguide() -> None:
    modes = find_modes_waveguide(core_width=0.45, resolution=20, cache=None)
    m1 = modes[1]
    m2 = modes[2]

    # neff1 = 2.3815558509779744
    # neff2 = 1.7749644180250004

    neff1 = 2.3494603726390664
    neff2 = 1.7030929743774146

    assert np.isclose(m1.neff, neff1), m1.neff
    assert np.isclose(m2.neff, neff2), m2.neff

    # Using cache
    modes = find_modes_waveguide(core_width=0.45, resolution=20)
    m1 = modes[1]
    m2 = modes[2]

    assert np.isclose(m1.neff, neff1), m1.neff
    assert np.isclose(m2.neff, neff2), m2.neff


if __name__ == "__main__":
    test_find_modes_waveguide()
    # ms = get_mode_solver_rib(core_width=0.45)
    # modes = find_neff(mode_solver=ms)
    # m1 = modes[1]
    # m2 = modes[2]
    # print(m1.neff)
    # print(m2.neff)
    # neff1 = 2.342628111145838
    # neff2 = 1.7286034634949181

    # assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    # assert np.isclose(m2.neff, neff2), (m2.neff, neff2)
