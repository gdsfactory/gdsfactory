import numpy as np

from gdsfactory.simulation.modes.find_modes import find_modes_waveguide


def test_find_modes_waveguide():
    modes = find_modes_waveguide(wg_width=0.45, resolution=20)
    m1 = modes[1]
    m2 = modes[2]
    # print(f"neff1 = {m1.neff}")
    # print(f"neff2 = {m2.neff}")

    neff1 = 2.3815558509779744
    neff2 = 1.7749644180250004

    assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    assert np.isclose(m2.neff, neff2), (m2.neff, neff2)


if __name__ == "__main__":
    test_find_modes_waveguide()
    # ms = get_mode_solver_rib(wg_width=0.45)
    # modes = find_neff(mode_solver=ms)
    # m1 = modes[1]
    # m2 = modes[2]
    # print(m1.neff)
    # print(m2.neff)
    # neff1 = 2.342628111145838
    # neff2 = 1.7286034634949181

    # assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    # assert np.isclose(m2.neff, neff2), (m2.neff, neff2)
