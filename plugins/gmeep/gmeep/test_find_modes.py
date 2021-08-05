import numpy as np
import gmeep as gm


def test_find_modes():
    ms = gm.get_mode_solver_rib(wg_width=0.45)
    modes = gm.find_modes(mode_solver=ms)
    m1 = modes[1]
    neff1 = 2.3294688520397777
    ng1 = 4.079864977802547

    m2 = modes[2]
    neff2 = 2.3294688520397777
    ng2 = 4.079864977802547

    assert np.isclose(m1.meff, neff1), m1.neff
    assert np.isclose(m1.ng, ng1), m1.ng

    assert np.isclose(m2.meff, neff2), m2.neff
    assert np.isclose(m2.ng, ng2), m2.ng


if __name__ == "__main__":
    test_find_modes()
