import numpy as np

from gdsfactory.simulation.modes.find_mode_dispersion import find_mode_dispersion
from gdsfactory.simulation.modes.get_mode_solver_rib import get_mode_solver_rib


def test_find_modes_dispersion():
    ms = get_mode_solver_rib(wg_width=0.45)
    modes = find_mode_dispersion(mode_solver=ms, resolution=10)
    m1 = modes

    # print(f"neff1 = {m1.neff}")
    # print(f"ng1 = {m1.ng}")

    neff1 = 2.3502667681404428
    ng1 = 4.119391400983421

    assert np.isclose(m1.neff, neff1), (m1.neff, neff1)
    assert np.isclose(m1.ng, ng1), (m1.ng, ng1)


if __name__ == "__main__":
    test_find_modes_dispersion()
