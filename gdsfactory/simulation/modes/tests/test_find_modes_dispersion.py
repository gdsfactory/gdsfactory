from __future__ import annotations

import numpy as np

from gdsfactory.simulation.modes.find_mode_dispersion import find_mode_dispersion


def test_find_modes_waveguide_dispersion() -> None:
    modes = find_mode_dispersion(core_width=0.45, resolution=20, cache=None)
    m1 = modes

    # print(f"neff1 = {m1.neff}")
    # print(f"ng1 = {m1.ng}")

    # neff1 = 2.3948
    # ng1 = 4.23194

    neff1 = 2.362907833437435
    ng1 = 4.202169359808116

    assert np.isclose(m1.neff, neff1, rtol=1e-2), (m1.neff, neff1)
    assert np.isclose(m1.ng, ng1, rtol=1e-2), (m1.ng, ng1)


if __name__ == "__main__":
    test_find_modes_waveguide_dispersion()
