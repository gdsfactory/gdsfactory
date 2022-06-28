import numpy as np

from gdsfactory.simulation.gtidy3d.modes import Waveguide, si, sio2


def test_index():
    c = Waveguide(
        wavelength=1.55,
        wg_width=0.5,
        wg_thickness=0.22,
        slab_thickness=0.0,
        ncore=si,
        nclad=sio2,
    )
    c.compute_modes()
    n0 = abs(c.neffs[0])
    assert np.isclose(n0, 2.46586, rtol=0.01), n0


if __name__ == "__main__":
    test_index()
