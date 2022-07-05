import numpy as np

from gdsfactory.simulation.gtidy3d.modes import Waveguide, group_index, si, sio2


def test_neff_cached():
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


def test_neff_no_cache():
    c = Waveguide(
        wavelength=1.55,
        wg_width=0.5,
        wg_thickness=0.22,
        slab_thickness=0.0,
        ncore=si,
        nclad=sio2,
        cache=None,
    )
    c.compute_modes()
    n0 = abs(c.neffs[0])
    assert np.isclose(n0, 2.46586, rtol=0.01), n0


def test_ng_no_cache():
    wg_settings = dict(
        wavelength=1.55,
        wg_width=0.5,
        wg_thickness=0.22,
        slab_thickness=0.0,
        ncore=si,
        nclad=sio2,
        cache=None,
    )
    ng = group_index(**wg_settings)
    assert np.isclose(ng, 4.169, rtol=0.01), ng


if __name__ == "__main__":
    test_ng_no_cache()
