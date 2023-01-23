from __future__ import annotations

import numpy as np

import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.simulation.gtidy3d.modes import Waveguide, group_index

nm = 1e-3
ncore = "si"
nclad = "sio2"


def test_neff_cached():
    c = Waveguide(
        wavelength=1.55,
        wg_width=0.5,
        wg_thickness=0.22,
        slab_thickness=0.0,
        ncore=ncore,
        nclad=nclad,
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
        ncore=ncore,
        nclad=nclad,
        cache=False,
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
        ncore=ncore,
        nclad=nclad,
        cache=False,
    )
    ng = group_index(**wg_settings)
    assert np.isclose(ng, 4.169, rtol=0.01), ng


def test_sweep_width(dataframe_regression) -> None:
    df = gt.modes.sweep_width(
        width1=200 * nm,
        width2=1000 * nm,
        steps=1,
        wavelength=1.55,
        wg_thickness=220 * nm,
        slab_thickness=0 * nm,
        ncore=ncore,
        nclad=nclad,
        cache=False,
    )

    if dataframe_regression:
        dataframe_regression.check(df, default_tolerance=dict(atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    test_ng_no_cache()
