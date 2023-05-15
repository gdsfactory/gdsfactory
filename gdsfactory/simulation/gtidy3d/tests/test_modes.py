from __future__ import annotations

import numpy as np

import gdsfactory.simulation.gtidy3d as gt


def test_neff() -> None:
    wg = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=0.5,
        core_thickness=0.22,
        core_material="si",
        clad_material="sio2",
        cache=False,
    )
    n_eff = wg.n_eff[0].real
    assert np.isclose(n_eff, 2.447, rtol=0.1), n_eff


def test_neff_high_accuracy() -> None:
    wg = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=0.5,
        core_thickness=0.22,
        core_material="si",
        clad_material="sio2",
        grid_resolution=40,
        cache=False,
    )
    n_eff = wg.n_eff[0].real
    assert np.isclose(n_eff, 2.447, rtol=0.01), n_eff
