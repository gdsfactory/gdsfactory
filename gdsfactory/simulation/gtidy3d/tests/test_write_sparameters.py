from __future__ import annotations

import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt


def test_sparameters_straight(overwrite=False) -> None:
    """Checks Sparameters for a straight waveguide in 2D."""
    c = gf.components.straight(length=2)
    sp = gt.write_sparameters_1x1(c, overwrite=overwrite, is_3d=False)

    # Check reasonable reflection/transmission
    assert np.abs(sp["o1@0,o2@0"]).min() > 0.89, np.abs(sp["o1@0,o2@0"]).min()
    assert np.abs(sp["o1@0,o1@0"]).max() < 0.1, np.abs(sp["o1@0,o1@0"]).max()

    # assert np.allclose(sp["o2@0,o1@0"], 1, atol=1e-02), sp["o2@0,o1@0"]
    # assert np.allclose(sp["o1@0,o1@0"], 0, atol=5e-02), sp["o1@0,o1@0"]
    # assert np.allclose(sp["o2@0,o2@0"], 0, atol=5e-02), sp["o2@0,o2@0"]

    # if dataframe_regression:
    #     dataframe_regression.check(sp)


if __name__ == "__main__":
    test_sparameters_straight()

    # c = gf.components.straight(length=2)
    # sp = gt.write_sparameters_1x1(c, overwrite=True)

    # # Check reasonable reflection/transmission
    # assert np.allclose(sp["o1@0,o2@0"], 1, atol=1e-02), sp["o1@0,o2@0"]
    # assert np.allclose(sp["o2@0,o1@0"], 1, atol=1e-02), sp["o2@0,o1@0"]
    # assert np.allclose(sp["o1@0,o1@0"], 0, atol=5e-02), sp["o1@0,o1@0"]
    # assert np.allclose(sp["o2@0,o2@0"], 0, atol=5e-02), sp["o2@0,o2@0"]
