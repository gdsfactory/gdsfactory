"""
Compares the modes of a gdsfactory + MEEP waveguide cross-section vs a direct MPB calculation
"""
import numpy as np

import gdsfactory as gf
from gdsfactory.simulation.gmeep.get_sparameters import get_sparametersNxN


def test_sparameterNxN():
    """
    Checks that get_sparameterNxN properly sources, monitors, and sweeps over the ports
    Uses low resolution 2D simulations
    """

    c = gf.components.crossing()
    df = get_sparametersNxN(c, overwrite=True, animate=False)

    # Check reciprocity
    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:
                continue
            else:
                assert np.allclose(
                    df["s{}{}m".format(i, j)].to_numpy(),
                    df["s{}{}m".format(j, i)].to_numpy(),
                    atol=1e-02,
                )
                assert np.allclose(
                    df["s{}{}a".format(i, j)].to_numpy(),
                    df["s{}{}a".format(j, i)].to_numpy(),
                    atol=1e-02,
                )


if __name__ == "__main__":
    test_sparameterNxN()
