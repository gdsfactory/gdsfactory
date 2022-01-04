"""test meep sparameters"""
import numpy as np

import gdsfactory as gf
from gdsfactory.simulation.gmeep.get_sparameters import get_sparametersNxN


def test_sparameterNxN(dataframe_regression):
    """
    Checks that get_sparameterNxN properly sources, monitors, and sweeps over the ports
    Uses low resolution 2D simulations to run faster
    """
    # c = gf.components.crossing()
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    df = get_sparametersNxN(c, overwrite=True, animate=False)

    # Check reciprocity
    for i in range(1, len(c.ports) + 1):
        for j in range(1, len(c.ports) + 1):
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
    dataframe_regression.check(df)


if __name__ == "__main__":
    test_sparameterNxN()
