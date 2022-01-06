"""test meep sparameters"""
import numpy as np

import gdsfactory as gf
from gdsfactory.simulation.gmeep.write_sparameters_meep import write_sparameters_meep


def test_sparameterNxN_straight(dataframe_regression):
    """Checks that computed transmission is reasonable
    to see if there are issues in get_simulation + transmission analysis
    """
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    df = write_sparameters_meep(c, overwrite=True, animate=False)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02)
    assert np.allclose(df["s21m"], 1, atol=1e-02)
    assert np.allclose(df["s11m"], 0, atol=5e-02)
    assert np.allclose(df["s22m"], 0, atol=5e-02)

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameterNxN_crossing(dataframe_regression):
    """Checks that get_sparameterNxN properly sources, monitors,
    and sweeps over the ports of all orientations
    Uses low resolution 2D simulations to run faster
    """
    # c = gf.components.crossing()
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    df = write_sparameters_meep(c, overwrite=True, animate=False)

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
    if dataframe_regression:
        dataframe_regression.check(df)


if __name__ == "__main__":
    test_sparameterNxN_crossing(None)
