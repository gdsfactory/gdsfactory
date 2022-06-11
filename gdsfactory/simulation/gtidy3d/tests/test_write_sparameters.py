import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt


def test_sparameters_straight() -> None:
    """Checks Sparameters for a straight waveguide in 2D."""
    c = gf.components.straight(length=2)
    df = gt.write_sparameters_1x1(c, overwrite=True, is_3d=False)

    # Check reasonable reflection/transmission
    assert df["s12m"].min() > 0.89, df["s12m"].min()
    assert df["s11m"].max() < 0.1, df["s11m"].max()

    # assert np.allclose(df["s21m"], 1, atol=1e-02), df["s21m"]
    # assert np.allclose(df["s11m"], 0, atol=5e-02), df["s11m"]
    # assert np.allclose(df["s22m"], 0, atol=5e-02), df["s22m"]

    # if dataframe_regression:
    #     dataframe_regression.check(df)


if __name__ == "__main__":
    # test_sparameters_straight(None)

    c = gf.components.straight(length=2)
    df = gt.write_sparameters_1x1(c, overwrite=True)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02), df["s12m"]
    assert np.allclose(df["s21m"], 1, atol=1e-02), df["s21m"]
    assert np.allclose(df["s11m"], 0, atol=5e-02), df["s11m"]
    assert np.allclose(df["s22m"], 0, atol=5e-02), df["s22m"]
