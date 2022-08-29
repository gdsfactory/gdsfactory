import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt


def test_sparameters_straight(overwrite=False) -> None:
    """Checks Sparameters for a straight waveguide in 2D."""
    c = gf.components.straight(length=2)
    sp = gt.write_sparameters_1x1(c, overwrite=overwrite, is_3d=False)

    # Check reasonable reflection/transmission
    assert sp["s12m"].min() > 0.89, sp["s12m"].min()
    assert sp["s11m"].max() < 0.1, sp["s11m"].max()

    # assert np.allclose(sp["s21m"], 1, atol=1e-02), sp["s21m"]
    # assert np.allclose(sp["s11m"], 0, atol=5e-02), sp["s11m"]
    # assert np.allclose(sp["s22m"], 0, atol=5e-02), sp["s22m"]

    # if dataframe_regression:
    #     dataframe_regression.check(sp)


if __name__ == "__main__":
    test_sparameters_straight()

    # c = gf.components.straight(length=2)
    # sp = gt.write_sparameters_1x1(c, overwrite=True)

    # # Check reasonable reflection/transmission
    # assert np.allclose(sp["s12m"], 1, atol=1e-02), sp["s12m"]
    # assert np.allclose(sp["s21m"], 1, atol=1e-02), sp["s21m"]
    # assert np.allclose(sp["s11m"], 0, atol=5e-02), sp["s11m"]
    # assert np.allclose(sp["s22m"], 0, atol=5e-02), sp["s22m"]
