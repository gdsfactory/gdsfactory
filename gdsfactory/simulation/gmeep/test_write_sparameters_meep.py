"""test meep sparameters"""
import time

import numpy as np
import pandas as pd

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm


def test_sparameterNxN_straight(dataframe_regression):
    """Checks that computed transmission is reasonable
    to see if there are issues in get_simulation + transmission analysis
    """
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    df = gm.write_sparameters_meep(c, overwrite=True, animate=False)

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
    df = gm.write_sparameters_meep(c, overwrite=True, animate=False)

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



def test_sparameterNxN_symmetries_straight(dataframe_regression):
    """Checks the duplication of Sparameters when using port_symmetry toggle
    Uses a straight to be faster than crossing, although crossing works well too
    """

    # No symmetry toggle
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    start = time.time()
    df = write_sparameters_meep(
        c,
        overwrite=True,
        resolution=50,  # Comparison needs higher resolution
        animate=False,
    )
    stop = time.time()
    time_full = stop - start

    # port_symmetries for straight
    port_symmetries = {
        "o1": {
            "s11": ["s22"],
            "s21": ["s12"],
        }
    }
    start = time.time()
    df_symm = write_sparameters_meep(
        c,
        overwrite=True,
        animate=False,
        resolution=50,  # Comparison needs higher resolution
        port_symmetries=port_symmetries,
    )
    stop = time.time()
    time_symm = stop - start

    # Compare symmetry to no symmetry
    for i in range(1, len(c.ports) + 1):
        for j in range(1, len(c.ports) + 1):
            assert np.allclose(
                df["s{}{}m".format(i, j)].to_numpy(),
                df_symm["s{}{}m".format(i, j)].to_numpy(),
                atol=5e-02,
            )
            assert np.allclose(
                df["s{}{}a".format(i, j)].to_numpy(),
                df_symm["s{}{}a".format(i, j)].to_numpy(),
                atol=5e-02,
            )
    # Check that it was shorter
    assert time_full > time_symm

    if dataframe_regression:
        dataframe_regression.check(df)


# def test_sparameterNxN_symmetries_crossing(dataframe_regression):
#     """Checks the duplication of Sparameters when using port_symmetry toggle
#     Uses a straight to be faster than crossing, although crossing works well too
#     """

#     # No symmetry toggle
#     c = gf.components.crossing()
#     # c = gf.components.straight(length=2)
#     # p = 3
#     # c = gf.add_padding_container(c, default=0, top=p, bottom=p)
#     start = time.time()
#     df = write_sparameters_meep(
#         c,
#         overwrite=True,
#         resolution=50,  # Comparison needs higher resolution
#         animate=False,
#     )
#     stop = time.time()
#     time_full = stop - start

#     # port_symmetries for crossing
#     port_symmetries={
#             "o1": {
#                 "s11": ["s22", "s33", "s44"],
#                 "s21": ["s12", "s34", "s43"],
#                 "s31": ["s13", "s24", "s42"],
#                 "s41": ["s14", "s23", "s32"],
#             }
#         }
#     start = time.time()
#     df_symm = write_sparameters_meep(
#         c,
#         overwrite=True,
#         animate=False,
#         resolution=50,  # Comparison needs higher resolution
#         port_symmetries=port_symmetries,
#     )
#     stop = time.time()
#     time_symm = stop - start

#     # Compare symmetry to no symmetry
#     for i in range(1, len(c.ports) + 1):
#         for j in range(1, len(c.ports) + 1):
#             assert np.allclose(
#                 df["s{}{}m".format(i, j)].to_numpy(),
#                 df_symm["s{}{}m".format(i, j)].to_numpy(),
#                 atol=5e-02,
#             )
#             assert np.allclose(
#                 df["s{}{}a".format(i, j)].to_numpy(),
#                 df_symm["s{}{}a".format(i, j)].to_numpy(),
#                 atol=5e-02,
#             )
#     # Check that it was shorter
#     assert time_full > time_symm

#     if dataframe_regression:
#         dataframe_regression.check(df)
#         dataframe_regression.check(df_symm)

def test_sparameter_straight_mpi(dataframe_regression):
    """Checks that computed transmission is reasonable
    to see if there are issues in get_simulation + transmission analysis
    """
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    filepath = gm.write_sparameters_meep_mpi(c, overwrite=True)
    df = pd.read_csv(filepath)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02)
    assert np.allclose(df["s21m"], 1, atol=1e-02)
    assert np.allclose(df["s11m"], 0, atol=5e-02)
    assert np.allclose(df["s22m"], 0, atol=5e-02)

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameter_straight_mpi_pool(dataframe_regression):
    """Checks that computed transmission is reasonable
    to see if there are issues in get_simulation + transmission analysis
    """

    components = []
    for length in [2, 3]:
        c = gf.components.straight(length=length)
        p = 3
        c = gf.add_padding_container(c, default=0, top=p, bottom=p)
        components.append(c)

    filepaths = gm.write_sparameters_meep_mpi_pool(
        [{"component": c, "overwrite": True} for c in components],
    )

    filepath = filepaths[0]
    df = pd.read_csv(filepath)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02)
    assert np.allclose(df["s21m"], 1, atol=1e-02)
    assert np.allclose(df["s11m"], 0, atol=5e-02)
    assert np.allclose(df["s22m"], 0, atol=5e-02)

    if dataframe_regression:
        dataframe_regression.check(df)


if __name__ == "__main__":
    test_sparameter_straight_mpi_pool(None)
    # test_sparameterNxN_straight(None)
    # test_sparameterNxN_crossing(None)
