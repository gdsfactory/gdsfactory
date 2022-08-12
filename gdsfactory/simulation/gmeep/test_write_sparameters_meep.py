"""test meep sparameters."""

import numpy as np
import pandas as pd

import gdsfactory as gf
import gdsfactory.simulation as sim
import gdsfactory.simulation.gmeep as gm
from gdsfactory.tech import LAYER_STACK

RESOLUTION = 20


def test_sparameters_straight(dataframe_regression) -> None:
    """Checks Sparameters for a straight waveguide."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    df = gm.write_sparameters_meep(c, ymargin=0, overwrite=True, resolution=RESOLUTION)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02), df["s12m"]
    assert np.allclose(df["s21m"], 1, atol=1e-02), df["s21m"]
    assert np.allclose(df["s11m"], 0, atol=5e-02), df["s11m"]
    assert np.allclose(df["s22m"], 0, atol=5e-02), df["s22m"]

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameters_straight_symmetric(dataframe_regression) -> None:
    """Checks Sparameters for a straight waveguide."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    # port_symmetries for straight
    port_symmetries = {
        "o1": {
            "s11": ["s22"],
            "s21": ["s12"],
        }
    }
    df = gm.write_sparameters_meep(
        c,
        overwrite=True,
        resolution=RESOLUTION,
        port_symmetries=port_symmetries,
        ymargin=0,
    )

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02), df["s12m"]
    assert np.allclose(df["s21m"], 1, atol=1e-02), df["s21m"]
    assert np.allclose(df["s11m"], 0, atol=5e-02), df["s11m"]
    assert np.allclose(df["s22m"], 0, atol=5e-02), df["s22m"]

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameters_crossing_symmetric(dataframe_regression) -> None:
    """Checks Sparameters for a waveguide crossing.

    Exploits symmetries.

    """
    c = gf.components.crossing()
    port_symmetries = {
        "o1": {
            "s11": ["s22", "s33", "s44"],
            "s21": ["s12", "s34", "s43"],
            "s31": ["s13", "s24", "s42"],
            "s41": ["s14", "s23", "s32"],
        }
    }
    df = gm.write_sparameters_meep(
        c,
        overwrite=True,
        resolution=RESOLUTION,
        port_symmetries=port_symmetries,
        ymargin=0,
    )

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameters_straight_mpi(dataframe_regression) -> None:
    """Checks Sparameters for a straight waveguide using MPI."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    filepath = gm.write_sparameters_meep_mpi(c, ymargin=0, overwrite=True)
    df = pd.read_csv(filepath)

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02)
    assert np.allclose(df["s21m"], 1, atol=1e-02)
    assert np.allclose(df["s11m"], 0, atol=5e-02)
    assert np.allclose(df["s22m"], 0, atol=5e-02)

    if dataframe_regression:
        dataframe_regression.check(df)


def test_sparameters_straight_batch(dataframe_regression) -> None:
    """Checks Sparameters for a straight waveguide using an MPI pool."""

    components = []
    p = 3
    for length in [2]:
        c = gf.components.straight(length=length)
        c = gf.add_padding_container(c, default=0, top=p, bottom=p)
        components.append(c)

    filepaths = gm.write_sparameters_meep_batch(
        [{"component": c, "overwrite": True} for c in components],
    )

    filepath = filepaths[0]
    df = pd.read_csv(filepath)

    filepath2 = sim.get_sparameters_path_meep(component=c, layer_stack=LAYER_STACK)
    assert (
        filepath2 == filepaths[0]
    ), f"filepath returned {filepaths[0]} differs from {filepath2}"

    # Check reasonable reflection/transmission
    assert np.allclose(df["s12m"], 1, atol=1e-02)
    assert np.allclose(df["s21m"], 1, atol=1e-02)
    assert np.allclose(df["s11m"], 0, atol=5e-02)
    assert np.allclose(df["s22m"], 0, atol=5e-02)

    if dataframe_regression:
        dataframe_regression.check(df)


if __name__ == "__main__":
    test_sparameters_straight(None)
    # test_sparameters_straight_symmetric(False)
    # test_sparameters_straight_batch(None)
    # test_sparameters_straight_mpi(None)
    # test_sparameters_crossing_symmetric(False)
