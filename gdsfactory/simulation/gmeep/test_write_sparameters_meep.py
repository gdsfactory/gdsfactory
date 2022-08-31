"""test meep sparameters."""

import numpy as np

import gdsfactory as gf
import gdsfactory.simulation as sim
import gdsfactory.simulation.gmeep as gm
from gdsfactory.tech import LAYER_STACK

RESOLUTION = 20


def test_sparameters_straight() -> None:
    """Checks Sparameters for a straight waveguide."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    sp = gm.write_sparameters_meep(c, ymargin=0, overwrite=True, resolution=RESOLUTION)

    # Check reasonable reflection/transmission
    assert np.allclose(np.abs(sp["o1@0,o2@0"]), 1, atol=1e-02), np.abs(sp["o1@0,o2@0"])
    assert np.allclose(np.abs(sp["o2@0,o1@0"]), 1, atol=1e-02), np.abs(sp["o2@0,o1@0"])
    assert np.allclose(np.abs(sp["o1@0,o1@0"]), 0, atol=5e-02), np.abs(sp["o1@0,o1@0"])
    assert np.allclose(np.abs(sp["o2@0,o2@0"]), 0, atol=5e-02), np.abs(sp["o2@0,o2@0"])


def test_sparameters_straight_symmetric() -> None:
    """Checks Sparameters for a straight waveguide."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    # port_symmetries for straight
    sp = gm.write_sparameters_meep(
        c,
        overwrite=True,
        resolution=RESOLUTION,
        port_symmetries=sim.port_symmetries.port_symmetries_1x1,
        ymargin=0,
    )

    # Check reasonable reflection/transmission
    assert np.allclose(np.abs(sp["o1@0,o2@0"]), 1, atol=1e-02), np.abs(sp["o1@0,o2@0"])
    assert np.allclose(np.abs(sp["o2@0,o1@0"]), 1, atol=1e-02), np.abs(sp["o2@0,o1@0"])
    assert np.allclose(np.abs(sp["o1@0,o1@0"]), 0, atol=5e-02), np.abs(sp["o1@0,o1@0"])
    assert np.allclose(np.abs(sp["o2@0,o2@0"]), 0, atol=5e-02), np.abs(sp["o2@0,o2@0"])


def test_sparameters_crossing_symmetric() -> None:
    """Checks Sparameters for a waveguide crossing.

    Exploits symmetries.
    """
    c = gf.components.crossing()
    sp = gm.write_sparameters_meep(
        c,
        overwrite=True,
        resolution=RESOLUTION,
        port_symmetries=sim.port_symmetries.port_symmetries_crossing,
        ymargin=0,
    )
    assert sp


def test_sparameters_straight_mpi() -> None:
    """Checks Sparameters for a straight waveguide using MPI."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    filepath = gm.write_sparameters_meep_mpi(c, ymargin=0, overwrite=True)
    sp = np.load(filepath)

    # Check reasonable reflection/transmission
    assert np.allclose(np.abs(sp["o1@0,o2@0"]), 1, atol=1e-02), np.abs(sp["o1@0,o2@0"])
    assert np.allclose(np.abs(sp["o2@0,o1@0"]), 1, atol=1e-02), np.abs(sp["o2@0,o1@0"])
    assert np.allclose(np.abs(sp["o1@0,o1@0"]), 0, atol=5e-02), np.abs(sp["o1@0,o1@0"])
    assert np.allclose(np.abs(sp["o2@0,o2@0"]), 0, atol=5e-02), np.abs(sp["o2@0,o2@0"])


def test_sparameters_straight_batch() -> None:
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
    sp = np.load(filepath)

    filepath2 = sim.get_sparameters_path_meep(component=c, layer_stack=LAYER_STACK)
    assert (
        filepath2 == filepaths[0]
    ), f"filepath returned {filepaths[0]} differs from {filepath2}"

    # Check reasonable reflection/transmission
    assert np.allclose(np.abs(sp["o1@0,o2@0"]), 1, atol=1e-02), np.abs(sp["o1@0,o2@0"])
    assert np.allclose(np.abs(sp["o2@0,o1@0"]), 1, atol=1e-02), np.abs(sp["o2@0,o1@0"])
    assert np.allclose(np.abs(sp["o1@0,o1@0"]), 0, atol=5e-02), np.abs(sp["o1@0,o1@0"])
    assert np.allclose(np.abs(sp["o2@0,o2@0"]), 0, atol=5e-02), np.abs(sp["o2@0,o2@0"])


def test_sparameters_grating_coupler() -> None:
    """Checks Sparameters for a grating coupler."""
    sp = gm.write_sparameters_grating()  # fiber_angle_deg = 20
    assert sp


def test_sparameterslazy_parallelism() -> None:
    """Checks that the Sparameters computed using MPI and lazy_parallelism flag give the same results as the serial calculation."""
    c = gf.components.straight(length=2)
    p = 3
    c = gf.add_padding_container(c, default=0, top=p, bottom=p)
    print("PARALLEL")
    filepath_parallel = gm.write_sparameters_meep_mpi(
        c, ymargin=0, overwrite=True, lazy_parallelism=True
    )
    sp_parallel = np.load(filepath_parallel)
    print("SERIAL")
    filepath_serial = gm.write_sparameters_meep_mpi(
        c, ymargin=0, overwrite=True, lazy_parallelism=False
    )
    sp_serial = np.load(filepath_serial)

    # Check matching reflection/transmission
    assert np.allclose(sp_parallel["o1@0,o1@0"], sp_serial["o1@0,o1@0"], atol=1e-2)
    assert np.allclose(sp_parallel["o2@0,o1@0"], sp_serial["o2@0,o1@0"], atol=1e-2)
    assert np.allclose(sp_parallel["o1@0,o2@0"], sp_serial["o1@0,o2@0"], atol=1e-2)
    assert np.allclose(sp_parallel["o2@0,o2@0"], sp_serial["o2@0,o2@0"], atol=1e-2)


if __name__ == "__main__":
    # test_sparameters_straight(None)
    # test_sparameters_straight_symmetric(False)
    # test_sparameters_straight_batch(None)
    # test_sparameters_straight_mpi(None)
    # test_sparameters_crossing_symmetric(False)
    # test_sparameterslazy_parallelism()
    # test_sparameters_straight()
    # test_sparameters_straight_symmetric()
    test_sparameters_straight_batch()
    # test_sparameters_straight_mpi()
    # test_sparameters_crossing_symmetric()
