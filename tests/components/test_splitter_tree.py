import numpy as np

import gdsfactory as gf


def test_splitter_tree_ports() -> None:
    c = gf.c.splitter_tree(
        coupler="mmi2x2",
        noutputs=4,
    )
    assert len(c.ports) == 8, len(c.ports)


def test_splitter_tree_ports_no_sbend() -> None:
    c = gf.c.splitter_tree(
        coupler="mmi2x2",
        noutputs=4,
        bend_s=None,
    )
    assert len(c.ports) == 8, len(c.ports)


def test_length_spiral_racetrack() -> None:
    length = 1000
    c = gf.c.spiral_racetrack_fixed_length(length=length, cross_section="strip")
    length_computed = c.area(layer=(1, 0)) / 0.5
    np.isclose(length, length_computed)


def test_ports() -> None:
    c = gf.c.straight_heater_metal(length=50.0)
    assert c.ports["o2"].dcenter[0] == 50.0, c.ports["o2"].dcenter[0]
