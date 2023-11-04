from __future__ import annotations

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.port import csv2port


def test_csv2port(data_regression) -> None:
    name = "straight"
    csvpath = gf.PATH.gdsdir / f"{name}.ports"

    ports = csv2port(csvpath)
    data_regression.check(ports)


def test_get_ports_sort_clockwise() -> None:
    """.. code::

        3   4
        |___|_
    2 -|      |- 5
       |      |
    1 -|______|- 6
        |   |
        8   7

    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = ref.get_ports_list(clockwise=True)
    p1 = p[0]
    p8 = p[-1]

    assert p1.name == "o1", p1.name
    assert p1.orientation == 180, p1.orientation
    assert p8.name == "o8", p8.name
    assert p8.orientation == 270, p8.orientation


def test_get_ports_sort_counter_clockwise() -> None:
    """.. code::

        4   3
        |___|_
    5 -|      |- 2
       |      |
    6 -|______|- 1
        |   |
        7   8

    """
    c = gf.Component()
    nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    ref = c << nxn
    p = ref.get_ports_list(clockwise=False)
    p1 = p[0]
    p8 = p[-1]
    assert p1.name == "o6", p1.name
    assert p1.orientation == 0, p1.orientation
    assert p8.name == "o7", p8.name
    assert p8.orientation == 270, p8.orientation


def test_get_ports() -> None:
    c = gf.components.mzi_phase_shifter_top_heater_metal(length_x=123)

    p = c.get_ports_dict()
    assert len(p) == 10, len(p)

    p_electrical = c.get_ports_dict(width=11.0)
    p_electrical_layer = c.get_ports_dict(layer=(49, 0))
    assert len(p_electrical) == 8, f"{len(p_electrical)}"
    assert len(p_electrical_layer) == 8, f"{len(p_electrical_layer)}"

    p_optical = c.get_ports_dict(width=0.5)
    assert len(p_optical) == 2, f"{len(p_optical)}"

    p_optical_west = c.get_ports_dict(orientation=180, width=0.5)
    p_optical_east = c.get_ports_dict(orientation=0, width=0.5)
    assert len(p_optical_east) == 1, f"{len(p_optical_east)}"
    assert len(p_optical_west) == 1, f"{len(p_optical_west)}"


@pytest.mark.parametrize("port_type", ["electrical", "optical", "placement"])
def test_rename_ports(port_type, data_regression: DataRegressionFixture):
    c = gf.components.nxn(port_type=port_type)
    data_regression.check(c.to_dict())


if __name__ == "__main__":
    test_get_ports_sort_counter_clockwise()
    test_get_ports_sort_clockwise()

    # c = gf.Component()
    # nxn = gf.components.nxn(west=2, north=2, east=2, south=2)
    # ref = c << nxn
    # p = ref.get_ports_list(clockwise=False)
    # p1 = p[0]
    # p8 = p[-1]

    # assert p1.name == "o6", p1.name
    # assert p1.orientation == 0, p1.orientation
    # assert p8.name == "o7", p8.name
    # assert p8.orientation == 270, p8.orientation
