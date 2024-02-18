from __future__ import annotations

import gdsfactory as gf
from gdsfactory.read.import_gds import import_gds


def test_import_ports_inside() -> None:
    """Make sure you can import the ports"""
    c0 = gf.add_pins.add_pins_container(gf.components.straight())
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
        unique_names=False,
    )
    c1 = gf.add_ports.add_ports_from_markers_inside(c1)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"


def test_import_ports_center(data_regression) -> None:
    """Make sure you can import the ports"""
    c0 = gf.add_pins.add_pins_container_center(gf.components.straight())
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
        unique_names=False,
    )
    c1 = gf.add_ports.add_ports_from_markers_center(c1)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


def test_import_ports_siepic(data_regression) -> None:
    """Make sure you can import the ports"""
    c0 = gf.add_pins.add_pins_container_siepic(gf.components.straight())
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins, unique_names=False
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


if __name__ == "__main__":
    test_import_ports_center(None)
    # test_import_ports_siepic(None)
    # test_import_ports_inside(None)
    # c0 = gf.components.straight(
    #     decorator=gf.add_pins.add_pins_siepic, cross_section="xs_sc_no_pins"
    # )
    # gdspath = c0.write_gds()

    # c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins)
    # assert len(c1.ports) == 2, f"{len(c1.ports)}"
