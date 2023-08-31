from __future__ import annotations

import gdsfactory as gf
from gdsfactory.read.import_gds import import_gds


def test_import_ports_inside(data_regression) -> None:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(decorator=gf.add_pins.add_pins)
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
        decorator=gf.add_ports.add_ports_from_markers_inside,
        unique_names=False,
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


def test_import_ports_center(data_regression) -> None:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(decorator=gf.add_pins.add_pins_center)
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath,
        decorator=gf.add_ports.add_ports_from_markers_center,
        unique_names=False,
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


def test_import_ports_siepic(data_regression) -> None:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(
        decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    )
    gdspath = c0.write_gds()

    c1 = import_gds(
        gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins, unique_names=False
    )
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())


if __name__ == "__main__":
    # test_import_ports_center(None)
    # test_import_ports_siepic(None)
    # test_import_ports_inside(None)
    c0 = gf.components.straight(
        decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    )
    gdspath = c0.write_gds()

    c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
