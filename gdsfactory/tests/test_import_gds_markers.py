from __future__ import annotations

import gdsfactory as gf
from gdsfactory.read.import_gds import import_gds


def test_import_ports_inside(data_regression) -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(decorator=gf.add_pins.add_pins)
    gdspath = c0.write_gds()

    gf.clear_cache()
    c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_markers_inside)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())
    return c1


def test_import_ports_center(data_regression) -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(decorator=gf.add_pins.add_pins_center)
    gdspath = c0.write_gds()

    gf.clear_cache()
    c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_markers_center)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())
    return c1


def test_import_ports_siepic(data_regression) -> gf.Component:
    """Make sure you can import the ports"""
    c0 = gf.components.straight(
        decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    )
    gdspath = c0.write_gds()

    gf.clear_cache()
    c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins)
    assert len(c1.ports) == 2, f"{len(c1.ports)}"
    if data_regression:
        data_regression.check(c1.to_dict())
    return c1


if __name__ == "__main__":
    # c = test_import_ports_center(None)
    # c = test_import_ports_siepic(None)
    c = test_import_ports_inside(None)
    c.pprint_ports()
    c.show(show_ports=True)

    # c0 = gf.components.straight(
    #     decorator=gf.add_pins.add_pins_siepic, cross_section="strip_no_pins"
    # )
    # gdspath = c0.write_gds()
    # c0.show()

    # gf.clear_cache()
    # c1 = import_gds(gdspath, decorator=gf.add_ports.add_ports_from_siepic_pins)
    # c1.pprint_ports()
    # c1.show()
