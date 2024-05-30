from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_ports import (
    add_ports_from_markers_inside,
    add_ports_from_siepic_pins,
)


def test_add_ports() -> None:
    c = gf.Component()
    s = c << gf.components.straight()
    c.add_ports(s.ports)
    assert len(c.ports) == 2, len(c.ports)


def test_add_ports_from_pins() -> None:
    c = gf.components.straight(length=1.235)
    c = gf.add_pins.add_pins_container(c)
    gdspath = c.write_gds()
    c2 = gf.import_gds(gdspath, post_process=add_ports_from_markers_inside)
    assert c2.ports["o1"].dcenter[0] == 0


def test_add_ports_from_pins_path() -> None:
    c = gf.components.straight(length=1.239)
    c = gf.add_pins.add_pins_siepic_container(c)
    gdspath = c.write_gds()
    c2 = gf.import_gds(gdspath, post_process=add_ports_from_siepic_pins)
    assert c2.ports["o1"].dcenter[0] == 0
