from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.add_ports import (
    add_ports_from_labels,
    add_ports_from_markers_inside,
    add_ports_from_siepic_pins,
)
from gdsfactory.generic_tech import LAYER


def test_add_ports() -> None:
    c = gf.Component()
    s = c << gf.components.straight()
    c.add_ports(s.ports)
    assert len(c.ports) == 2, len(c.ports)


def test_add_ports_from_pins() -> None:
    x = 1.235
    c = gf.components.straight(length=x)
    c = c.copy()
    c.flatten()
    c = gf.add_pins.add_pins_container(c)

    gdspath = c.write_gds(with_metadata=False)
    add_ports = partial(
        add_ports_from_markers_inside, pin_layer=LAYER.PORT, inside=True
    )

    c2 = gf.import_gds(gdspath, post_process=(add_ports,))
    assert c2.ports["o1"].center[0] == 0, c2.ports["o1"].center[0]
    assert c2.ports["o2"].center[0] == x, c2.ports["o2"].center[0]


def test_add_ports_from_pins_path() -> None:
    x = 1.239
    c = gf.components.straight(length=x)
    c = gf.add_pins.add_pins_siepic_container(c)
    assert c.ports["o1"].center[0] == 0
    assert c.ports["o2"].center[0] == x, c.ports["o2"].center[0]
    gdspath = c.write_gds(with_metadata=False)
    c2 = gf.import_gds(gdspath, post_process=(add_ports_from_siepic_pins,))
    assert c2.ports["o1"].center[0] == 0, c2.ports["o1"].center[0]
    assert c2.ports["o2"].center[0] == x, c2.ports["o2"].center[0]


def test_add_ports_from_labels() -> None:
    x = 1.238
    c = gf.components.straight(length=x)
    c = gf.add_pins.add_pins_container(c)
    port_width = c.ports["o1"].width
    gdspath = c.write_gds(with_metadata=False)
    add_ports = partial(
        add_ports_from_labels, port_layer=LAYER.TEXT, port_width=port_width
    )

    c2 = gf.import_gds(gdspath, post_process=(add_ports,))
    assert c2.ports["o1"].center[0] == 0
    assert c2.ports["o2"].center[0] == x, c2.ports["o2"].center[0]


if __name__ == "__main__":
    test_add_ports_from_pins()
