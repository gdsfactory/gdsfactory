from __future__ import annotations

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.components.containers import extension
from gdsfactory.components.containers.extension import extend_ports, move_polar_rad_copy


def test_extend_ports_default() -> None:
    c = gf.c.mmi1x2(cross_section="rib")
    extended_c = extend_ports(component=c)
    assert len(extended_c.ports) > 0


def test_extend_ports_with_extension() -> None:
    c0 = gf.c.straight(width=5)
    t = gf.components.taper(length=10, width1=5, width2=0.5)
    extended_c = extend_ports(component=c0, extension=t)
    assert len(extended_c.ports) > 0


def test_extend_ports_with_auto_taper() -> None:
    c0 = gf.c.straight(width=0.5)
    extended_c = extend_ports(component=c0, auto_taper=True, cross_section="strip")
    assert len(extended_c.ports) > 0


def test_extend_ports_with_custom_port_names() -> None:
    c0 = gf.c.straight(width=5)
    extended_c = extend_ports(component=c0, port_names=["o1", "o2"])
    assert len(extended_c.ports) > 0


def test_extend_ports_with_port_names_to_extend() -> None:
    c0 = gf.c.straight(width=5)
    with pytest.raises(ValueError):
        extend_ports(component=c0, port_names=["o1"], extension_port_names=["o2"])


def test_warning_for_invalid_port_name() -> None:
    c0 = gf.c.straight(width=5)
    with pytest.warns(UserWarning, match="Port Name 'invalid_port' not in"):
        extend_ports(component=c0, port_names=["invalid_port"])


def test_extend_ports_with_centered() -> None:
    c0 = gf.c.straight(width=5)
    extended_c = extend_ports(component=c0, centered=True)
    assert len(extended_c.ports) > 0


def test_extend_ports_with_allow_width_mismatch() -> None:
    c0 = gf.c.straight(width=5)
    t = gf.components.taper(length=10, width1=5, width2=0.5)
    extended_c = extend_ports(component=c0, extension=t, allow_width_mismatch=True)
    assert len(extended_c.ports) > 0


def test_extend_ports_with_custom_cross_section() -> None:
    c0 = gf.c.straight(width=0.5)
    extended_c = extend_ports(component=c0, cross_section="strip")
    assert len(extended_c.ports) > 0


def test_line_function() -> None:
    p_start = gf.Port(name="o1", center=(0, 0), width=0.5, orientation=0, layer=1)
    p_end = gf.Port(name="o2", center=(10, 0), width=0.5, orientation=0, layer=1)
    p0, p1, p2, p3 = extension.line(p_start, p_end)
    assert p0 is not None
    assert p1 is not None
    assert p2 is not None
    assert p3 is not None


def test_line_function_coordinates() -> None:
    p0, p1, p2, p3 = extension.line((0, 0), (10, 0), width=0.5)
    assert p0 is not None
    assert p1 is not None
    assert p2 is not None
    assert p3 is not None


def test_move_polar_rad_copy_function() -> None:
    pos = (0, 0)
    angle = np.pi / 4
    length = 10
    new_pos = move_polar_rad_copy(pos, angle, length)
    assert new_pos is not None
    assert len(new_pos) == 2
