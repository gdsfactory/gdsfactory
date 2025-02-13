import pytest

import gdsfactory as gf
from gdsfactory import cross_section as xs


def test_add_fiber_array_optical_south_electrical_north_default() -> None:
    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal=xs.metal_routing,
        pad_pitch=100,
    )
    assert c is not None
    assert len(c.ports) > 0


def test_add_fiber_array_optical_south_electrical_north_with_loopback() -> None:
    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal=xs.metal_routing,
        with_loopback=True,
        pad_pitch=100,
    )
    assert c is not None
    assert len(c.ports) > 0


def test_add_fiber_array_optical_south_electrical_north_no_loopback() -> None:
    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal=xs.metal_routing,
        with_loopback=False,
        pad_pitch=100,
    )
    assert c is not None
    assert len(c.ports) > 0


def test_add_fiber_array_optical_south_electrical_north_custom_pitch() -> None:
    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal=xs.metal_routing,
        pad_pitch=150,
    )
    assert c is not None
    assert len(c.ports) > 0


def test_add_fiber_array_optical_south_electrical_north_custom_electrical_ports() -> (
    None
):
    c = gf.c.add_fiber_array_optical_south_electrical_north(
        component=gf.c.straight_heater_metal,
        pad=gf.c.pad,
        grating_coupler=gf.c.grating_coupler_te,
        cross_section_metal=xs.metal_routing,
        electrical_port_names=["l_e1", "l_e2"],
        pad_pitch=100,
    )
    assert c is not None
    assert len(c.ports) > 0


if __name__ == "__main__":
    pytest.main([__file__])
