"""Test that all grating couplers and edge couplers have exactly two ports.

grating_coupler_dual_pol is excluded from the two-port check because it is
a dual-polarization coupler with three ports by design (o1, o2, vertical_te).
A separate test verifies it has exactly three ports.
"""

from __future__ import annotations

import pytest

import gdsfactory as gf

grating_couplers = [
    "grating_coupler_rectangular",
    "grating_coupler_elliptical",
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
    "grating_coupler_elliptical_arbitrary",
    "grating_coupler_elliptical_uniform",
    "grating_coupler_elliptical_trenches",
    "grating_coupler_te",
    "grating_coupler_tm",
    "grating_coupler_rectangular_arbitrary",
    "grating_coupler_elliptical_lumerical",
    "grating_coupler_elliptical_lumerical_etch70",
    # grating_coupler_dual_pol is excluded: it has 3 ports (o1, o2, vertical_te)
    # by design as a dual-polarization coupler. Tested separately below.
]

edge_couplers = [
    "edge_coupler_silicon",
]


@pytest.mark.parametrize("component_name", grating_couplers + edge_couplers)
def test_coupler_has_two_ports(component_name: str) -> None:
    """Each standard grating/edge coupler must expose exactly 2 ports."""
    c = getattr(gf.components, component_name)()
    port_names = [p.name for p in c.ports]
    assert len(port_names) == 2, (
        f"{component_name} has {len(port_names)} ports {port_names}, expected 2"
    )


def test_grating_coupler_dual_pol_has_three_ports() -> None:
    """grating_coupler_dual_pol is a dual-polarization coupler with 3 ports."""
    c = gf.components.grating_coupler_dual_pol()
    port_names = sorted(p.name for p in c.ports)
    assert len(port_names) == 3, (
        f"grating_coupler_dual_pol has {len(port_names)} ports {port_names}, expected 3"
    )
