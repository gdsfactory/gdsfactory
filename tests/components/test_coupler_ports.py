import pytest

import gdsfactory as gf

COUPLER_FACTORY_NAMES = (
    "edge_coupler_silicon",
    "grating_coupler_elliptical",
    "grating_coupler_elliptical_arbitrary",
    "grating_coupler_elliptical_lumerical",
    "grating_coupler_elliptical_lumerical_etch70",
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
    "grating_coupler_elliptical_trenches",
    "grating_coupler_elliptical_uniform",
    "grating_coupler_rectangular",
    "grating_coupler_rectangular_arbitrary",
    "grating_coupler_te",
    "grating_coupler_tm",
)


@pytest.mark.parametrize("coupler_factory_name", COUPLER_FACTORY_NAMES)
def test_coupler_has_one_optical_and_one_external_port(
    coupler_factory_name: str,
) -> None:
    ports = list(getattr(gf.components, coupler_factory_name)().ports)

    assert len(ports) == 2
    assert sum(port.port_type == "optical" for port in ports) == 1
    assert sum(port.port_type != "optical" for port in ports) == 1


def test_dual_pol_grating_coupler_has_three_ports() -> None:
    ports = list(gf.components.grating_coupler_dual_pol().ports)

    assert len(ports) == 3
    assert sum(port.port_type == "optical" for port in ports) == 2
    assert sum(port.port_type == "vertical_te" for port in ports) == 1
