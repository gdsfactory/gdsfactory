from collections.abc import Callable

import pytest

import gdsfactory as gf


@pytest.mark.parametrize(
    "component_factory",
    [gf.components.interdigital_capacitor, gf.components.spiral_inductor],
)
def test_analog_component_defaults_to_electrical_ports_on_metal(
    component_factory: Callable[..., gf.Component],
) -> None:
    component = component_factory()

    assert component.layers == [(41, 0)]
    assert [port.name for port in component.ports] == ["e1", "e2"]
    assert all(port.port_type == "electrical" for port in component.ports)


@pytest.mark.parametrize(
    "component_factory",
    [gf.components.interdigital_capacitor, gf.components.spiral_inductor],
)
def test_analog_component_accepts_custom_layer(
    component_factory: Callable[..., gf.Component],
) -> None:
    component = component_factory(layer="M2")

    assert component.layers == [(45, 0)]
    assert all(port.layer == gf.get_layer("M2") for port in component.ports)
