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


def test_spiral_inductor_preserves_port_cross_section_defaults() -> None:
    component = gf.components.spiral_inductor(width=3.246)
    port = gf.Port(
        name="e1",
        width=3.246,
        layer=gf.get_layer("M1"),
        center=(0, 0),
        orientation=0,
        port_type="electrical",
    )

    assert component.ports["e1"].cross_section == port.cross_section
    assert port.cross_section.radius == pytest.approx(3.246)
