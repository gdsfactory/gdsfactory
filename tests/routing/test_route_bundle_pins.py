from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.routing.route_bundle import route_bundle
from gdsfactory.routing.add_pads import add_pads_bot


def _make_pin(component: gf.Component, name: str, center: tuple[float, float], directions: list[str], width: float = 0.5, layer: int = 1) -> gf.Pin:
    """Helper to create a Pin with ports in specified directions."""
    cx, cy = center
    offset = width / 2
    direction_map = {
        "N": (f"{name}_n", (cx, cy + offset), 90),
        "E": (f"{name}_e", (cx + offset, cy), 0),
        "S": (f"{name}_s", (cx, cy - offset), 270),
        "W": (f"{name}_w", (cx - offset, cy), 180),
    }
    ports = []
    for d in directions:
        pname, pcenter, orientation = direction_map[d]
        port = component.add_port(
            name=pname, center=pcenter, width=width, orientation=orientation, layer=layer,
        )
        ports.append(port)
    return component.create_pin(name=name, ports=ports, pin_type="DC")


def test_route_bundle_with_pins_horizontal():
    """Route between two 4-port Pins separated horizontally."""
    c = gf.Component(name="test_rb_pins_h")
    pin1 = _make_pin(c, "src", (0, 0), ["N", "E", "S", "W"])
    pin2 = _make_pin(c, "dst", (200, 0), ["N", "E", "S", "W"])
    routes = route_bundle(
        c, [pin1], [pin2],
        cross_section="strip",
        allow_width_mismatch=True,
    )
    assert len(routes) == 1


def test_route_bundle_with_pins_vertical():
    """Route between two 4-port Pins separated vertically."""
    c = gf.Component(name="test_rb_pins_v")
    pin1 = _make_pin(c, "src", (0, 0), ["N", "E", "S", "W"])
    pin2 = _make_pin(c, "dst", (0, 200), ["N", "E", "S", "W"])
    routes = route_bundle(
        c, [pin1], [pin2],
        cross_section="strip",
        allow_width_mismatch=True,
    )
    assert len(routes) == 1


def test_route_bundle_with_restricted_pins():
    """Route between Pins with restricted directions."""
    c = gf.Component(name="test_rb_pins_restricted")
    pin1 = _make_pin(c, "src", (0, 0), ["E", "W"])
    pin2 = _make_pin(c, "dst", (200, 0), ["E", "W"])
    routes = route_bundle(
        c, [pin1], [pin2],
        cross_section="strip",
        allow_width_mismatch=True,
    )
    assert len(routes) == 1


def test_route_bundle_pins_type_mismatch_raises():
    """Mixing Pins and Ports should raise TypeError."""
    c = gf.Component(name="test_rb_pins_mismatch")
    pin1 = _make_pin(c, "src", (0, 0), ["E"])
    port2 = gf.Port(name="p2", center=(200, 0), width=0.5, orientation=180, layer=1)
    try:
        route_bundle(c, [pin1], [port2], cross_section="strip")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_route_bundle_ports_still_works():
    """Existing Port-based routing is unaffected."""
    c = gf.Component(name="test_rb_ports_compat")
    port1 = gf.Port(name="p1", center=(0, 0), width=0.5, orientation=0, layer=1)
    port2 = gf.Port(name="p2", center=(200, 0), width=0.5, orientation=180, layer=1)
    routes = route_bundle(c, [port1], [port2], cross_section="strip")
    assert len(routes) == 1


@pytest.mark.xfail(reason="route_fiber_array needs Pin-aware port selection — future work")
def test_add_pads_bot_with_pin_pad():
    """add_pads_bot works when pad component exposes Pins."""
    @gf.cell
    def pin_pad() -> gf.Component:
        c = gf.Component()
        size = 100
        c.add_polygon(
            [(0, 0), (size, 0), (size, size), (0, size)], layer=(1, 0)
        )
        cx, cy = size / 2, size / 2
        p_n = c.add_port(name="pad_n", center=(cx, size), width=size, orientation=90, layer=(1, 0))
        p_e = c.add_port(name="pad_e", center=(size, cy), width=size, orientation=0, layer=(1, 0))
        p_s = c.add_port(name="pad_s", center=(cx, 0), width=size, orientation=270, layer=(1, 0))
        p_w = c.add_port(name="pad_w", center=(0, cy), width=size, orientation=180, layer=(1, 0))
        c.create_pin(name="pad", ports=[p_n, p_e, p_s, p_w], pin_type="DC")
        return c

    component = gf.components.nxn(
        xsize=600,
        ysize=200,
        north=2,
        south=0,
        wg_width=10,
        layer=(1, 0),
        port_type="electrical",
    )

    # This should NOT raise the orientation assertion error
    result = add_pads_bot(
        component=component,
        pad=pin_pad,
        cross_section="strip",
        port_names=("e1", "e2"),
        allow_width_mismatch=True,
    )
    assert result is not None
