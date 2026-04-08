from __future__ import annotations

from unittest.mock import MagicMock

import gdsfactory as gf
from gdsfactory.routing.resolve_pins import resolve_pin_pair, resolve_pins


def _make_pin(component: gf.Component, name: str, center: tuple[float, float], directions: list[str], width: float = 1.0, layer: tuple[int, int] = (1, 0)) -> gf.Pin:
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


def test_resolve_pin_pair_picks_closest_ports():
    """Pin A is left of Pin B — should pick A's east port and B's west port."""
    c = gf.Component()
    pin_a = _make_pin(c, "a", (0, 0), ["N", "E", "S", "W"])
    pin_b = _make_pin(c, "b", (100, 0), ["N", "E", "S", "W"])
    port_a, port_b = resolve_pin_pair(pin_a, pin_b)
    assert port_a.orientation == 0    # east
    assert port_b.orientation == 180  # west


def test_resolve_pin_pair_vertical():
    """Pin A is below Pin B — should pick A's north and B's south."""
    c = gf.Component()
    pin_a = _make_pin(c, "a", (0, 0), ["N", "E", "S", "W"])
    pin_b = _make_pin(c, "b", (0, 100), ["N", "E", "S", "W"])
    port_a, port_b = resolve_pin_pair(pin_a, pin_b)
    assert port_a.orientation == 90   # north
    assert port_b.orientation == 270  # south


def test_resolve_pin_pair_restricted_directions():
    """Pin B only has east and west — should pick closest available."""
    c = gf.Component()
    pin_a = _make_pin(c, "a", (0, 0), ["N", "E", "S", "W"])
    pin_b = _make_pin(c, "b", (0, 100), ["E", "W"])
    port_a, port_b = resolve_pin_pair(pin_a, pin_b)
    assert port_b.orientation in (0, 180)


def test_resolve_pin_pair_single_port():
    """Pin with one port behaves like a Port."""
    c = gf.Component()
    pin_a = _make_pin(c, "a", (0, 0), ["E"])
    pin_b = _make_pin(c, "b", (100, 0), ["W"])
    port_a, port_b = resolve_pin_pair(pin_a, pin_b)
    assert port_a.orientation == 0
    assert port_b.orientation == 180


def test_resolve_pin_pair_empty_pin_raises():
    """Pin with no ports should raise ValueError.

    Note: kfactory prevents creating a real Pin with zero ports, so we use a
    mock to test the defensive guard in resolve_pin_pair itself.
    """
    c = gf.Component()
    pin_a = _make_pin(c, "a", (0, 0), ["E"])
    # kfactory enforces non-empty ports at creation time, so use a mock to
    # simulate a Pin that somehow has an empty port list.
    pin_b = MagicMock()
    pin_b.name = "empty"
    pin_b.ports = []
    try:
        resolve_pin_pair(pin_a, pin_b)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower() or "no ports" in str(e).lower()


def test_resolve_pins_list():
    """resolve_pins handles lists of pin pairs."""
    c = gf.Component()
    pins1 = [
        _make_pin(c, "a1", (0, 0), ["N", "E", "S", "W"]),
        _make_pin(c, "a2", (0, 50), ["N", "E", "S", "W"]),
    ]
    pins2 = [
        _make_pin(c, "b1", (100, 0), ["N", "E", "S", "W"]),
        _make_pin(c, "b2", (100, 50), ["N", "E", "S", "W"]),
    ]
    ports1, ports2 = resolve_pins(pins1, pins2)
    assert len(ports1) == 2
    assert len(ports2) == 2
    assert all(p.orientation == 0 for p in ports1)
    assert all(p.orientation == 180 for p in ports2)


def test_resolve_pins_length_mismatch_raises():
    """Unequal pin lists should raise ValueError."""
    c = gf.Component()
    pins1 = [_make_pin(c, "a1", (0, 0), ["E"])]
    pins2 = [
        _make_pin(c, "b1", (100, 0), ["W"]),
        _make_pin(c, "b2", (100, 50), ["W"]),
    ]
    try:
        resolve_pins(pins1, pins2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "length" in str(e).lower() or "equal" in str(e).lower()
