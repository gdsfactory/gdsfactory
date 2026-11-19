from __future__ import annotations

import gdsfactory as gf

PN_PORTS = ("sig_left", "gnd_left", "sig_right", "gnd_right")


def test_ring_double_pn_pn_via_ports() -> None:
    """The add-drop pn ring exposes pinned electrical ports on its pn vias."""
    c = gf.c.ring_double_pn()
    ports = {p.name: p for p in c.ports}
    for name in PN_PORTS:
        assert ports[name].port_type == "electrical"
    assert set(PN_PORTS) <= {pin.name for pin in c.pins}


def test_ring_double_pn_pn_vias_are_point_symmetric() -> None:
    """Left and right pn vias mirror each other through one common point."""
    c = gf.c.ring_double_pn()
    ports = {p.name: p for p in c.ports}
    midpoints = {
        side: (
            (ports[f"{side}_left"].x + ports[f"{side}_right"].x) / 2,
            (ports[f"{side}_left"].y + ports[f"{side}_right"].y) / 2,
        )
        for side in ("sig", "gnd")
    }
    assert abs(midpoints["sig"][0] - midpoints["gnd"][0]) < 1e-3
    assert abs(midpoints["sig"][1] - midpoints["gnd"][1]) < 1e-3
