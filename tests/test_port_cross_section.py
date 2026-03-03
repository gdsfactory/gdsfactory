from __future__ import annotations

import gdsfactory as gf


def test_add_port_registers_cross_section() -> None:
    """Test that when adding a kf.Port to a gf.Component, the gf.CrossSection is registered"""
    c = gf.Component()
    xs = gf.cross_section.cross_section(width=1, layer=(1, 0))
    c.add_port("o1", center=(0, 0), cross_section=xs, register_cross_section=True)
    xs_name = c["o1"].info["cross_section"]
    assert gf.get_cross_section(xs_name) == xs
