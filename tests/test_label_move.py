from __future__ import annotations

import gdsfactory as gf


@gf.cell
def component_with_label() -> None:
    c = gf.Component("component_with_label")
    _ = c << gf.components.rectangle()
    c.add_label(text="demo", position=(0.0, 0.0), layer=(66, 0))


def test_label_move() -> None:
    """test that when we move references their label also move."""
    c = gf.Component("component_with_label_move")
    ref = c << gf.components.rectangle(centered=True)
    ref.d.movex(10)
    assert ref.d.center.x == 10
