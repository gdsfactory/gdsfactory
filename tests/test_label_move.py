from __future__ import annotations

import gdsfactory as gf


@gf.cell
def component_with_label() -> None:
    c = gf.Component("component_with_label")
    c << gf.components.rectangle()
    c.add_label(text="demo", position=(0.0, 0.0), layer=(66, 0))


def test_label_move() -> None:
    """test that when we move references their label also move."""
    c = gf.Component("component_with_label_move")
    ref = c << gf.components.rectangle()
    ref.movex(10)
    assert ref.origin[0] == 10
    # assert ref.labels[0].center[0] == 10


if __name__ == "__main__":
    test_label_move()
