from __future__ import annotations

import gdsfactory as gf


def test_extract() -> None:
    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=[(111, 0)],
        bbox_offsets=[3],
    )

    c = gf.components.straight(
        length=11.124,
        cross_section=xs,
    )
    c = gf.add_pins.add_pins_container(c)
    c2 = c.extract(layers=[(1, 10)])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 2, len(c2.polygons)

    # assert (111, 0) in c.layers
    assert (1, 10) in c2.layers


if __name__ == "__main__":
    c = test_extract()
    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=[(111, 0)],
        bbox_offsets=[3],
    )

    c = gf.components.straight(
        length=11.124,
        cross_section=xs,
    )
    c = gf.add_pins.add_pins_container(c)
    c2 = c.extract(layers=[(1, 10)])

    c.show()
