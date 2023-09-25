from __future__ import annotations

import gdsfactory as gf


def test_extract() -> None:
    xs = gf.cross_section.strip(
        width=0.5,
        bbox_layers=[(111, 0)],
        bbox_offsets=[3],
        add_pins_function_name="add_pins_siepic",
    )

    c = gf.components.straight(
        length=11.124,
        cross_section=xs,
    )
    c2 = c.extract(layers=[(1, 10)])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 0, len(c2.polygons)

    assert len(c.paths) == 2, len(c.paths)
    assert len(c2.paths) == 2, len(c2.paths)
    assert (111, 0) in c.layers
    assert (1, 10) in c2.layers


if __name__ == "__main__":
    c = test_extract()
