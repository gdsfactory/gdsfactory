from __future__ import annotations

import pytest

import gdsfactory as gf


def test_duplicated_cells_error() -> None:
    w = h = 10
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c1 = gf.Component("test_duplicated_cells_error")
    c1.add_polygon(points, layer=(1, 0))

    w = h = 20
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]

    c2 = gf.Component()
    c2.name = "test_duplicated_cells_error"
    c2.add_polygon(points, layer=(2, 0))

    c3 = gf.Component()
    c3 << c1
    c3 << c2

    with pytest.raises(ValueError):
        c3.write_gds("rectangles.gds", on_duplicate_cell="error")


def test_duplicated_cells_pass() -> None:
    gf.Component("duplicated_cells_pass")
    c1 = gf.Component("duplicated_cells_pass")
    assert c1.name == "duplicated_cells_pass$1"
    c2 = gf.Component("duplicated_cells_pass")
    assert c2.name == "duplicated_cells_pass$2", c2.name


if __name__ == "__main__":
    # test_duplicated_cells_error()
    # test_duplicated_cells_pass()

    w = h = 10
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c1 = gf.Component("test_duplicated_cells_error")
    c1.add_polygon(points, layer=(1, 0))

    w = h = 20
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]

    c2 = gf.Component()
    c2.name = "test_duplicated_cells_error"
    c2.add_polygon(points, layer=(2, 0))

    c3 = gf.Component("top")
    c3 << c1
    c3 << c2
    c3.show()
