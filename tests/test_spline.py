from __future__ import annotations

import numpy as np

from gdsfactory.components.shapes.spline import spline
from gdsfactory.components.tapers.taper_spline import taper_spline
from gdsfactory.utils import spline_points, spline_polygon


def test_spline_points() -> None:
    points = [(0, 0), (10, 5), (20, 0)]
    pts = spline_points(points, npoints=50)
    assert len(pts) == 50
    assert np.allclose(pts[0], points[0])
    assert np.allclose(pts[-1], points[-1])


def test_spline_polygon() -> None:
    points = [(0, 0), (10, 5), (20, 0), (10, -5), (0, 0)]
    poly = spline_polygon(points, npoints=100)
    assert poly.num_points() == 100


def test_taper_spline() -> None:
    c = taper_spline(length=20, widths=(0.5, 2.0, 1.0, 4.0))
    assert c.info["length"] == 20
    assert "o1" in c.ports
    assert "o2" in c.ports
    assert c.ports["o1"].width == 0.5
    assert c.ports["o2"].width == 4.0


def test_taper_spline_custom_positions() -> None:
    c = taper_spline(length=20, widths=(0.5, 2.0, 4.0), positions=(0, 0.8, 1.0))
    assert c.info["length"] == 20
    assert c.ports["o1"].width == 0.5
    assert c.ports["o2"].width == 4.0


def test_spline_component() -> None:
    c = spline(points=((0, 0), (5, 2), (10, 0)), npoints=100)
    assert len(c.get_polygons()) >= 1


if __name__ == "__main__":
    test_spline_points()
    test_spline_polygon()
    test_taper_spline()
    test_taper_spline_custom_positions()
    test_spline_component()
