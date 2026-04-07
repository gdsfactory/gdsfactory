from __future__ import annotations

import numpy as np

import gdsfactory as gf
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
    c = taper_spline(length=20, width1=0.5, width2=4.0, widths=(2.0, 1.0))
    assert c.info["length"] == 20
    assert "o1" in c.ports
    assert "o2" in c.ports
    assert c.ports["o1"].width == 0.5
    assert c.ports["o2"].width == 4.0


def test_taper_spline_monotonic() -> None:
    # If we provide monotonic widths, PCHIP should keep it monotonic
    c = taper_spline(
        length=20, width1=0.5, width2=4.0, widths=(1.0, 2.0), monotonic=True
    )
    layer = gf.get_layer(c.ports["o1"].layer)
    polys = c.get_polygons(by="index")
    poly = polys[layer][0]

    # Extract points using each_point_hull or converting to DPolygon
    pts = np.array([[p.x, p.y] for p in poly.each_point_hull()])

    # Handle DBU conversion if poly is in integer units
    if isinstance(poly, gf.kdb.Polygon):
        pts = pts * c.kcl.dbu

    # The first npoints are the top curve
    top_y = pts[:100, 1]
    assert np.all(np.diff(top_y) >= -1e-9)


def test_spline_component() -> None:
    c = spline(points=((0, 0), (5, 2), (10, 0)), npoints=100)
    layer = gf.get_layer("WG")
    assert len(c.get_polygons(by="index")[layer]) >= 1


if __name__ == "__main__":
    test_spline_points()
    test_spline_polygon()
    test_taper_spline()
    test_taper_spline_monotonic()
    test_spline_component()
