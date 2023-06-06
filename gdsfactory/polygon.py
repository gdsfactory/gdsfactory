"""Polygon

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations
import numpy as np
import shapely as sp
import gdstk
from gdsfactory.component_layout import _GeometryHelper, _parse_move, _simplify
from gdsfactory.snap import snap_to_grid


class Polygon(gdstk.Polygon, _GeometryHelper):
    """Polygonal geometric object.

    Args:
        points: array-like[N][2] Coordinates of the vertices of the Polygon.
        gds_layer: int GDSII layer of the Polygon.
        gds_datatype: int GDSII datatype of the Polygon.
    """

    def __repr__(self) -> str:
        """Returns path points."""
        return f"Polygon(layer=({self.layer}, {self.datatype}), points={self.points})"

    def __init__(self, points, layer):
        from gdsfactory.pdk import get_layer

        layer, datatype = get_layer(layer)
        super().__init__(list(points), layer, datatype)

    @property
    def bbox(self):
        """Returns the bounding box of the Polygon."""
        return self.bounding_box()

    def rotate(
        self, angle: float = 45, center: tuple[float, float] = (0, 0)
    ) -> Polygon:
        """Rotates a Polygon by the specified angle.

        Args:
            angle: Angle to rotate the Polygon in degrees.
            center: Midpoint of the Polygon.
        """
        super().rotate(angle=angle * np.pi / 180, center=center)
        return self

    def move(
        self,
        origin: tuple[float, float] = (0, 0),
        destination: tuple[float, float] | None = None,
        axis: str | None = None,
    ) -> Polygon:
        """Moves elements of the Device from the origin point to the
        destination. Both origin and destination can be 1x2 array-like, Port,
        or a key corresponding to one of the Ports in this device.

        Args:
            origin: Origin point of the move.
            destination : Destination point of the move.
            axis: {'x', 'y'} Direction of move.

        """
        dx, dy = _parse_move(origin, destination, axis)

        super().translate(dx, dy)
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)) -> Polygon:
        """Mirrors a Polygon across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Args:
            p1: First point of the line.
            p2: Second point of the line.
        """
        return super().mirror(p1, p2)

    def simplify(self, tolerance: float = 1e-3) -> Polygon:
        """Removes points from the polygon but does not change the polygon
        shape by more than `tolerance` from the original. Uses the
        Ramer-Douglas-Peucker algorithm.

        Args:
            tolerance: float
                Tolerance value for the simplification algorithm.  All points that
                can be removed without changing the resulting polygon by more than
                the value listed here will be removed. Also known as `epsilon` here
                https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        """

        points = _simplify(self.points, tolerance=tolerance)
        return Polygon(points, (self.layer, self.datatype))

    def snap(self, nm: int = 1) -> sp.Polygon:
        """Returns new polygon snap points to grid"""
        points = snap_to_grid(self.points, nm=nm)
        return Polygon(points, (self.layer, self.datatype))

    def to_shapely(self) -> sp.Polygon:
        return sp.Polygon(self.points)

    @classmethod
    def from_shapely(cls, polygon: sp.Polygon, layer) -> Polygon:
        from gdsfactory.pdk import get_layer

        layer, datatype = get_layer(layer)
        points_on_grid = np.round(polygon.exterior.coords, 3)
        polygon = Polygon(points_on_grid, (layer, datatype))
        return polygon


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("demo")
    p0 = Polygon(zip((-8, 6, 7, 9), (-6, 8, 17, 5)), layer="WG")
    p1 = sp.Polygon(zip((-8, 6, 7, 9), (-6, 8, 17, 5)))
    p2 = p1.buffer(1)
    p3 = c.add_polygon(p2, layer=1)
    p4 = p3.simplify(tolerance=0.1)
    p5 = c.add_polygon(p4, layer=2)
    p5.mirror()
    c.show()
