"""Helper functions for layout.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import kfactory as kf
import numpy as np
from numpy import cos, pi, sin
from numpy.linalg import norm
from rich.console import Console
from rich.table import Table

Coordinate = tuple[float, float]


def pprint_ports(ports: kf.Ports) -> None:
    """Prints ports in a rich table."""
    console = Console()
    table = Table(show_header=True, header_style="bold")
    ports_list = ports
    if not ports_list:
        return
    p0 = ports_list[0]
    filtered_dict = {
        key: value for key, value in p0.to_dict().items() if value is not None
    }
    keys = filtered_dict.keys()

    for key in keys:
        table.add_column(key)

    for port in ports_list:
        port_dict = port.to_dict()
        row = [str(port_dict.get(key, "")) for key in keys]
        table.add_row(*row)

    console.print(table)


class _GeometryHelper:
    """Helper class for a class with functions move() and the property bbox.

    It uses that function+property to enable you to do things like check what the
    center of the bounding box is (self.dcenter), and also to do things like move
    the bounding box such that its maximum x value is 5.2 (self.dxmax = 5.2).
    """

    @property
    def dcenter(self):
        """Returns the center of the bounding box."""
        return np.sum(self.dbbox, 0) / 2

    @dcenter.setter
    def dcenter(self, destination) -> None:
        """Sets the center of the bounding box.

        Args:
            destination : array-like[2] Coordinates of the new bounding box center.
        """
        self.dmove(destination=destination, origin=self.dcenter)

    @property
    def dx(self):
        """Returns the x-coordinate of the center of the bounding box."""
        return np.sum(self.dbbox, 0)[0] / 2

    @dx.setter
    def dx(self, destination) -> None:
        """Sets the x-coordinate of the center of the bounding box.

        Args:
            destination : int or float x-coordinate of the bbox center.
        """
        destination = (destination, self.dcenter[1])
        self.dmove(destination=destination, origin=self.dcenter, axis="x")

    @property
    def dy(self):
        """Returns the y-coordinate of the center of the bounding box."""
        return np.sum(self.dbbox, 0)[1] / 2

    @dy.setter
    def dy(self, destination) -> None:
        """Sets the y-coordinate of the center of the bounding box.

        Args:
        destination : int or float
            y-coordinate of the bbox center.
        """
        destination = (self.dcenter[0], destination)
        self.dmove(destination=destination, origin=self.dcenter, axis="y")

    @property
    def dxmax(self):
        """Returns the maximum x-value of the bounding box."""
        return self.dbbox[1][0]

    @dxmax.setter
    def dxmax(self, destination) -> None:
        """Sets the x-coordinate of the maximum edge of the bounding box.

        Args:
        destination : int or float
            x-coordinate of the maximum edge of the bbox.
        """
        self.dmove(destination=(destination, 0), origin=self.dbbox[1], axis="x")

    @property
    def dymax(self):
        """Returns the maximum y-value of the bounding box."""
        return self.dbbox[1][1]

    @dymax.setter
    def dymax(self, destination) -> None:
        """Sets the y-coordinate of the maximum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the maximum edge of the bbox.
        """
        self.dmove(destination=(0, destination), origin=self.dbbox[1], axis="y")

    @property
    def dxmin(self):
        """Returns the minimum x-value of the bounding box."""
        return self.dbbox[0][0]

    @dxmin.setter
    def dxmin(self, destination) -> None:
        """Sets the x-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float x-coordinate of the minimum edge of the bbox.
        """
        self.dmove(destination=(destination, 0), origin=self.dbbox[0], axis="x")

    @property
    def dymin(self):
        """Returns the minimum y-value of the bounding box."""
        return self.dbbox[0][1]

    @dymin.setter
    def dymin(self, destination) -> None:
        """Sets the y-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the minimum edge of the bbox.
        """
        self.dmove(destination=(0, destination), origin=self.dbbox[0], axis="y")

    @property
    def dsize(self):
        """Returns the (x, y) size of the bounding box."""
        dbbox = self.dbbox
        return dbbox[1] - dbbox[0]

    @property
    def xsize(self):
        """Returns the horizontal size of the bounding box."""
        bbox = self.dbbox
        return bbox[1][0] - bbox[0][0]

    @property
    def ysize(self):
        """Returns the vertical size of the bounding box."""
        bbox = self.dbbox
        return bbox[1][1] - bbox[0][1]

    def dmovex(self, origin=0, destination=None):
        """Moves an object by a specified x-distance.

        Args:
            origin: array-like[2], Port, or key Origin point of the move.
            destination: array-like[2], Port, key, or None Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        return self.dmove(origin=(origin, 0), destination=(destination, 0))

    def dmovey(self, origin=0, destination=None):
        """Moves an object by a specified y-distance.

        Args:
            origin : array-like[2], Port, or key Origin point of the move.
            destination : array-like[2], Port, or key Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        return self.dmove(origin=(0, origin), destination=(0, destination))


def _rotate_points(points, angle: float = 45, center=(0, 0)):
    """Rotates points around a centerpoint defined by ``center``.

    ``points`` may be input as either single points [1,2] or array-like[N][2],
    and will return in kind.

    Args:
        points : array-like[N][2]
            Coordinates of the element to be rotated.
        angle : int or float
            Angle to rotate the points.
        center : array-like[2]
            Centerpoint of rotation.

    Returns:
        A new set of points that are rotated around ``center``.
    """
    if angle == 0:
        return points
    angle = angle * pi / 180
    ca = cos(angle)
    sa = sin(angle)
    sa = np.array((-sa, sa))
    c0 = np.array(center)
    if np.asarray(points).ndim == 2:
        return (points - c0) * ca + (points - c0)[:, ::-1] * sa + c0
    if np.asarray(points).ndim == 1:
        return (points - c0) * ca + (points - c0)[::-1] * sa + c0


def _reflect_points(points, p1=(0, 0), p2=(1, 0)):
    """Reflects points across the line formed by p1 and p2.

    from https://github.com/amccaugh/phidl/pull/181

    ``points`` may be input as either single points [1,2] or array-like[N][2],
    and will return in kind.

    Args:
        points : array-like[N][2]
            Coordinates of the element to be reflected.
        p1 : array-like[2]
            Coordinates of the start of the reflecting line.
        p2 : array-like[2]
            Coordinates of the end of the reflecting line.

    Returns:
        A new set of points that are reflected across ``p1`` and ``p2``.
    """
    original_shape = np.shape(points)
    points = np.atleast_2d(points)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    line_vec = p2 - p1
    line_vec_norm = norm(line_vec) ** 2

    # Compute reflection
    proj = np.sum(line_vec * (points - p1), axis=-1, keepdims=True)
    reflected_points = 2 * (p1 + (p2 - p1) * proj / line_vec_norm) - points

    return reflected_points if original_shape[0] > 1 else reflected_points[0]


def _parse_coordinate(c):
    """Translates various inputs (lists, tuples, Ports) to an (x,y) coordinate.

    Args:
        c: array-like[N] or Port
            Input to translate into a coordinate.

    Returns:
        c : array-like[2]
            Parsed coordinate.
    """
    if hasattr(c, "center"):
        return c.dcenter
    elif np.array(c).size == 2:
        return c
    else:
        raise ValueError(
            "Could not parse coordinate, input should be array-like (e.g. [1.5,2.3] or a Port"
        )


def _parse_move(origin, destination, axis):
    """Translates input coordinates to changes in position in the x and y directions.

    Args:
        origin : array-like[2] of int or float, Port, or key
            Origin point of the move.
        destination : array-like[2] of int or float, Port, key, or None
            Destination point of the move.
        axis : {'x', 'y'} Direction of move.

    Returns:
        dx : int or float
            Change in position in the x-direction.
        dy : int or float
            Change in position in the y-direction.
    """
    # If only one set of coordinates is defined, make sure it's used to move things
    if destination is None:
        destination = origin
        origin = [0, 0]

    d = _parse_coordinate(destination)
    o = _parse_coordinate(origin)
    if axis == "x":
        d = (d[0], o[1])
    if axis == "y":
        d = (o[0], d[1])
    dx, dy = np.array(d) - o

    return dx, dy


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.c.straight()

    c = gf.grid(tuple(gf.components.straight(length=i) for i in range(1, 5)))
