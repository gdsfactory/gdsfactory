"""Helper functions for layout.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import numbers
import typing

import numpy as np
from numpy import cos, pi, sin
from numpy.linalg import norm
from rich.console import Console
from rich.table import Table

if typing.TYPE_CHECKING:
    from gdsfactory.port import Port


def pprint_ports(ports: dict[str, Port] or list[Port]) -> None:
    """Prints ports in a rich table."""
    console = Console()
    table = Table(show_header=True, header_style="bold")
    ports_list = ports if isinstance(ports, list) else list(ports.values())
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


def _parse_layer(layer):
    """Check if the variable layer is a Layer object, a 2-element list like \
    [0, 1] representing layer = 0 and datatype = 1, or just a layer number.

    Args:
        layer: int, array-like[2], or set Variable to check.

    Returns:
        (gds_layer, gds_datatype) : array-like[2]
            The layer number and datatype of the input.
    """
    if hasattr(layer, "gds_layer"):
        gds_layer, gds_datatype = layer.gds_layer, layer.gds_datatype
    elif np.shape(layer) == (2,):  # In form [3,0]
        gds_layer, gds_datatype = layer[0], layer[1]
    elif np.shape(layer) == (1,):  # In form [3]
        gds_layer, gds_datatype = layer[0], 0
    elif layer is None:
        gds_layer, gds_datatype = 0, 0
    elif isinstance(layer, numbers.Number):
        gds_layer, gds_datatype = layer, 0
    else:
        raise ValueError(
            """_parse_layer() was passed something
            that could not be interpreted as a layer: layer = %s"""
            % layer
        )
    if not isinstance(gds_layer, int):
        raise ValueError(f"invalid layer {layer}")
    if not isinstance(gds_datatype, int):
        raise ValueError(f"invalid layer {layer}")
    return (gds_layer, gds_datatype)


class _GeometryHelper:
    """Helper class for a class with functions move() and the property bbox.

    It uses that function+property to enable you to do things like check what the
    center of the bounding box is (self.center), and also to do things like move
    the bounding box such that its maximum x value is 5.2 (self.xmax = 5.2).
    """

    @property
    def center(self):
        """Returns the center of the bounding box."""
        return np.sum(self.bbox, 0) / 2

    @center.setter
    def center(self, destination) -> None:
        """Sets the center of the bounding box.

        Args:
            destination : array-like[2] Coordinates of the new bounding box center.
        """
        self.center = destination

    @property
    def x(self):
        """Returns the x-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[0] / 2

    @x.setter
    def x(self, destination) -> None:
        """Sets the x-coordinate of the center of the bounding box.

        Args:
            destination : int or float x-coordinate of the bbox center.
        """
        self.x = destination

    @property
    def y(self):
        """Returns the y-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[1] / 2

    @y.setter
    def y(self, destination) -> None:
        """Sets the y-coordinate of the center of the bounding box.

        Args:
        destination : int or float
            y-coordinate of the bbox center.
        """
        destination = (self.center[0], destination)
        self.move(other=destination, origin=self.center, axis="y")

    @property
    def xmax(self):
        """Returns the maximum x-value of the bounding box."""
        return self.bbox[1][0]

    @xmax.setter
    def xmax(self, destination) -> None:
        """Sets the x-coordinate of the maximum edge of the bounding box.

        Args:
        destination : int or float
            x-coordinate of the maximum edge of the bbox.
        """
        self.move(other=(destination, 0), origin=self.bbox[1], axis="x")

    @property
    def ymax(self):
        """Returns the maximum y-value of the bounding box."""
        return self.bbox[1][1]

    @ymax.setter
    def ymax(self, destination) -> None:
        """Sets the y-coordinate of the maximum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the maximum edge of the bbox.
        """
        self.move(other=(0, destination), origin=self.bbox[1], axis="y")

    @property
    def xmin(self):
        """Returns the minimum x-value of the bounding box."""
        return self.bbox[0][0]

    @xmin.setter
    def xmin(self, destination) -> None:
        """Sets the x-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float x-coordinate of the minimum edge of the bbox.
        """
        self.move(other=(destination, 0), origin=self.bbox[0], axis="x")

    @property
    def ymin(self):
        """Returns the minimum y-value of the bounding box."""
        return self.bbox[0][1]

    @ymin.setter
    def ymin(self, destination) -> None:
        """Sets the y-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the minimum edge of the bbox.
        """
        self.move((0, destination))

    @property
    def size(self):
        """Returns the (x, y) size of the bounding box."""
        bbox = self.bbox
        return bbox[1] - bbox[0]

    @property
    def xsize(self):
        """Returns the horizontal size of the bounding box."""
        bbox = self.bbox
        return bbox[1][0] - bbox[0][0]

    @property
    def ysize(self):
        """Returns the vertical size of the bounding box."""
        bbox = self.bbox
        return bbox[1][1] - bbox[0][1]

    def movex(self, origin=0, other=None):
        """Moves an object by a specified x-distance.

        Args:
            origin: array-like[2], Port, or key Origin point of the move.
            other: array-like[2], Port, key, or None Destination point of the move.
        """
        if other is None:
            pass
        return self.move((other, 0))

    def movey(self, origin=0, other=None):
        """Moves an object by a specified y-distance.

        Args:
            origin : array-like[2], Port, or key Origin point of the move.
            destination : array-like[2], Port, or key Destination point of the move.
        """
        if other is None:
            pass
        return self.move((0, other))


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


def _is_iterable(items):
    """Checks if the passed variable is iterable.

    Args:
        items: any Item to check for iterability.
    """
    return isinstance(items, list | tuple | set | np.ndarray)


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
        return c.center
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


def _distribute(elements, direction="x", spacing=100, separation=True, edge=None):
    """Takes a list of elements and distributes them either equally along a \
    grid or with a fixed spacing between them.

    Args:
        elements: array-like of gdsfactory objects
            Elements to distribute.
        direction: {'x', 'y'}
            Direction of distribution; either a line in the x-direction or
            y-direction.
        spacing: int or float
            Distance between elements.
        separation: bool
            If True, guarantees elements are separated with a fixed spacing between;
            if False, elements are spaced evenly along a grid.
        edge: {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
            Which edge to perform the distribution along (unused if
            separation == True)

    Returns:
        elements : Component, ComponentReference, Port, Polygon, Label, or Group
            Distributed elements.
    """
    if len(elements) == 0:
        return elements
    if direction not in ({"x", "y"}):
        raise ValueError("distribute(): 'direction' argument must be either 'x' or'y'")
    if (
        (direction == "x")
        and (edge not in ({"x", "xmin", "xmax"}))
        and (not separation)
    ):
        raise ValueError(
            "distribute(): When `separation` == False and direction == 'x',"
            " the `edge` argument must be one of {'x', 'xmin', 'xmax'}"
        )
    if (
        (direction == "y")
        and (edge not in ({"y", "ymin", "ymax"}))
        and (not separation)
    ):
        raise ValueError(
            "distribute(): When `separation` == False and direction == 'y',"
            " the `edge` argument must be one of {'y', 'ymin', 'ymax'}"
        )

    if direction == "y":
        sizes = [e.ysize for e in elements]
    if direction == "x":
        sizes = [e.xsize for e in elements]

    spacing = np.array([spacing] * len(elements))

    if separation:  # Then `edge` doesn't apply
        if direction == "x":
            edge = "xmin"
        if direction == "y":
            edge = "ymin"
    else:
        sizes = np.zeros(len(spacing))

    # Calculate new positions and move each element
    start = elements[0].__getattribute__(edge)
    positions = np.cumsum(np.concatenate(([start], (spacing + sizes))))
    for n, e in enumerate(elements):
        e.__setattr__(edge, positions[n])
    return elements


def _line_distances(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.cross(vec, start - points)
    return np.divide(abs(cross), np.linalg.norm(vec))


def _simplify(points, tolerance=0):
    """Ramer–Douglas–Peucker algorithm for line simplification.

    Takes an array of points of shape (N,2) and removes excess points in the line.
    The remaining points form a identical line to within `tolerance` from the original
    """
    # From https://github.com/fhirschmann/rdp/issues/7
    # originally written by Kirill Konevets https://github.com/kkonevets

    M = np.asarray(points)
    start, end = M[0], M[-1]
    dists = _line_distances(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax <= tolerance:
        return np.array([start, end])

    result1 = _simplify(M[: index + 1], tolerance)
    result2 = _simplify(M[index:], tolerance)

    return np.vstack((result1[:-1], result2))


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.c.straight()

    c = gf.grid(tuple(gf.components.straight(length=i) for i in range(1, 5)))
