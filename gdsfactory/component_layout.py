"""Helper functions for layout.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import numbers
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gdstk import Label, Polygon
from numpy import cos, pi, sin
from numpy.linalg import norm


def get_polygons(
    instance,
    by_spec: Union[bool, Tuple[int, int]] = False,
    depth: Optional[int] = None,
    include_paths: bool = True,
    as_array: bool = True,
) -> Union[List[Polygon], Dict[Tuple[int, int], List[Polygon]]]:
    """Return a list of polygons in this cell.

    Args:
        by_spec: bool or layer
            If True, the return value is a dictionary with the
            polygons of each individual pair (layer, datatype), which
            are used as keys.  If set to a tuple of (layer, datatype),
            only polygons with that specification are returned.
        depth: integer or None
            If not None, defines from how many reference levels to
            retrieve polygons.  References below this level will result
            in a bounding box.  If `by_spec` is True the key will be the
            name of this cell.
        include_paths: If True, polygonal representation of paths are also included in the result.
        as_array: when as_array=false, return the Polygon objects instead.
            polygon objects have more information (especially when by_spec=False) and are faster to retrieve.

    Returns
        out: list of array-like[N][2] or dictionary
            List containing the coordinates of the vertices of each
            polygon, or dictionary with with the list of polygons (if
            `by_spec` is True).

    Note:
        Instances of `FlexPath` and `RobustPath` are also included in
        the result by computing their polygonal boundary.
    """
    import gdsfactory as gf

    if hasattr(instance, "_cell"):
        layers = instance.get_layers()
        gdstk_instance = instance._cell

    else:
        layers = instance.parent.get_layers()
        gdstk_instance = instance._reference

    if not by_spec:
        polygons = gdstk_instance.get_polygons(depth=depth, include_paths=include_paths)

    elif by_spec is True:
        polygons = {
            layer: gdstk_instance.get_polygons(
                depth=depth,
                layer=layer[0],
                datatype=layer[1],
                include_paths=include_paths,
            )
            for layer in layers
        }

    else:
        by_spec = gf.get_layer(by_spec)
        polygons = gdstk_instance.get_polygons(
            depth=depth,
            layer=by_spec[0],
            datatype=by_spec[1],
            include_paths=include_paths,
        )

    if not as_array:
        return polygons
    if by_spec is not True:
        return [polygon.points for polygon in polygons]
    layer_to_polygons = defaultdict(list)
    for layer, polygons_list in polygons.items():
        for polygon in polygons_list:
            layer_to_polygons[layer].append(polygon.points)
    return layer_to_polygons


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
    def center(self, destination):
        """Sets the center of the bounding box.

        Args:
            destination : array-like[2] Coordinates of the new bounding box center.
        """
        self.move(destination=destination, origin=self.center)

    @property
    def x(self):
        """Returns the x-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[0] / 2

    @x.setter
    def x(self, destination):
        """Sets the x-coordinate of the center of the bounding box.

        Args:
            destination : int or float x-coordinate of the bbox center.
        """
        destination = (destination, self.center[1])
        self.move(destination=destination, origin=self.center, axis="x")

    @property
    def y(self):
        """Returns the y-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[1] / 2

    @y.setter
    def y(self, destination):
        """Sets the y-coordinate of the center of the bounding box.

        Args:
        destination : int or float
            y-coordinate of the bbox center.
        """
        destination = (self.center[0], destination)
        self.move(destination=destination, origin=self.center, axis="y")

    @property
    def xmax(self):
        """Returns the maximum x-value of the bounding box."""
        return self.bbox[1][0]

    @xmax.setter
    def xmax(self, destination):
        """Sets the x-coordinate of the maximum edge of the bounding box.

        Args:
        destination : int or float
            x-coordinate of the maximum edge of the bbox.
        """
        self.move(destination=(destination, 0), origin=self.bbox[1], axis="x")

    @property
    def ymax(self):
        """Returns the maximum y-value of the bounding box."""
        return self.bbox[1][1]

    @ymax.setter
    def ymax(self, destination):
        """Sets the y-coordinate of the maximum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the maximum edge of the bbox.
        """
        self.move(destination=(0, destination), origin=self.bbox[1], axis="y")

    @property
    def xmin(self):
        """Returns the minimum x-value of the bounding box."""
        return self.bbox[0][0]

    @xmin.setter
    def xmin(self, destination):
        """Sets the x-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float x-coordinate of the minimum edge of the bbox.
        """
        self.move(destination=(destination, 0), origin=self.bbox[0], axis="x")

    @property
    def ymin(self):
        """Returns the minimum y-value of the bounding box."""
        return self.bbox[0][1]

    @ymin.setter
    def ymin(self, destination):
        """Sets the y-coordinate of the minimum edge of the bounding box.

        Args:
            destination : int or float y-coordinate of the minimum edge of the bbox.
        """
        self.move(destination=(0, destination), origin=self.bbox[0], axis="y")

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

    def movex(self, origin=0, destination=None):
        """Moves an object by a specified x-distance.

        Args:
            origin: array-like[2], Port, or key Origin point of the move.
            destination: array-like[2], Port, key, or None Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        return self.move(origin=(origin, 0), destination=(destination, 0))

    def movey(self, origin=0, destination=None):
        """Moves an object by a specified y-distance.

        Args:
            origin : array-like[2], Port, or key Origin point of the move.
            destination : array-like[2], Port, or key Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        return self.move(origin=(0, origin), destination=(0, destination))

    def __add__(self, element):
        """Adds an element to a Group.

        Args:
            element: Component, ComponentReference, Port, Polygon,
                Label, or Group to add.
        """
        if isinstance(self, Group):
            G = Group()
            G.add(self.elements)
            G.add(element)
        else:
            G = Group([self, element])
        return G


class Group(_GeometryHelper):
    """Group objects together so you can manipulate them as a single object \
            (move/rotate/mirror)."""

    def __init__(self, *args):
        """Initialize Group."""
        self.elements = []
        self.add(args)

    def __repr__(self) -> str:
        """Prints the number of elements in the Group."""
        return f"Group ({len(self.elements)} elements total)"

    def __len__(self) -> float:
        """Returns the number of elements in the Group."""
        return len(self.elements)

    def __iadd__(self, element) -> Group:
        """Adds an element to the Group.

        Args:
            element: Component, ComponentReference, Port, Polygon,
                Label, or Group to add.

        """
        return self.add(element)

    @property
    def bbox(self):
        """Returns the bounding boxes of the Group."""
        if len(self.elements) == 0:
            raise ValueError("Group is empty, no bbox is available")
        bboxes = np.empty([len(self.elements), 4])
        for n, e in enumerate(self.elements):
            bboxes[n] = e.bbox.flatten()

        bbox = (
            (bboxes[:, 0].min(), bboxes[:, 1].min()),
            (bboxes[:, 2].max(), bboxes[:, 3].max()),
        )
        return np.array(bbox)

    def add(self, element) -> Group:
        """Adds an element to the Group.

        Args:
            element: Component, ComponentReference, Port, Polygon,
                Label, or Group to add.
        """
        from gdsfactory.component import Component
        from gdsfactory.component_reference import ComponentReference

        if _is_iterable(element):
            [self.add(e) for e in element]
        elif element is None:
            return self
        elif isinstance(
            element, (Component, ComponentReference, Polygon, Label, Group)
        ):
            self.elements.append(element)
        else:
            raise ValueError(
                "add() Could not add element to Group, the only "
                "allowed element types are "
                "(Component, ComponentReference, Polygon, Label, Group)"
            )
        # Remove non-unique entries
        used = set()
        self.elements = [
            x for x in self.elements if x not in used and (used.add(x) or True)
        ]
        return self

    def rotate(self, angle: float = 45, center=(0, 0)) -> Group:
        """Rotates all elements in a Group around the specified centerpoint.

        Args:
            angle : int or float
                Angle to rotate the Group in degrees.
            center : array-like[2] or None
                center of the Group.
        """
        for e in self.elements:
            e.rotate(angle=angle, center=center)
        return self

    def move(self, origin=(0, 0), destination=None, axis=None) -> Group:
        """Moves the Group from the origin point to the destination.

        Both origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this Group.

        Args:
            origin : array-like[2], Port, or key
                Origin point of the move.
            destination : array-like[2], Port, or key
                Destination point of the move.
            axis : {'x', 'y'}
                Direction of the move.
        """
        for e in self.elements:
            e.move(origin=origin, destination=destination, axis=axis)
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)) -> Group:
        """Mirrors a Group across the line formed between the two specified points.

        ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Args:
            p1 : array-like[N][2]
                First point of the line.
            p2 : array-like[N][2]
                Second point of the line.
        """
        for e in self.elements:
            e.mirror(p1=p1, p2=p2)
        return self

    def distribute(
        self, direction="x", spacing=100, separation=True, edge="center"
    ) -> Group:
        """Distributes the elements in the Group.

        Args:
            direction : {'x', 'y'}
                Direction of distribution; either a line in the x-direction or
                y-direction.
            spacing : int or float
                Distance between elements.
            separation : bool
                If True, guarantees elements are separated with a fixed spacing
                between; if False, elements are spaced evenly along a grid.
            edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
                Which edge to perform the distribution along (unused if
                separation == True)
        """
        _distribute(
            elements=self.elements,
            direction=direction,
            spacing=spacing,
            separation=separation,
            edge=edge,
        )
        return self

    def align(self, alignment="ymax") -> Group:
        """Aligns the elements in the Group.

        Args:
            alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
                Which edge to align along (e.g. 'ymax' will align move the elements
                such that all of their topmost points are aligned)
        """
        _align(elements=self.elements, alignment=alignment)
        return self


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
    # From http://math.stackexchange.com/questions/11515/point-reflection-across-a-line
    points = np.array(points)
    p1 = np.array(p1)
    p2 = np.array(p2)
    if np.asarray(points).ndim == 1:
        return (
            2 * (p1 + (p2 - p1) * np.dot((p2 - p1), (points - p1)) / norm(p2 - p1) ** 2)
            - points
        )
    if np.asarray(points).ndim == 2:
        return np.array(
            [
                2 * (p1 + (p2 - p1) * np.dot((p2 - p1), (p - p1)) / norm(p2 - p1) ** 2)
                - p
                for p in points
            ]
        )


def _is_iterable(items):
    """Checks if the passed variable is iterable.

    Args:
        items: any Item to check for iterability.
    """
    return isinstance(items, (list, tuple, set, np.ndarray))


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
        elements : array-like of gdsfactory objects
            Elements to distribute.
        direction : {'x', 'y'}
            Direction of distribution; either a line in the x-direction or
            y-direction.
        spacing : int or float
            Distance between elements.
        separation : bool
            If True, guarantees elements are separated with a fixed spacing between;
            if False, elements are spaced evenly along a grid.
        edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
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


def _align(elements, alignment="ymax"):
    """Aligns lists of gdsfactory elements.

    Args:
        elements : array-like of gdsfactory objects
            Elements to align.
        alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
            Which edge to align along (e.g. 'ymax' will align move the elements such
            that all of their topmost points are aligned)


    Returns
        elements : array-like of gdsfactory objects
            Aligned elements.
    """
    if len(elements) == 0:
        return elements
    if alignment not in (["x", "y", "xmin", "xmax", "ymin", "ymax"]):
        raise ValueError(
            "'alignment' argument must be one of 'x','y','xmin', 'xmax', 'ymin','ymax'"
        )
    value = Group(elements).__getattribute__(alignment)
    for e in elements:
        e.__setattr__(alignment, value)
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
