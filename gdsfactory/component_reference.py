"""Component references.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import typing
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import gdstk
import numpy as np
import shapely
from numpy import cos, mod, ndarray, pi, sin

from gdsfactory.component_layout import Polygon, _GeometryHelper, get_polygons
from gdsfactory.port import (
    Port,
    map_ports_layer_to_orientation,
    map_ports_to_orientation_ccw,
    map_ports_to_orientation_cw,
    select_ports,
)

if typing.TYPE_CHECKING:
    from gdsfactory.component import Component, Coordinate, Coordinates


class SizeInfo:
    def __init__(self, bbox: ndarray) -> None:
        """Initialize this object."""
        self.west = bbox[0, 0]
        self.east = bbox[1, 0]
        self.south = bbox[0, 1]
        self.north = bbox[1, 1]

        self.width = self.east - self.west
        self.height = self.north - self.south

        xc = 0.5 * (self.east + self.west)
        yc = 0.5 * (self.north + self.south)

        self.sw = np.array([self.west, self.south])
        self.se = np.array([self.east, self.south])
        self.nw = np.array([self.west, self.north])
        self.ne = np.array([self.east, self.north])

        self.cw = np.array([self.west, yc])
        self.ce = np.array([self.east, yc])
        self.nc = np.array([xc, self.north])
        self.sc = np.array([xc, self.south])
        self.cc = self.center = np.array([xc, yc])

    def get_rect(
        self, padding=0, padding_w=None, padding_e=None, padding_n=None, padding_s=None
    ) -> Tuple[Coordinate, Coordinate, Coordinate, Coordinate]:
        w, e, s, n = self.west, self.east, self.south, self.north

        padding_n = padding if padding_n is None else padding_n
        padding_e = padding if padding_e is None else padding_e
        padding_w = padding if padding_w is None else padding_w
        padding_s = padding if padding_s is None else padding_s

        w = w - padding_w
        e = e + padding_e
        s = s - padding_s
        n = n + padding_n

        return ((w, s), (e, s), (e, n), (w, n))

    @property
    def rect(self) -> Tuple[Coordinate, Coordinate]:
        return self.get_rect()

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"w: {self.west}\ne: {self.east}\ns: {self.south}\nn: {self.north}\n"


def _rotate_points(
    points: Coordinates,
    angle: float = 45.0,
    center: Coordinate = (
        0.0,
        0.0,
    ),
) -> ndarray:
    """Rotates points around a center point.

    accepts single points [1,2] or array-like[N][2], and will return in kind

    Args:
        points: rotate points around center point.
        angle: in degrees.
        center: x, y.
    """
    # First check for common, easy values of angle
    p_arr = np.asarray(points)
    if angle == 0:
        return p_arr

    c0 = np.asarray(center)
    displacement = p_arr - c0
    if angle == 180:
        return c0 - displacement

    if p_arr.ndim == 2:
        perpendicular = displacement[:, ::-1]
    elif p_arr.ndim == 1:
        perpendicular = displacement[::-1]

    # Fall back to trigonometry
    angle = angle * pi / 180
    ca = cos(angle)
    sa = sin(angle)
    sa = np.array((-sa, sa))
    return displacement * ca + perpendicular * sa + c0


class ComponentReference(_GeometryHelper):
    """Pointer to a Component with x, y, rotation, mirror.

    Args:
        component: Component The referenced Component.
        columns: Number of columns in the array.
        rows: Number of rows in the array.
        spacing: Distances between adjacent columns and adjacent rows.
        origin: array-like[2] of int or float
            Position where the cell is inserted.
        rotation : int or float
            Angle of rotation of the reference (in `degrees`).
        magnification : int or float
            Magnification factor for the reference.
        x_reflection : bool
            If True, the reference is reflected parallel to the x direction
            before being rotated.
        name : str (optional)
            A name for the reference (if provided).

    """

    def __init__(
        self,
        component: Component,
        origin: Coordinate = (0, 0),
        rotation: float = 0,
        magnification: float = 1,
        x_reflection: bool = False,
        visual_label: str = "",
        columns: int = 1,
        rows: int = 1,
        spacing=None,
        name: Optional[str] = None,
        v1: Optional[Tuple[float, float]] = None,
        v2: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize the ComponentReference object."""
        self._reference = gdstk.Reference(
            cell=component._cell,
            origin=origin,
            rotation=np.deg2rad(rotation),
            magnification=magnification,
            x_reflection=x_reflection,
            columns=columns,
            rows=rows,
            spacing=spacing,
        )
        if v1 or v2:
            self._reference.repetition = gdstk.Repetition(
                columns=columns, rows=rows, v1=v1, v2=v2
            )

        self._ref_cell = component
        self._owner = None
        self._name = name

        # The ports of a ComponentReference have their own unique id (uid),
        # since two ComponentReferences of the same parent Component can be
        # in different locations and thus do not represent the same port
        self._local_ports = {
            name: port._copy() for name, port in component.ports.items()
        }
        self.visual_label = visual_label
        # self.uid = str(uuid.uuid4())[:8]

    @property
    def v1(self) -> Optional[Tuple[float, float]]:
        return self._reference.repetition.v1

    @property
    def v2(self) -> Optional[Tuple[float, float]]:
        return self._reference.repetition.v2

    @property
    def rows(self) -> int:
        return self._reference.repetition.rows or 1

    @property
    def columns(self) -> int:
        return self._reference.repetition.columns or 1

    @property
    def spacing(self) -> Optional[Tuple[float, float]]:
        return self._reference.repetition.spacing

    @property
    def ref_cell(self):
        return self._ref_cell

    @property
    def parent(self):
        return self._ref_cell

    @property
    def origin(self):
        return self._reference.origin

    @origin.setter
    def origin(self, value) -> None:
        self._reference.origin = value

    @property
    def magnification(self) -> float:
        return self._reference.magnification

    @magnification.setter
    def magnification(self, value) -> None:
        self._reference.magnification = value

    @property
    def rotation(self) -> float:
        return np.rad2deg(self._reference.rotation)

    @rotation.setter
    def rotation(self, value) -> None:
        self._reference.rotation = np.deg2rad(value)

    @property
    def x_reflection(self) -> bool:
        return self._reference.x_reflection

    @x_reflection.setter
    def x_reflection(self, value) -> None:
        self._reference.x_reflection = value

    def _set_ref_cell(self, value) -> None:
        self._ref_cell = value
        self._reference.cell = value._cell

    @ref_cell.setter
    def ref_cell(self, value) -> None:
        self._set_ref_cell(value)

    @parent.setter
    def parent(self, value) -> None:
        self._set_ref_cell(value)

    def get_polygon_enclosure(self) -> shapely.Polygon:
        return shapely.Polygon(self._reference.convex_hull())

    def get_polygon_bbox(
        self,
        default: float = 0.0,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        right: Optional[float] = None,
        left: Optional[float] = None,
    ) -> shapely.Polygon:
        """Returns shapely Polygon with padding.

        Args:
            default: default padding in um.
            top: north padding in um.
            bottom: south padding in um.
            right: east padding in um.
            left: west padding in um.
        """
        (xmin, ymin), (xmax, ymax) = self.bbox
        top = top if top is not None else default
        bottom = bottom if bottom is not None else default
        right = right if right is not None else default
        left = left if left is not None else default
        points = [
            [xmin - left, ymin - bottom],
            [xmax + right, ymin - bottom],
            [xmax + right, ymax + top],
            [xmin - left, ymax + top],
        ]
        return shapely.Polygon(points)

    def get_polygons(
        self,
        by_spec: bool = False,
        depth: Optional[int] = None,
        include_paths: bool = True,
        as_array: bool = True,
        as_shapely: bool = False,
        as_shapely_merged: bool = False,
    ) -> Union[List[Polygon], Dict[Tuple[int, int], List[Polygon]]]:
        """Return the list of polygons created by this reference.

        Args:
            by_spec : bool or tuple
                If True, the return value is a dictionary with the
                polygons of each individual pair (layer, datatype).
                If set to a tuple of (layer, datatype), only polygons
                with that specification are returned.
            depth : integer or None
                If not None, defines from how many reference levels to
                retrieve polygons.  References below this level will result
                in a bounding box.  If `by_spec` is True the key will be the
                name of the referenced cell.
            include_paths: If True, polygonal representation of paths are also included in the result.
            as_array: when as_array=false, return the Polygon objects instead.
                polygon objects have more information (especially when by_spec=False)
                and are faster to retrieve.
            as_shapely: returns shapely polygons.
            as_shapely_merged: returns a shapely polygonize.

        Returns
            out : list of array-like[N][2] or dictionary
                List containing the coordinates of the vertices of each
                polygon, or dictionary with the list of polygons (if
                `by_spec` is True).

        Note:
            Instances of `FlexPath` and `RobustPath` are also included in
            the result by computing their polygonal boundary.
        """
        return get_polygons(
            instance=self,
            by_spec=by_spec,
            depth=depth,
            include_paths=include_paths,
            as_array=as_array,
            as_shapely=as_shapely,
            as_shapely_merged=as_shapely_merged,
        )

    def get_labels(self, depth=None, set_transform=True):
        """Return the list of labels created by this reference.

        Args:
            depth : integer or None
                If not None, defines from how many reference levels to
                retrieve labels from.
            set_transform : bool
                If True, labels will include the transformations from
                the reference.

        Returns:
            out : list of `Label`
                List containing the labels in this cell and its references.
        """
        if set_transform:
            return self._reference.get_labels(depth=depth)
        else:
            return self.parent.get_labels(depth=depth)

    def get_bounding_box(self):
        return self._reference.bounding_box()

    @property
    def layers(self) -> Set[Tuple[int, int]]:
        return self.parent.layers

    @property
    def settings(self):
        return self.parent.settings

    def get_paths(self, depth=None):
        """Return the list of paths created by this reference.

        Args:
            depth : integer or None
                If not None, defines from how many reference levels to
                retrieve paths from.

        Returns:
            list of `FlexPath` or `RobustPath`
                List containing the paths in this cell and its references.
        """
        return self._reference.get_paths(depth=depth)

    def translate(self, dx, dy) -> ComponentReference:
        x0, y0 = self._reference.origin
        self.origin = (x0 + dx, y0 + dy)
        return self

    def area(self, by_spec=False):
        """Calculate total area.

        Args:
            by_spec : bool
                If True, the return value is a dictionary with the areas
                of each individual pair (layer, datatype).

        Returns:
            out : number, dictionary
                Area of this cell.
        """
        return self._reference.area(by_spec=by_spec)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        if self.owner is None or value is None:
            self._owner = value
        elif value != self._owner:
            raise ValueError(
                f"Cannot reset owner of a reference once it has already been set!"
                f" Reference: {self}. Current owner: {self._owner}. "
                f"Attempting to re-assign to {value!r}"
            )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        if value != self._name:
            if self.owner and value in self.owner.named_references:
                raise ValueError(
                    f"This reference's owner already has a reference with name {value!r}. Please choose another name."
                )
            if self.owner:
                self.owner._named_references.pop(self._name, None)
                self.owner._named_references[value] = self
            self._name = value

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            'ComponentReference (parent Component "%s", ports %s, origin %s, rotation %s,'
            " x_reflection %s)"
            % (
                self.parent.name,
                list(self.ports.keys()),
                self.origin,
                self.rotation,
                self.x_reflection,
            )
        )

    def to_dict(self):
        d = self.parent.to_dict()
        d.update(
            origin=self.origin,
            rotation=self.rotation,
            magnification=self.magnification,
            x_reflection=self.x_reflection,
        )
        return d

    @property
    def bbox(self):
        """Return the bounding box of the ComponentReference.

        it snaps to 3 decimals in um (0.001um = 1nm precision)
        """
        bbox = self.get_bounding_box()
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.array(bbox)

    @classmethod
    def __get_validators__(cls):
        """Get validators."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Check with pydantic ComponentReference valid type."""
        assert isinstance(
            v, ComponentReference
        ), f"TypeError, Got {type(v)}, expecting ComponentReference"
        return v

    def __getitem__(self, key):
        """Access reference ports."""
        if key not in self.ports:
            ports = list(self.ports.keys())
            raise ValueError(f"{key!r} not in {ports}")

        return self.ports[key]

    @property
    def ports(self) -> Dict[str, Port]:
        """This property allows you to access myref.ports, and receive a copy.

        of the ports dict which is correctly rotated and translated.
        """
        for name, port in self.parent.ports.items():
            port = self.parent.ports[name]
            new_center, new_orientation = self._transform_port(
                port.center,
                port.orientation,
                self.origin,
                self.rotation,
                self.x_reflection,
            )
            if name not in self._local_ports:
                self._local_ports[name] = port.copy()
            self._local_ports[name].center = new_center
            self._local_ports[name].orientation = (
                mod(new_orientation, 360) if new_orientation else new_orientation
            )
            self._local_ports[name].parent = self
        # Remove any ports that no longer exist in the reference's parent
        parent_names = self.parent.ports.keys()
        local_names = list(self._local_ports.keys())
        for name in local_names:
            if name not in parent_names:
                self._local_ports.pop(name)
        for k in list(self._local_ports):
            self._local_ports[k].reference = self
        return self._local_ports

    @property
    def info(self) -> Dict[str, Any]:
        return self.parent.info

    @property
    def metadata_child(self) -> Dict[str, Any]:
        return self.parent.metadata_child

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(self.bbox)

    def pprint_ports(self) -> None:
        """Pretty print component ports."""
        ports_list = self.get_ports_list()
        for port in ports_list:
            print(port)

    def _transform_port(
        self,
        point: ndarray,
        orientation: float,
        origin: Coordinate = (0, 0),
        rotation: Optional[int] = None,
        x_reflection: bool = False,
    ) -> Tuple[ndarray, float]:
        """Apply GDS-type transformation to a port (x_ref)."""
        new_point = np.array(point)
        new_orientation = orientation

        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = None if orientation is None else -orientation
        if rotation is not None and orientation is not None:
            new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)

        if orientation is not None:
            new_orientation = mod(new_orientation, 360)

        return new_point, new_orientation

    def _transform_point(
        self,
        point: ndarray,
        origin: Coordinate = (0, 0),
        rotation: Optional[int] = None,
        x_reflection: bool = False,
    ) -> ndarray:
        """Apply GDS-type transformation to a point."""
        new_point = np.array(point)

        if x_reflection:
            new_point[1] = -new_point[1]
        if rotation is not None:
            new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
        if origin is not None:
            new_point = new_point + np.array(origin)

        return new_point

    def move(
        self,
        origin: Union[Port, Coordinate, str] = (0, 0),
        destination: Optional[Union[Port, Coordinate, str]] = None,
        axis: Optional[str] = None,
    ) -> ComponentReference:
        """Move the ComponentReference from origin point to destination.

        Both origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this device_ref.

        Args:
            origin: Port, port_name or Coordinate.
            destination: Port, port_name or Coordinate.
            axis: for the movement.

        Returns:
            ComponentReference.
        """
        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = (0, 0)

        if isinstance(origin, str):
            if origin not in self.ports:
                raise ValueError(f"{origin} not in {self.ports.keys()}")

            origin = self.ports[origin]
            origin = cast(Port, origin)
            o = origin.center
        elif hasattr(origin, "center"):
            origin = cast(Port, origin)
            o = origin.center
        elif np.array(origin).size == 2:
            o = origin
        else:
            raise ValueError(
                f"move(origin={origin})\n"
                f"Invalid origin = {origin!r} needs to be"
                f"a coordinate, port or port name {list(self.ports.keys())}"
            )

        if isinstance(destination, str):
            if destination not in self.ports:
                raise ValueError(f"{destination} not in {self.ports.keys()}")

            destination = self.ports[destination]
            destination = cast(Port, destination)
            d = destination.center
        if hasattr(destination, "center"):
            destination = cast(Port, destination)
            d = destination.center
        elif np.array(destination).size == 2:
            d = destination

        else:
            raise ValueError(
                f"{self.parent.name}.move(destination={destination!r}) \n"
                f"Invalid destination = {destination!r} needs to be"
                f"a coordinate, a port, or a valid port name {list(self.ports.keys())}"
            )

        # Lock one axis if necessary
        if axis == "x":
            d = (d[0], o[1])
        if axis == "y":
            d = (o[0], d[1])

        # This needs to be done in two steps otherwise floating point errors can accrue
        dxdy = np.array(d) - np.array(o)
        self.origin = np.array(self.origin) + dxdy
        self._bb_valid = False
        return self

    def rotate(
        self,
        angle: float = 45,
        center: Union[Coordinate, str, int] = (0.0, 0.0),
    ) -> ComponentReference:
        """Return rotated ComponentReference.

        Args:
            angle: in degrees.
            center: x, y.
        """
        if angle == 0:
            return self
        if isinstance(center, (int, str)):
            center = self.ports[center].center

        if isinstance(center, Port):
            center = center.center
        self.rotation += angle
        self.rotation %= 360
        self.origin = _rotate_points(self.origin, angle, center)
        self._bb_valid = False
        return self

    def mirror_x(
        self, port_name: Optional[str] = None, x0: Optional[Coordinate] = None
    ) -> ComponentReference:
        """Perform horizontal mirror using x0 or port as axis (default, x0=0).

        This is the default for mirror along X=x0 axis
        """
        if port_name is None and x0 is None:
            x0 = -self.x

        if port_name is not None:
            position = self.ports[port_name]
            x0 = position.x
        self.mirror((x0, 1), (x0, 0))
        return self

    def mirror_y(
        self, port_name: Optional[str] = None, y0: Optional[float] = None
    ) -> ComponentReference:
        """Perform vertical mirror using y0 as axis (default, y0=0)."""
        if port_name is None and y0 is None:
            y0 = 0.0

        if port_name is not None:
            position = self.ports[port_name]
            y0 = position.y
        self.mirror((1, y0), (0, y0))
        return self

    def mirror(
        self,
        p1: Coordinate = (0.0, 1.0),
        p2: Coordinate = (0.0, 0.0),
    ) -> ComponentReference:
        """Mirrors.

        Args:
            p1: point 1.
            p2: point 2.
        """
        if isinstance(p1, Port):
            p1 = p1.center
        if isinstance(p2, Port):
            p2 = p2.center
        p1 = np.array(p1)
        p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        self.origin = _rotate_points(self.origin, angle=-angle, center=(0, 0))
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin = (self.origin[0], -1 * self.origin[1])

        self.rotation = -1 * self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle=angle, center=(0, 0))
        self.rotation += angle
        self.rotation = self.rotation % 360
        self.origin = self.origin + p1

        self._bb_valid = False
        return self

    def connect(
        self,
        port: Union[str, Port],
        destination: Port,
        overlap: float = 0.0,
        preserve_orientation: bool = False,
    ) -> ComponentReference:
        """Return ComponentReference where port connects to a destination.

        Args:
            port: origin (port, or port name) to connect.
            destination: destination port.
            overlap: how deep does the port go inside.
            preserve_orientation: True, does not rotate the reference to align port
                orientation and reference keep its orientation pre-connection.

        Returns:
            ComponentReference: with correct rotation to connect to destination.
        """
        # port can either be a string with the name, port index, or an actual Port
        if port in self.ports:
            p = self.ports[port]
        elif isinstance(port, Port):
            p = port
        else:
            ports = list(self.ports.keys())
            raise ValueError(
                f"port = {port!r} not in {self.parent.name!r} ports {ports}"
            )

        if (
            destination.orientation is not None
            and p.orientation is not None
            and not preserve_orientation
        ):
            angle = 180 + destination.orientation - p.orientation
            angle = angle % 360
            self.rotate(angle=angle, center=p.center)

        self.move(origin=p, destination=destination)

        if destination.orientation is not None:
            self.move(
                -overlap
                * np.array(
                    [
                        cos(destination.orientation * pi / 180),
                        sin(destination.orientation * pi / 180),
                    ]
                )
            )

        return self

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Return a list of ports.

        Keyword Args:
            layer: port GDS layer.
            prefix: port name prefix.
            orientation: in degrees.
            width: port width.
            layers_excluded: List of layers to exclude.
            port_type: optical, electrical, ...
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return list(select_ports(self.ports, **kwargs).values())

    def get_ports_dict(self, **kwargs) -> Dict[str, Port]:
        """Return a dict of ports.

        Keyword Args:
            layer: port GDS layer.
            prefix: port name prefix.
            orientation: in degrees.
            width: port width.
            layers_excluded: List of layers to exclude.
            port_type: optical, electrical, ...
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return select_ports(self.ports, **kwargs)

    @property
    def ports_layer(self) -> Dict[str, str]:
        """Return a mapping from layer0_layer1_E0: portName."""
        return map_ports_layer_to_orientation(self.ports)

    def port_by_orientation_cw(self, key: str, **kwargs):
        """Return port by indexing them clockwise."""
        m = map_ports_to_orientation_cw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def port_by_orientation_ccw(self, key: str, **kwargs):
        """Return port by indexing them clockwise."""
        m = map_ports_to_orientation_ccw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def get_ports_xsize(self, **kwargs) -> float:
        """Return xdistance from east to west ports.

        Keyword Args:
            kwargs: orientation, port_type, layer.
        """
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return ports_ccw[0].x - ports_cw[0].x

    def get_ports_ysize(self, **kwargs) -> float:
        """Returns ydistance from east to west ports."""
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return ports_ccw[0].y - ports_cw[0].y


def test_move() -> None:
    import gdsfactory as gf

    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler())
    bend.move("o1", mzi.ports["o2"])


def test_get_polygons() -> None:
    import gdsfactory as gf

    ref = gf.components.straight()
    p0 = ref.get_polygons(by_spec="WG", as_array=False)
    p1 = ref.get_polygons(by_spec=(1, 0), as_array=True)
    p2 = ref.get_polygons(by_spec=(1, 0), as_array=False)

    p3 = ref.get_polygons(by_spec=True, as_array=True)[(1, 0)]
    p4 = ref.get_polygons(by_spec=True, as_array=False)[(1, 0)]

    assert len(p1) == len(p2) == len(p3) == len(p4) == 1 == len(p0)
    assert p1[0].dtype == p3[0].dtype == float
    assert isinstance(p2[0], Polygon)
    assert isinstance(p4[0], Polygon)


def test_get_polygons_ref() -> None:
    import gdsfactory as gf

    ref = gf.components.straight().ref()
    p0 = ref.get_polygons(by_spec="WG", as_array=False)
    p1 = ref.get_polygons(by_spec=(1, 0), as_array=True)
    p2 = ref.get_polygons(by_spec=(1, 0), as_array=False)

    p3 = ref.get_polygons(by_spec=True, as_array=True)[(1, 0)]
    p4 = ref.get_polygons(by_spec=True, as_array=False)[(1, 0)]

    assert len(p1) == len(p2) == len(p3) == len(p4) == 1 == len(p0)
    assert p1[0].dtype == p3[0].dtype == float
    assert isinstance(p2[0], Polygon)
    assert isinstance(p4[0], Polygon)


def test_pads_no_orientation() -> None:
    import gdsfactory as gf

    c = gf.Component("pads_no_orientation")
    pt = c << gf.components.pad()
    pb = c << gf.components.pad()
    pb.connect("pad", pt["pad"])


if __name__ == "__main__":
    # test_get_polygons_ref()
    # test_get_polygons()
    import gdsfactory as gf

    c = gf.Component("parent")
    ref = c << gf.components.straight()
    c.add_ports(ref.ports)
    ref.movex(5)
    # assert c.ports['o1'].center[0] == 5, print(c.ports['o1'])
    print(c.ports["o1"].center)
    c.show(show_ports=True)

    # p = ref.get_polygons(by_spec=(1, 0), as_array=False)

    # c = gf.Component("parent")
    # c2 = gf.Component("child")
    # length = 10
    # width = 0.5
    # layer = (1, 0)
    # c2.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    # c << c2

    # c = gf.c.dbr()
    # c.show()

    # import gdsfactory as gf

    # c = gf.Component()
    # mzi = c.add_ref(gf.components.mzi())
    # bend = c.add_ref(gf.components.bend_euler())
    # bend.move("o1", mzi.ports["o2"])
    # bend.move("o1", "o2")
