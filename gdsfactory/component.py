import copy as python_copy
import datetime
import functools
import hashlib
import itertools
import json
import pathlib
import tempfile
import uuid
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import gdspy
import networkx as nx
import numpy as np
import omegaconf
from numpy import cos, float64, int64, mod, ndarray, pi, sin
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from phidl.device_layout import CellArray, Device, DeviceReference, _parse_layer

from gdsfactory.config import __version__
from gdsfactory.cross_section import CrossSection
from gdsfactory.port import (
    Port,
    auto_rename_ports,
    auto_rename_ports_layer_orientation,
    auto_rename_ports_orientation,
    map_ports_layer_to_orientation,
    map_ports_to_orientation_ccw,
    map_ports_to_orientation_cw,
    select_ports,
)
from gdsfactory.snap import snap_to_grid


class MutabilityError(ValueError):
    pass


Number = Union[float64, int64, float, int]
Coordinate = Union[Tuple[Number, Number], ndarray, List[Number]]
Coordinates = Union[List[Coordinate], ndarray, List[Number], Tuple[Number, ...]]
PathType = Union[str, Path]

tmp = pathlib.Path(tempfile.TemporaryDirectory().name).parent / "gdsfactory"
tmp.mkdir(exist_ok=True)
_timestamp2019 = datetime.datetime.fromtimestamp(1572014192.8273)
MAX_NAME_LENGTH = 32


def copy(D: Device) -> Device:
    """returns a deep copy of a Component.
    based on phidl.geometry with CellArray support
    """
    D_copy = Component(name=D.name)
    D_copy.info = python_copy.deepcopy(D.info)
    for ref in D.references:
        if isinstance(ref, DeviceReference):
            new_ref = ComponentReference(
                ref.parent,
                origin=ref.origin,
                rotation=ref.rotation,
                magnification=ref.magnification,
                x_reflection=ref.x_reflection,
            )
            new_ref.owner = D_copy
        elif isinstance(ref, gdspy.CellArray):
            new_ref = CellArray(
                device=ref.parent,
                columns=ref.columns,
                rows=ref.rows,
                spacing=ref.spacing,
                origin=ref.origin,
                rotation=ref.rotation,
                magnification=ref.magnification,
                x_reflection=ref.x_reflection,
            )
        D_copy.add(new_ref)
        for alias_name, alias_ref in D.aliases.items():
            if alias_ref == ref:
                D_copy.aliases[alias_name] = new_ref

    for port in D.ports.values():
        D_copy.add_port(port=port)
    for poly in D.polygons:
        D_copy.add_polygon(poly)
    for label in D.labels:
        D_copy.add_label(
            text=label.text,
            position=label.position,
            layer=(label.layer, label.texttype),
        )
    return D_copy


class SizeInfo:
    def __init__(self, bbox: ndarray) -> None:
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
    ):
        w, e, s, n = self.west, self.east, self.south, self.north

        padding_n = padding if padding_n is None else padding_n
        padding_e = padding if padding_e is None else padding_e
        padding_w = padding if padding_w is None else padding_w
        padding_s = padding if padding_s is None else padding_s

        w = w - padding_w
        e = e + padding_e
        s = s - padding_s
        n = n + padding_n

        return [(w, s), (e, s), (e, n), (w, n)]

    @property
    def rect(self):
        return self.get_rect()

    def __str__(self):
        return "w: {}\ne: {}\ns: {}\nn: {}\n".format(
            self.west, self.east, self.south, self.north
        )


def _rotate_points(
    points: Coordinates,
    angle: Number = 45,
    center: Coordinate = (
        0.0,
        0.0,
    ),
) -> ndarray:
    """Rotates points around a centerpoint defined by ``center``.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind
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


class ComponentReference(DeviceReference):
    def __init__(
        self,
        component: Device,
        origin: Coordinate = (0, 0),
        rotation: Number = 0,
        magnification: None = None,
        x_reflection: bool = False,
        visual_label: str = "",
    ) -> None:
        super().__init__(
            device=component,
            origin=origin,
            rotation=rotation,
            magnification=magnification,
            x_reflection=x_reflection,
        )
        self.parent = component
        # The ports of a DeviceReference have their own unique id (uid),
        # since two DeviceReferences of the same parent Device can be
        # in different locations and thus do not represent the same port
        self._local_ports = {
            name: port._copy(new_uid=True) for name, port in component.ports.items()
        }
        self.visual_label = visual_label
        # self.uid = str(uuid.uuid4())[:8]

    def __repr__(self) -> str:
        return (
            'DeviceReference (parent Device "%s", ports %s, origin %s, rotation %s,'
            " x_reflection %s)"
            % (
                self.parent.name,
                list(self.ports.keys()),
                self.origin,
                self.rotation,
                self.x_reflection,
            )
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, val):
        """This allows you to access an alias from the reference's parent, and receive
        a copy of the reference which is correctly rotated and translated"""
        try:
            alias_device = self.parent[val]
        except Exception as exc:
            raise ValueError(
                '[PHIDL] Tried to access alias "%s" from parent '
                'Device "%s", which does not exist' % (val, self.parent.name)
            ) from exc
        new_reference = ComponentReference(
            alias_device.parent,
            origin=alias_device.origin,
            rotation=alias_device.rotation,
            magnification=alias_device.magnification,
            x_reflection=alias_device.x_reflection,
        )

        if self.x_reflection:
            new_reference.reflect((1, 0))
        if self.rotation is not None:
            new_reference.rotate(self.rotation)
        if self.origin is not None:
            new_reference.move(self.origin)

        return new_reference

    @property
    def ports(self) -> Dict[str, Port]:
        """This property allows you to access myref.ports, and receive a copy
        of the ports dict which is correctly rotated and translated"""
        for name, port in self.parent.ports.items():
            port = self.parent.ports[name]
            new_midpoint, new_orientation = self._transform_port(
                port.midpoint,
                port.orientation,
                self.origin,
                self.rotation,
                self.x_reflection,
            )
            if name not in self._local_ports:
                self._local_ports[name] = port._copy(new_uid=True)
            self._local_ports[name].midpoint = new_midpoint
            self._local_ports[name].orientation = mod(new_orientation, 360)
            self._local_ports[name].parent = self
        # Remove any ports that no longer exist in the reference's parent
        parent_names = self.parent.ports.keys()
        local_names = list(self._local_ports.keys())
        for name in local_names:
            if name not in parent_names:
                self._local_ports.pop(name)
        return self._local_ports

    @property
    def info(self) -> Dict[str, Any]:
        return self.parent.info

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(self.bbox)

    def _transform_port(
        self,
        point: ndarray,
        orientation: Number,
        origin: Coordinate = (0, 0),
        rotation: Optional[Number] = None,
        x_reflection: bool = False,
    ) -> Tuple[ndarray, Number]:
        # Apply GDS-type transformations to a port (x_ref)
        new_point = np.array(point)
        new_orientation = orientation

        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = -orientation
        if rotation is not None:
            new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)
        new_orientation = mod(new_orientation, 360)

        return new_point, new_orientation

    def _transform_point(
        self,
        point: ndarray,
        origin: Coordinate = (0, 0),
        rotation: Optional[Number] = None,
        x_reflection: bool = False,
    ) -> ndarray:
        # Apply GDS-type transformations to a port (x_ref)
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
        origin: Union[Port, Coordinate] = (0, 0),
        destination: Optional[Any] = None,
        axis: Optional[str] = None,
    ) -> "ComponentReference":
        """Moves the DeviceReference from the origin point to the destination.
        Both origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this device_ref

        Returns:
            ComponentReference
        """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = (0, 0)

        if hasattr(origin, "midpoint"):
            origin = cast(Port, origin)
            o = origin.midpoint
        elif np.array(origin).size == 2:
            o = origin
        elif origin in self.ports:
            origin = self.ports[origin]
            origin = cast(Port, origin)
            o = origin.midpoint
        else:
            raise ValueError(
                f"{self.parent.name}.move(origin={origin}) not array-like, a port, or port name {list(self.ports.keys())}"
            )

        if hasattr(destination, "midpoint"):
            destination = cast(Port, destination)
            d = destination.midpoint
        elif np.array(destination).size == 2:
            d = destination
        elif destination in self.ports:
            destination = self.ports[destination]
            destination = cast(Port, destination)
            d = destination.midpoint
        else:
            raise ValueError(
                f"{self.parent.name}.move(destination={destination}) not array-like, a port, or port name {list(self.ports.keys())}"
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
        angle: Number = 45,
        center: Coordinate = (0.0, 0.0),
    ) -> "ComponentReference":
        """Return ComponentReference rotated:

        Args:
            angle: in degrees
            center: x,y
        """
        if angle == 0:
            return self
        if isinstance(center, str) or isinstance(center, int):
            center = self.ports[center].position

        if isinstance(center, Port):
            center = center.midpoint
        self.rotation += angle
        self.rotation = self.rotation % 360
        self.origin = _rotate_points(self.origin, angle, center)
        self._bb_valid = False
        return self

    def reflect_h(
        self, port_name: Optional[str] = None, x0: Optional[Coordinate] = None
    ) -> None:
        """Perform horizontal mirror using x0 or port as axis (default, x0=0)."""
        if port_name is None and x0 is None:
            x0 = -self.x

        if port_name is not None:
            position = self.ports[port_name]
            x0 = position.x
        self.reflect((x0, 1), (x0, 0))

    def reflect_v(
        self, port_name: Optional[str] = None, y0: Optional[Number] = None
    ) -> None:
        """Perform vertical mirror using y0 as axis (default, y0=0)"""
        if port_name is None and y0 is None:
            y0 = 0.0

        if port_name is not None:
            position = self.ports[port_name]
            y0 = position.y
        self.reflect((1, y0), (0, y0))

    def reflect(
        self,
        p1: Coordinate = (0.0, 1.0),
        p2: Coordinate = (0.0, 0.0),
    ) -> "ComponentReference":
        if isinstance(p1, Port):
            p1 = p1.midpoint
        if isinstance(p2, Port):
            p2 = p2.midpoint
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
        self.origin[1] = -1 * self.origin[1]
        self.rotation = -1 * self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle=angle, center=(0, 0))
        self.rotation += angle
        self.rotation = self.rotation % 360
        self.origin = self.origin + p1

        self._bb_valid = False
        return self

    def connect(
        self, port: Union[str, Port], destination: Port, overlap: Number = 0.0
    ) -> "ComponentReference":
        """Returns Component reference
        origin port_name connects to a destination

        Args:
            port: origin port name
            destination: destination port
            overlap: how deep does the port go inside

        Returns:
            ComponentReference
        """
        # ``port`` can either be a string with the name or an actual Port
        if port in self.ports:  # Then ``port`` is a key for the ports dict
            p = self.ports[port]
        elif isinstance(port, Port):
            p = port
        else:
            raise ValueError(
                f"{self.parent.name}.connect({port}): {port} not in {list(self.ports.keys())}"
            )

        angle = 180 + destination.orientation - p.orientation
        angle = angle % 360
        self.rotate(angle=angle, center=p.midpoint)

        self.move(origin=p, destination=destination)
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

    def get_property(self, property: str) -> Union[str, int]:
        if hasattr(self, property):
            return self.property

        return self.ref_cell.get_property(property)

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Returns a list of ports.

        Args:
            layer: port GDS layer
            prefix:
            orientation:
        """
        return list(select_ports(self.ports, **kwargs).values())

    def get_ports_dict(self, **kwargs) -> List[Port]:
        """Returns a list of ports.

        Args:
            layer: port GDS layer
            prefix: for example "E" for east, "W" for west ...
        """
        return select_ports(self.ports, **kwargs)

    def get_settings(self, **kwargs) -> Dict[str, Any]:
        """Returns settings from the Comonent."""
        return self.parent.get_settings(**kwargs)

    @property
    def ports_layer(self) -> Dict[str, str]:
        """Returns a mapping from layer0_layer1_E0: portName"""
        return map_ports_layer_to_orientation(self.ports)

    def port_by_orientation_cw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise"""
        m = map_ports_to_orientation_cw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def port_by_orientation_ccw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise"""
        m = map_ports_to_orientation_ccw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]


class Component(Device):
    """adds some functions to phidl.Device:

    - get/write JSON metadata
    - get ports by type (optical, electrical ...)
    - set data_analysis and test_protocols

    Args:
        name:
        polarization: 'te' or 'tm'
        wavelength: (nm)
        test_protocol: dict
        data_analysis_protocol: dict
        ignore: list of settings to ingnore

    """

    def __init__(self, name: str = "Unnamed", *args, **kwargs) -> None:
        # Allow name to be set like Component('arc') or Component(name = 'arc')

        self.settings = kwargs
        self.settings_changed = kwargs
        self.__ports__ = {}
        self.info = {}
        self.aliases = {}
        self.uid = str(uuid.uuid4())[:8]
        self.ignore = {
            "path",
            "netlist",
            "properties",
            "library",
            "_initialized",
            "component",
        }
        self.include = {"name", "function_name", "module"}
        self.test_protocol = {}
        self.data_analysis_protocol = {}

        if "with_uuid" in kwargs or name == "Unnamed":
            name += "_" + self.uid

        super(Component, self).__init__(name=name, exclude_from_current=True)
        self.name = name
        self.name_long = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """a valid component:
        - name characters < MAX_NAME_LENGTH
        - is not empty (has references or polygons)
        """
        assert isinstance(v, Component)
        assert (
            len(v.name) <= MAX_NAME_LENGTH
        ), f"name `{v.name}` {len(v.name)} > {MAX_NAME_LENGTH} "
        assert v.references or v.polygons, f"No references or  polygons in {v.name}"
        return v

    @property
    def ports_layer(self) -> Dict[str, str]:
        """Returns a mapping from layer0_layer1_E0: portName"""
        return map_ports_layer_to_orientation(self.ports)

    def port_by_orientation_cw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise"""
        m = map_ports_to_orientation_cw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def port_by_orientation_ccw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise"""
        m = map_ports_to_orientation_ccw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def get_ports_xsize(self, **kwargs) -> float:
        """Returns a the xdistance from east to west ports

        Args:
            kwargs: orientation, port_type, layer
        """
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return snap_to_grid(ports_ccw[0].x - ports_cw[0].x)

    def get_ports_ysize(self, **kwargs) -> float:
        """Returns a the ydistance from east to west ports"""
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return snap_to_grid(ports_ccw[0].y - ports_cw[0].y)

    def plot_netlist(
        self, with_labels: bool = True, font_weight: str = "normal"
    ) -> nx.Graph:
        """plots a netlist graph with networkx
        https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

        Args:
            with_labels: label nodes
            font_weight: normal, bold
        """
        netlist = self.get_netlist()
        connections = netlist["connections"]
        placements = netlist["placements"]

        G = nx.Graph()

        G.add_edges_from(
            [
                (",".join(k.split(",")[:-1]), ",".join(v.split(",")[:-1]))
                for k, v in connections.items()
            ]
        )

        pos = {k: (v["x"], v["y"]) for k, v in placements.items()}
        labels = {k: ",".join(k.split(",")[:1]) for k in placements.keys()}
        nx.draw(
            G,
            with_labels=with_labels,
            font_weight=font_weight,
            labels=labels,
            pos=pos,
        )
        return G

    def get_netlist_yaml(self) -> str:
        """Return YAML netlist."""
        return OmegaConf.to_yaml(self.get_netlist())

    def write_netlist(self, filepath: str, full_settings: bool = False) -> None:
        """Write netlist in YAML"""
        netlist = self.get_netlist(full_settings=full_settings)
        OmegaConf.save(netlist, filepath)

    def write_netlist_dot(self, filepath: Optional[str] = None) -> None:
        """Write netlist graph in DOT format."""
        from networkx.drawing.nx_agraph import write_dot

        filepath = filepath or f"{self.name}.dot"

        G = self.plot_netlist()
        write_dot(G, filepath)

    def get_netlist(self, full_settings: bool = False) -> Any:
        """Returns netlist dict(instances, placements, connections, ports)

        instances = {instances}
        placements = {instance_name,uid,x,y: dict(x=0, y=0, rotation=90), ...}
        connections = {instance_name_src_x_y,portName: instance_name_dst_x_y,portName}
        ports: {portName: instace_name,portName}

        Args:
            full_settings: exports all settings, when false only settings_changed
        """
        from gdsfactory.get_netlist import get_netlist

        return get_netlist(component=self, full_settings=full_settings)

    def get_name_long(self) -> str:
        """returns the long name if it's been truncated to MAX_NAME_LENGTH"""
        if self.name_long:
            return self.name_long
        else:
            return self.name

    def assert_ports_on_grid(self, nm: int = 1) -> None:
        """Asserts that all ports are on grid."""
        for port in self.ports.values():
            port.assert_on_grid(nm=nm)

    def get_ports_dict(self, **kwargs) -> Dict[str, Port]:
        """Returns a dict of ports.

        Args:
            layer: port GDS layer
            prefix: for example "E" for east, "W" for west ...
        """
        return select_ports(self.ports, **kwargs)

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Returns a list of ports.

        Args:
            layer: port GDS layer
            prefix: for example "E" for east, "W" for west ...
            orientation: angle in degrees for the port
        """
        return list(select_ports(self.ports, **kwargs).values())

    def get_ports_array(self) -> Dict[str, ndarray]:
        """returns ports as a dict of np arrays"""
        ports_array = {
            port_name: np.array(
                [
                    port.x,
                    port.y,
                    int(port.orientation),
                    port.width,
                    port.layer[0],
                    port.layer[1],
                ]
            )
            for port_name, port in self.ports.items()
        }
        return ports_array

    def ref(
        self,
        position: Coordinate = (0, 0),
        port_id: Optional[str] = None,
        rotation: int = 0,
        h_mirror: bool = False,
        v_mirror: bool = False,
    ) -> ComponentReference:
        """Returns Component reference."""
        _ref = ComponentReference(self)

        if port_id and port_id not in self.ports:
            raise ValueError(f"port {port_id} not in {self.ports.keys()}")

        if port_id:
            origin = self.ports[port_id].position
        else:
            origin = (0, 0)

        if h_mirror:
            _ref.reflect_h(port_id)

        if v_mirror:
            _ref.reflect_v(port_id)

        if rotation != 0:
            _ref.rotate(rotation, origin)
        _ref.move(origin, position)

        return _ref

    def ref_center(self, position=(0, 0)):
        """returns a reference of the component centered at (x=0, y=0)"""
        si = self.size_info
        yc = si.south + si.height / 2
        xc = si.west + si.width / 2
        center = (xc, yc)
        _ref = ComponentReference(self)
        _ref.move(center, position)
        return _ref

    def __repr__(self) -> str:
        return f"{self.name}: uid {self.uid}, ports {list(self.ports.keys())}, aliases {list(self.aliases.keys())}, {len(self.polygons)} polygons, {len(self.references)} references"

    def update_settings(self, **kwargs) -> None:
        """update settings dict"""
        for key, value in kwargs.items():
            self.settings[key] = _clean_value(value)

    def get_property(self, property: str) -> Any:
        if property in self.settings:
            return self.settings[property]
        if hasattr(self, property):
            return getattr(self, property)

    @property
    def pprint(self) -> None:
        """Prints component settings."""
        pprint(self.get_settings())

    @property
    def pprint_ports(self) -> None:
        """Prints component netlists."""
        ports_list = self.get_ports_list()
        for port in ports_list:
            print(port)

    def get_settings(
        self,
        ignore: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
        full_settings: bool = True,
    ) -> Dict[str, Any]:
        """Returns settings dictionary.

        Ignores items from self.ignore set.

        Returns *args, **kwargs from the signature, as well as any attributes

        attributes (such as Component.length) are exported into info: Dict[str, Any]

        Args:
            ignore: settings to ignore
            include: settings to include
            full_settings: export full settings or only changed settings

        """
        settings = self.settings if full_settings else self.settings_changed
        d = {}
        d["settings"] = {}  # function arguments
        d["info"] = {}  # function arguments

        ignore_keys = ignore or set()
        include_keys = include or set()
        ignore_keys = set(ignore_keys)
        include_keys = set(include_keys)

        include = set(include_keys).union(self.include) - ignore_keys
        ignore = (
            set(ignore_keys).union(self.ignore).union(set(dir(Component()))) - include
        )

        params = set(dir(self)) - ignore - ignore_keys - include

        # Properties from self.info and self.someThing
        for param in params:
            d["info"][param] = _clean_value(getattr(self, param))

        for key, value in self.info.items():
            if key not in ignore:
                d["info"][key] = _clean_value(value)

        # TOP Level (name, module, function_name)
        for setting in include:
            if hasattr(self, setting) and setting not in ignore:
                d[setting] = _clean_value(getattr(self, setting))

        # Settings from the function call
        for key, value in settings.items():
            if key not in ignore:
                d["settings"][key] = _clean_value(value)

        # for param in params:
        #     d['info'][param] = _clean_value(getattr(self, param))
        # d["hash"] = hashlib.md5(json.dumps(output).encode()).hexdigest()
        # d["hash_geometry"] = str(self.hash_geometry())

        # if 'tech' in d['settings']:
        #     d['tech']= getattr(self,'tech').dict()

        d = {k: d[k] for k in sorted(d)}
        return d

    def add_port(
        self,
        name: Optional[Union[str, int, object]] = None,
        midpoint: Tuple[float, float] = (
            0.0,
            0.0,
        ),
        width: float = 1.0,
        orientation: int = 45,
        port: Optional[Port] = None,
        layer: Tuple[int, int] = (1, 0),
        port_type: str = "optical",
        cross_section: Optional[CrossSection] = None,
    ) -> Port:
        """Can be called to copy an existing port like add_port(port = existing_port) or
        to create a new port add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port
        with a new name add_port(port = existing_port, name = new_name)"""

        if port:
            if not isinstance(port, Port):
                raise ValueError(f"add_port() needs a Port, got {type(port)}")
            p = port._copy(new_uid=True)
            if name is not None:
                p.name = name
            p.parent = self

        elif isinstance(name, Port):
            p = name._copy(new_uid=True)
            p.parent = self
            name = p.name
        else:
            half_width = width / 2
            half_width_correct = snap_to_grid(half_width, nm=1)
            if not np.isclose(half_width, half_width_correct):
                warnings.warn(
                    f"port width = {width} will create off-grid points.\n"
                    f"You can fix it by changing width to {2*half_width_correct}\n"
                    f"port {name}, {midpoint}  {orientation} deg",
                    stacklevel=3,
                )
            p = Port(
                name=name,
                midpoint=(snap_to_grid(midpoint[0]), snap_to_grid(midpoint[1])),
                width=snap_to_grid(width),
                orientation=orientation,
                parent=self,
                layer=layer,
                port_type=port_type,
                cross_section=cross_section,
            )
        if name is not None:
            p.name = name
        if p.name in self.ports:
            raise ValueError(f"add_port() Port name {p.name} exists in {self.name}")

        self.ports[p.name] = p
        return p

    def add_ports(self, ports: Union[List[Port], Dict[str, Port]], prefix: str = ""):
        ports = ports if isinstance(ports, list) else ports.values()
        for port in ports:
            name = f"{prefix}{port.name}" if prefix else port.name
            self.add_port(name=name, port=port)

    def snap_ports_to_grid(self, nm: int = 1) -> None:
        for port in self.ports.values():
            port.snap_to_grid(nm=nm)

    def get_json(self, **kwargs) -> Dict[str, Any]:
        """
        Returns:
            Dict with component metadata
        """
        jsondata = {
            "json_version": 7,
            "cells": recurse_structures(self),
            "test_protocol": self.test_protocol,
            "data_analysis_protocol": self.data_analysis_protocol,
            "version": __version__,
        }
        jsondata.update(**kwargs)

        if hasattr(self, "analysis"):
            jsondata["analysis"] = self.analysis

        return jsondata

    def remove_layers(
        self,
        layers: Union[List[Tuple[int, int]], Tuple[int, int]] = (),
        include_labels: bool = True,
        invert_selection: bool = False,
        recursive: bool = True,
    ) -> Device:
        """Remove a list of layers."""
        layers = [_parse_layer(layer) for layer in layers]
        all_D = list(self.get_dependencies(recursive))
        all_D += [self]
        for D in all_D:
            for polygonset in D.polygons:
                polygon_layers = zip(polygonset.layers, polygonset.datatypes)
                polygons_to_keep = [(pl in layers) for pl in polygon_layers]
                if not invert_selection:
                    polygons_to_keep = [(not p) for p in polygons_to_keep]
                polygonset.polygons = [
                    p for p, keep in zip(polygonset.polygons, polygons_to_keep) if keep
                ]
                polygonset.layers = [
                    p for p, keep in zip(polygonset.layers, polygons_to_keep) if keep
                ]
                polygonset.datatypes = [
                    p for p, keep in zip(polygonset.datatypes, polygons_to_keep) if keep
                ]

            if include_labels:
                new_labels = []
                for label in D.labels:
                    original_layer = (label.layer, label.texttype)
                    original_layer = _parse_layer(original_layer)
                    if invert_selection:
                        keep_layer = original_layer in layers
                    else:
                        keep_layer = original_layer not in layers
                    if keep_layer:
                        new_labels += [label]
                D.labels = new_labels
        return self

    def extract(
        self,
        layers: Union[List[Tuple[int, int]], Tuple[int, int]] = (),
    ) -> Device:
        """Extract polygons from a Component.
        adapted from phidl.geometry.
        """
        from gdsfactory.name import clean_value

        component = Component(f"{self.name}_{clean_value(layers)}")
        if type(layers) not in (list, tuple):
            raise ValueError("layers needs to be a list or tuple")
        poly_dict = self.get_polygons(by_spec=True)
        parsed_layer_list = [_parse_layer(layer) for layer in layers]
        for layer, polys in poly_dict.items():
            if _parse_layer(layer) in parsed_layer_list:
                component.add_polygon(polys, layer=layer)
        return component

    def copy(self) -> Device:
        return copy(self)

    @property
    def size_info(self) -> SizeInfo:
        """size info of the component"""
        # if self.__size_info__ == None:
        # self.__size_info__  = SizeInfo(self.bbox)
        return SizeInfo(self.bbox)  # self.__size_info__

    def add_ref(self, D: Device, alias: Optional[str] = None) -> "ComponentReference":
        """Takes a Component and adds it as a ComponentReference to the current
        Device."""
        if not isinstance(D, Component) and not isinstance(D, Device):
            raise TypeError(
                f"Component.add_ref() type = {type(D)} needs to be a Component."
            )
        ref = ComponentReference(D)  # Create a ComponentReference (CellReference)
        self.add(ref)  # Add ComponentReference (CellReference) to Device (Cell)

        if alias is not None:
            self.aliases[alias] = ref
        return ref

    def get_layers(self) -> Union[Set[Tuple[int, int]], Set[Tuple[int64, int64]]]:
        """returns a set of (layer, datatype)

        .. code ::

            import gdsfactory as gf
            gf.components.straight().get_layers() == {(1, 0), (111, 0)}

        """
        layers = set()
        for element in itertools.chain(self.polygons, self.paths):
            for layer, datatype in zip(element.layers, element.datatypes):
                layers.add((layer, datatype))
        for reference in self.references:
            for layer, datatype in reference.ref_cell.get_layers():
                layers.add((layer, datatype))
        for label in self.labels:
            layers.add((label.layer, 0))
        return layers

    def _repr_html_(self):
        """Print component, show geometry in matplotlib and in klayout
        when using jupyter notebooks
        """
        self.show(show_ports=False)
        self.plot()
        return self.__str__()

    def plot(
        self,
        clear_cache: bool = False,
    ) -> None:
        """Plot component in matplotlib"""
        from phidl import quickplot as plot

        from gdsfactory.cell import clear_cache as clear_cache_function

        plot(self)
        if clear_cache:
            clear_cache_function()

    def show(
        self,
        show_ports: bool = True,
        show_subports: bool = False,
        clear_cache: bool = False,
    ) -> None:
        """Show component in klayout

        if show_subports = True
        We add pins in a new component
        that contains a reference to the old component
        so we don't modify the original component

        Args:
            show_ports: shows component with port markers and labels
            show_subports: add ports markers and labels to component references
            clear_cache: after showing component clears cache (useful for jupyter)
        """
        from gdsfactory.add_pins import add_pins_container, add_pins_to_references
        from gdsfactory.show import show

        if show_subports:
            component = add_pins_to_references(component=self)
            component.name = self.name + "_show_subports"

        elif show_ports:
            component = add_pins_container(component=self)
            component.name = self.name + "_show_ports"
        else:
            component = self

        show(component, clear_cache=clear_cache)

    def plotqt(self):
        from phidl.quickplotter import quickplot2

        quickplot2(self)

    def write_gds(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: PathType = tmp,
        unit: float = 1e-6,
        precision: float = 1e-9,
        auto_rename: bool = False,
        timestamp: Optional[datetime.datetime] = _timestamp2019,
    ) -> Path:
        """Write component to GDS and returs gdspath

        Args:
            component: gf.Component.
            gdspath: GDS file path to write to.
            unit unit size for objects in library.
            precision: for the dimensions of the objects in the library (m).
            remove_previous_markers: clear previous ones to avoid duplicates.
            auto_rename: If True, fixes any duplicate cell names.
            timestamp: datetime object or boolean
                Sets the GDSII timestamp. Default = 2019-10-25 07:36:32.827300
                If None, defaults to Now.

        Returns:
            gdspath
        """
        gdsdir = pathlib.Path(gdsdir)
        gdspath = gdspath or gdsdir / (self.name + ".gds")
        gdspath = pathlib.Path(gdspath)
        gdsdir = gdspath.parent
        gdsdir.mkdir(exist_ok=True, parents=True)

        cells = self.get_dependencies()
        cell_names = [cell.name for cell in list(cells)]
        cell_names_unique = set(cell_names)

        if len(cell_names) != len(set(cell_names)):
            for cell_name in cell_names_unique:
                cell_names.remove(cell_name)

            cell_names_duplicated = "\n".join(set(cell_names))
            raise ValueError(
                f"Duplicated cell names in {self.name}:\n{cell_names_duplicated}"
            )

        referenced_cells = list(self.get_dependencies(recursive=True))
        all_cells = [self] + referenced_cells

        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        lib.write_gds(gdspath, cells=all_cells, timestamp=timestamp)
        self.path = gdspath
        return gdspath

    def write_gds_with_metadata(self, *args, **kwargs) -> Path:
        """Write component in GDS, ports in CSV and metadata (component settings) in JSON"""
        gdspath = self.write_gds(*args, **kwargs)
        ports_path = gdspath.with_suffix(".ports")
        json_path = gdspath.with_suffix(".json")

        # write component.ports to CSV
        if len(self.ports) > 0:
            with open(ports_path, "w") as fw:
                for port in self.ports.values():
                    layer, purpose = _parse_layer(port.layer)
                    fw.write(
                        f"{port.name}, {port.x:.3f}, {port.y:.3f}, {int(port.orientation)}, {port.width:.3f}, {layer}, {purpose}\n"
                    )

        # write component.json metadata dict to JSON
        json_path.write_text(json.dumps(self.get_json(), indent=2))

        # with open(json_path, "w+") as fw:
        #     fw.write(json.dumps(self.get_json(), indent=2))
        # metadata = omegaconf.OmegaConf.create(self.get_json())
        # json_path.write_text( OmegaConf.to_container(metadata))
        return gdspath

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dict representation of the compoment."""
        d = {}
        d["polygons"] = {}
        d["ports"] = {}
        layer_to_polygons = self.get_polygons(by_spec=True)

        for layer, polygons in layer_to_polygons.items():
            for polygon in polygons:
                layer_name = f"{layer[0]}_{layer[1]}"
                d["polygons"][layer_name] = [tuple(snap_to_grid(v)) for v in polygon]

        for port in self.get_ports_list():
            d["ports"][port.name] = port.settings

        d["settings"] = self.get_settings()["settings"]
        return d

    def auto_rename_ports(self, **kwargs) -> None:
        auto_rename_ports(self, **kwargs)

    def auto_rename_ports_layer_orientation(self, **kwargs) -> None:
        auto_rename_ports_layer_orientation(self, **kwargs)

    def auto_rename_ports_orientation(self, **kwargs) -> None:
        auto_rename_ports_orientation(self, **kwargs)

    def move(self, *args, **kwargs):
        raise MutabilityError(
            "Don't move Components. Create a reference and move the reference instead."
        )

    def rotate(self, *args, **kwargs):
        raise MutabilityError(
            "Don't rotate Components. Create a reference and move the reference instead."
        )


def test_get_layers() -> None:
    import gdsfactory as gf

    c = gf.components.straight(
        length=10, width=0.5, layer=(2, 0), layers_cladding=((111, 0),)
    )
    assert c.get_layers() == {(2, 0), (111, 0)}, c.get_layers()
    c.remove_layers((111, 0))
    assert c.get_layers() == {(2, 0)}, c.get_layers()
    return c


def _filter_polys(polygons, layers_excl):
    return [
        p
        for p, l, d in zip(polygons.polygons, polygons.layers, polygons.datatypes)
        if (l, d) not in layers_excl
    ]


IGNORE_FUNCTION_NAMES = set()
IGNORE_STRUCTURE_NAME_PREFIXES = set(["zz_conn"])


def recurse_structures(structure: Component) -> Dict[str, Any]:
    """Recurse over structures"""
    if (
        hasattr(structure, "function_name")
        and structure.function_name in IGNORE_FUNCTION_NAMES
    ):
        return {}

    if hasattr(structure, "name") and any(
        [structure.name.startswith(i) for i in IGNORE_STRUCTURE_NAME_PREFIXES]
    ):
        return {}
    if not hasattr(structure, "get_json"):
        return {}

    output = {structure.name: structure.get_settings()}
    for element in structure.references:
        if (
            isinstance(element, ComponentReference)
            and element.ref_cell.name not in output
        ):
            output.update(recurse_structures(element.ref_cell))

    return output


def clean_dict(d: Dict[str, Any]) -> None:
    """Cleans dictionary keys recursively."""
    for k, v in d.items():
        if isinstance(v, dict):
            clean_dict(v)
        else:
            d[k] = _clean_value(v)


def clean_key(key):
    print(type(key), key)
    if isinstance(key, tuple):
        key = key[0]
    else:
        key = str(key)

    return key


def _clean_value(value: Any) -> Any:
    """Returns a clean value that is JSON serializable"""
    if isinstance(value, CrossSection):
        value = value.info
    if isinstance(value, float) and float(int(value)) == value:
        value = int(value)
    if type(value) in [int, float, str, bool]:
        return value
    if isinstance(value, (np.int64, np.int32)):
        value = int(value)
    elif isinstance(value, np.float64):
        value = float(value)
    elif callable(value) and hasattr(value, "__name__"):
        value = value.__name__
    elif callable(value) and isinstance(value, functools.partial):
        v = value.keywords.copy()
        v.update(function=value.func.__name__)
        value = _clean_value(v)
    elif isinstance(value, dict):
        clean_dict(value)
    elif isinstance(value, omegaconf.dictconfig.DictConfig):
        clean_dict(value)
    elif isinstance(value, (tuple, list, ListConfig)):
        value = [_clean_value(i) for i in value]
    elif value is None:
        value = None
    elif hasattr(value, "name"):
        value = value.name
    else:
        value = str(value)

    return value


def test_same_uid() -> None:
    import gdsfactory as gf

    c = Component()
    c << gf.components.rectangle()
    c << gf.components.rectangle()

    r1 = c.references[0].parent
    r2 = c.references[1].parent

    print(r1.uid, r2.uid)
    print(r1 == r2)


def test_netlist_simple() -> None:
    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=1)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect(port="o1", destination=c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 2


def test_netlist_complex() -> None:
    import gdsfactory as gf

    c = gf.components.mzi()
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 4, len(netlist["instances"])


def test_netlist_plot() -> None:
    import gdsfactory as gf

    c = gf.components.mzi()
    c.plot_netlist()


def test_extract():
    import gdsfactory as gf

    c = gf.components.straight(length=10, width=0.5, layers_cladding=[gf.LAYER.WGCLAD])
    c2 = c.extract(layers=[gf.LAYER.WGCLAD])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 1, len(c2.polygons)


def hash_file(filepath):
    md5 = hashlib.md5()
    md5.update(filepath.read_bytes())
    return md5.hexdigest()


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    print(c)
