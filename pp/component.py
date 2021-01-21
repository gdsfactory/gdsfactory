import copy as python_copy
import itertools
import uuid
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy import cos, float64, int64, mod, ndarray, pi, sin
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from phidl.device_layout import Device, DeviceReference, Label, _parse_layer

from pp.compare_cells import hash_cells
from pp.config import conf
from pp.get_netlist import get_netlist
from pp.port import Port, select_ports


def copy(D):
    """returns a copy of a Component."""
    D_copy = Component(name=D._internal_name)
    D_copy.info = python_copy.deepcopy(D.info)
    for ref in D.references:
        new_ref = ComponentReference(
            ref.parent,
            origin=ref.origin,
            rotation=ref.rotation,
            magnification=ref.magnification,
            x_reflection=ref.x_reflection,
        )
        new_ref.owner = D_copy
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
    points: Union[Tuple[int, int], ndarray],
    angle: int = 45,
    center: Tuple[float, float] = (0.0, 0,),
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
        component,
        origin: Tuple[int, int] = (0, 0),
        rotation: int = 0,
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
        self.uid = str(uuid.uuid4())[:8]

    def __repr__(self):
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

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, val):
        """This allows you to access an alias from the reference's parent, and receive
        a copy of the reference which is correctly rotated and translated"""
        try:
            alias_device = self.parent[val]
        except Exception:
            raise ValueError(
                '[PHIDL] Tried to access alias "%s" from parent '
                'Device "%s", which does not exist' % (val, self.parent.name)
            )
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

    def get_labels(
        self, recursive: bool = True, associate_visual_labels: bool = True
    ) -> List[Any]:
        """
        access all labels correctly rotated, mirrored and translated
        """
        # params = {
        #     "recursive": recursive,
        #     "associate_visual_labels": associate_visual_labels,
        # }

        labels_untransformed = self.parent.get_labels()

        labels = []
        visual_label = self.visual_label
        for lbl in labels_untransformed:
            p = lbl.position
            position, _ = self._transform_port(
                p, 0, self.origin, self.rotation, self.x_reflection
            )

            text = lbl.text
            if visual_label and associate_visual_labels:
                text += " / " + visual_label

            new_lbl = Label(
                text=text,
                position=position,
                anchor="o",
                layer=lbl.layer,
                texttype=lbl.texttype,
            )

            labels += [new_lbl]
        if labels:
            print("R", self.parent.name, len(labels))
        return labels

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
    def info(self) -> Dict[str, Union[float64, float]]:
        return self.parent.info

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(self.bbox)

    def _transform_port(
        self,
        point: ndarray,
        orientation: Union[float64, int64],
        origin: Union[Tuple[int, int], ndarray] = (0, 0),
        rotation: Optional[Union[float64, int, int64]] = None,
        x_reflection: bool = False,
    ) -> Union[Tuple[ndarray, float64], Tuple[ndarray, int64]]:
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
        origin: Union[Tuple[int, int], ndarray] = (0, 0),
        rotation: Optional[Union[int64, int, float]] = None,
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
        origin: Union[
            Port,
            ndarray,
            List[Union[float, float64]],
            Tuple[int, int],
            List[Union[int, float]],
        ] = (0, 0),
        destination: Optional[Any] = None,
        axis: Optional[str] = None,
    ):
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
            o = origin.midpoint
        elif np.array(origin).size == 2:
            o = origin
        elif origin in self.ports:
            o = self.ports[origin].midpoint
        else:
            raise ValueError(
                f"{self.parent.name}.move(origin={origin}) not array-like, a port, or port name {list(self.ports.keys())}"
            )

        if hasattr(destination, "midpoint"):
            d = destination.midpoint
        elif np.array(destination).size == 2:
            d = destination
        elif destination in self.ports:
            d = self.ports[destination].midpoint
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
        self, angle: Union[float, int] = 45, center: Tuple[float, float] = (0.0, 0.0),
    ):
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

    def reflect_h(self, port_name=None, x0=None):
        """Perform horizontal mirror using x0 or port as axis (default, x0=0)."""
        if port_name is None and x0 is None:
            x0 = -self.x

        if port_name is not None:
            position = self.ports[port_name]
            x0 = position.x
        self.reflect((x0, 1), (x0, 0))

    def reflect_v(
        self, port_name: Optional[str] = None, y0: Optional[float] = None
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
        p1: Tuple[float, float] = (0.0, 1.0),
        p2: Tuple[float, float] = (0.0, 0.0),
    ):
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

    def connect(self, port: Union[str, Port], destination: Port, overlap: float = 0.0):
        """Returns a reference of the Component where a origin port_name connects to a destination

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
        # if hasattr(destination, "parent"):
        #     add_to_global_netlist(p, destination)
        return self

    def get_property(self, property: str) -> Union[str, int]:
        if hasattr(self, property):
            return self.property

        return self.ref_cell.get_property(property)

    def get_ports_list(self, port_type="optical", prefix=None) -> List[Port]:
        """returns a list of ports. Useful for routing bundles of ports

        Args:
            port_type: str or (int, int) layer
            prefix: for example "E" for east, "W" for west ...
        """
        return list(
            select_ports(self.ports, port_type=port_type, prefix=prefix).values()
        )

    def get_settings(self):
        """Returns settings from the Comonent."""
        return self.parent.get_settings()


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

    def __init__(self, name: str = "Unnamed", *args, **kwargs,) -> None:
        # Allow name to be set like Component('arc') or Component(name = 'arc')

        self.settings = kwargs
        self.settings_changed = kwargs
        self.__ports__ = {}
        self.info = {}
        self.aliases = {}
        self.uid = str(uuid.uuid4())[:8]
        self.ignore = set()
        self.test_protocol = {}
        self.data_analysis_protocol = {}

        if "with_uuid" in kwargs or name == "Unnamed":
            name += "_" + self.uid

        super(Component, self).__init__(name=name, exclude_from_current=True)
        self.name = name
        self.name_long = None

    def plot_netlist(self, with_labels=True, font_weight="normal"):
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
            G, with_labels=with_labels, font_weight=font_weight, labels=labels, pos=pos,
        )

    def get_netlist_yaml(self):
        """Return YAML netlist."""
        return OmegaConf.to_yaml(self.get_netlist())

    def write_netlist(self, filepath, full_settings=False):
        netlist = self.get_netlist(full_settings=full_settings)
        OmegaConf.save(netlist, filepath)

    def get_netlist(self, full_settings=False):
        """Returns netlist dict(instances, placements, connections, ports)

        instances = {instances}
        placements = {instance_name,uid,x,y: dict(x=0, y=0, rotation=90), ...}
        connections = {instance_name_src,uid,x,y,portName,portId: instance_name_dst,uid,x,y,portName,portId}
        ports: {portName: instace_name,portName}

        Args:
            full_settings: exports all the settings, when false only exports settings_changed
        """
        return get_netlist(component=self, full_settings=full_settings)

    def get_name_long(self):
        """ returns the long name if it's been truncated to MAX_NAME_LENGTH"""
        if self.name_long:
            return self.name_long
        else:
            return self.name

    def ports_on_grid(self) -> None:
        """ asserts if all ports ar eon grid """
        for port in self.ports.values():
            port.on_grid()

    def get_ports_dict(self, port_type="optical", prefix=None):
        """ returns a list of ports """
        return select_ports(self.ports, port_type=port_type, prefix=prefix)

    def get_ports_list(self, port_type="optical", prefix=None) -> List[Port]:
        """ returns a lit of  ports """
        return list(
            select_ports(self.ports, port_type=port_type, prefix=prefix).values()
        )

    def get_ports_array(self) -> Dict[str, ndarray]:
        """ returns ports as a dict of np arrays"""
        self.ports_on_grid()
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

    def get_properties(self):
        """ returns name, uid, ports, aliases and numer of references """
        return (
            f"name: {self._internal_name}, uid: {self.uid},  ports:"
            f" {self.ports.keys()}, aliases {self.aliases.keys()}, number of"
            f" references: {len(self.references)}"
        )

    def ref(
        self,
        position: Union[
            Tuple[float, float], Port, Tuple[int, float], ndarray, Tuple[int, int]
        ] = (0, 0),
        port_id: Optional[str] = None,
        rotation: Union[float64, int, int64] = 0,
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
        """ returns a reference of the component centered at (x=0, y=0)"""
        si = self.size_info
        yc = si.south + si.height / 2
        xc = si.west + si.width / 2
        center = (xc, yc)
        _ref = ComponentReference(self)
        _ref.move(center, position)
        return _ref

    def __repr__(self) -> str:
        return f"{self.name}: uid {self.uid}, ports {list(self.ports.keys())}, aliases {list(self.aliases.keys())}, {len(self.polygons)} polygons, {len(self.references)} references"

    def update_settings(self, **kwargs):
        """ update settings dict """
        for key, value in kwargs.items():
            self.settings[key] = _clean_value(value)

    def get_property(self, property: str) -> Union[str, int, float]:
        if property in self.settings:
            return self.settings[property]
        if hasattr(self, property):
            return getattr(self, property)

    def pprint(self, **kwargs):
        """Prints component settings."""
        pprint(self.get_settings(**kwargs))

    def get_settings(
        self,
        ignore=("layer", "layers_cladding", "cladding_offset", "path", "netlist"),
        include=("name", "function_name", "module", "info"),
        full_settings=True,
    ) -> Dict[str, Any]:
        """Returns settings dictionary.
        Ignores items from self.ignore set.

        Args:
            ignore: settings to ignore
            include: settings to include
            full_settings: export full settings or only changed settings

        """
        settings = self.settings if full_settings else self.settings_changed
        d = {}
        d["settings"] = {}  # function arguments
        d["info"] = {}  # function arguments

        include = set(include)
        ignore = set(ignore).union(self.ignore).union(set(dir(Component()))) - include

        params = set(dir(self)) - ignore - include
        for param in params:
            self.info[param] = _clean_value(getattr(self, param))

        # for param in params:
        #     d['info'][param] = _clean_value(getattr(self, param))
        # d["hash"] = hashlib.md5(json.dumps(output).encode()).hexdigest()
        # d["hash_geometry"] = str(self.hash_geometry())

        for setting in include:
            if hasattr(self, setting):
                d[setting] = _clean_value(getattr(self, setting))

        for key, value in settings.items():
            if key not in self.ignore:
                d["settings"][key] = _clean_value(value)
                # print(_clean_value(value))

        d = {k: d[k] for k in sorted(d)}
        return d

    def add_port(
        self,
        name: Optional[Union[str, int]] = None,
        midpoint: Any = (0, 0),
        width: Union[float64, int, float] = 1,
        orientation: Union[int, int64, float] = 45,
        port: Optional[Port] = None,
        layer: Tuple[int, int] = (1, 0),
        port_type: str = "optical",
    ) -> Port:
        """Can be called to copy an existing port like add_port(port = existing_port) or
        to create a new port add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port with a new name like add_port(port = existing_port, name = new_name)"""
        if port:
            if not isinstance(port, Port):
                raise ValueError(
                    f"[PHIDL] add_port({type(port)}) error: Argument `port` must be a Port for"
                    " copying"
                )
            p = port._copy(new_uid=True)
            if name is not None:
                p.name = name
            p.parent = self

        elif isinstance(name, Port):
            p = name._copy(new_uid=True)
            p.parent = self
            name = p.name
        else:
            assert len(layer) == 2, (
                f"Error defining a port with `layer = {layer}`. "
                + "You need to represent GDS layers with two integer numbers (gdslayer, gdspurpose)"
            )
            p = Port(
                name=name,
                midpoint=midpoint,
                width=width,
                orientation=orientation,
                parent=self,
                layer=layer,
                port_type=port_type,
            )
        if name is not None:
            p.name = name
        if p.name in self.ports:
            raise ValueError(
                '[DEVICE] add_port() error: Port name "%s" already exists in this'
                ' Device (name "%s", uid %s)' % (p.name, self._internal_name, self.uid)
            )

        self.ports[p.name] = p
        return p

    def snap_ports_to_grid(self, nm=1):
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
            "git_hash": conf["git_hash"],
            "version": conf["version"],
        }
        jsondata.update(**kwargs)

        if hasattr(self, "analysis"):
            jsondata["analysis"] = self.analysis

        return jsondata

    # def __hash__(self) -> int:
    #     h = dict2hash(**self.settings)
    #     return int(h, 16)

    def hash_geometry(self) -> str:
        """returns geometrical hash"""
        if self.references or self.polygons:
            h = hash_cells(self, {})[self.name]
        else:
            h = "empty_geometry"

        self.settings.update(hash=h)
        return h

    def remove_layers(
        self, layers=(), include_labels=True, invert_selection=False, recursive=True
    ):
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

    def copy(self):
        return copy(self)

    @property
    def size_info(self) -> SizeInfo:
        """ size info of the component """
        # if self.__size_info__ == None:
        # self.__size_info__  = SizeInfo(self.bbox)
        return SizeInfo(self.bbox)  # self.__size_info__

    def add_ref(self, D, alias: Optional[str] = None) -> ComponentReference:
        """Takes a Component and adds it as a ComponentReference to the current
        Device."""
        if type(D) in (list, tuple):
            return [self.add_ref(E) for E in D]
        if not isinstance(D, Component) and not isinstance(D, Device):
            raise TypeError(
                """[PP] add_ref() was passed a {}. This is not a Component object. """.format(
                    type(D)
                )
            )
        d = ComponentReference(D)  # Create a ComponentReference (CellReference)
        self.add(d)  # Add ComponentReference (CellReference) to Device (Cell)

        if alias is not None:
            self.aliases[alias] = d
        return d

    def get_layers(self):
        """returns a set of (layer, datatype)

        .. code ::

            import pp
            pp.c.waveguide().get_layers() == {(1, 0), (111, 0)}

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
        self.show()
        self.plot()
        return self.__str__()

    def plot(self, **kwargs) -> None:
        """Plot component in matplotlib

        Args:
            show_ports: True
            show_subports:
            label_ports:
            label_aliases:
            new_window

        """
        from phidl import quickplot as qp

        from pp.cell import clear_cache

        qp(self, **kwargs)
        clear_cache()

    def show(self) -> None:
        """Show component in klayout"""
        from pp.cell import clear_cache
        from pp.write_component import show

        show(self)
        clear_cache()


def test_get_layers():
    import pp

    c = pp.c.waveguide(layers_cladding=[(111, 0)])
    assert c.get_layers() == {(1, 0), (111, 0)}
    c.remove_layers((111, 0))
    assert c.get_layers() == {(1, 0)}


def _filter_polys(polygons, layers_excl):
    return [
        p
        for p, l, d in zip(polygons.polygons, polygons.layers, polygons.datatypes)
        if (l, d) not in layers_excl
    ]


IGNORE_FUNCTION_NAMES = set()
IGNORE_STRUCTURE_NAME_PREFIXES = set(["zz_conn"])


def recurse_structures(structure: Component) -> Dict[str, Any]:
    """ Recurse over structures """
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


def clean_dict(d):
    """Cleans dictionary keys"""
    from pp.component import _clean_value

    for k, v in d.items():
        if isinstance(v, dict):
            clean_dict(v)
        else:
            d[k] = _clean_value(v)


def _clean_value(value: Any) -> Any:
    """Returns a clean value that is JSON serializable"""
    if type(value) in [int, float, str, bool]:
        return value
    if isinstance(value, (np.int64, np.int32)):
        value = int(value)
    elif isinstance(value, np.float64):
        value = float(value)
    elif callable(value):
        value = value.__name__
    elif hasattr(value, "name"):
        value = value.name
    elif isinstance(value, dict):
        clean_dict(value)
    elif isinstance(value, (tuple, list, ListConfig)):
        value = [_clean_value(i) for i in value]
    elif value is None:
        value = None
    else:
        print(type(value))
        value = str(value)

    return value


def test_same_uid():
    import pp

    c = Component()
    c << pp.c.rectangle()
    c << pp.c.rectangle()

    r1 = c.references[0].parent
    r2 = c.references[1].parent

    print(r1.uid, r2.uid)
    print(r1 == r2)


def test_netlist_simple():
    import pp

    c = pp.Component()
    c1 = c << pp.c.waveguide(length=1, width=1)
    c2 = c << pp.c.waveguide(length=2, width=2)
    c2.connect(port="W0", destination=c1.ports["E0"])
    c.add_port("W0", port=c1.ports["W0"])
    c.add_port("E0", port=c2.ports["E0"])
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 2


def test_netlist_complex():
    import pp

    c = pp.c.mzi()
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 18


def test_netlist_plot():
    import pp

    c = pp.c.mzi()
    c.plot_netlist()


def test_path():
    from pp import CrossSection
    from pp import path as pa

    X1 = CrossSection()
    X1.add(width=1.2, offset=0, layer=2, name="wg", ports=("in1", "out1"))
    X1.add(width=2.2, offset=0, layer=3, name="etch")
    X1.add(width=1.1, offset=3, layer=1, name="wg2")

    # Create the second CrossSection that we want to transition to
    X2 = CrossSection()
    X2.add(width=1, offset=0, layer=2, name="wg", ports=("in2", "out2"))
    X2.add(width=3.5, offset=0, layer=3, name="etch")
    X2.add(width=3, offset=5, layer=1, name="wg2")

    Xtrans = pa.transition(cross_section1=X1, cross_section2=X2, width_type="sine")

    P1 = pa.straight(length=5)
    P2 = pa.straight(length=5)
    WG1 = P1.extrude(cross_section=X1)
    WG2 = P2.extrude(cross_section=X2)

    P4 = pa.euler(radius=25, angle=45, p=0.5, use_eff=False)
    WG_trans = P4.extrude(Xtrans)

    c = Component()
    wg1 = c << WG1
    wg2 = c << WG2
    wgt = c << WG_trans

    wgt.connect("in2", wg1.ports["out1"])
    wg2.connect("in2", wgt.ports["out1"])
    assert len(c.references) == 3


def demo_component(port):
    c = Component()
    c.add_port(name="p1", port=port)
    return c


if __name__ == "__main__":
    import pp

    c = pp.c.bend_circular()
    # c.get_settings()
    c.pprint()

    # c0 = pp.c.waveguide()
    # c = pp.c.waveguide(length=3.0)
    # c.info["c"] = c0
    # c.pprint()

    # import matplotlib.pyplot as plt

    # c = pp.c.ring_single()
    # c = pp.c.mzi()
    # c.plot_netlist()

    # coupler_lengths = [10, 20, 30]
    # coupler_gaps = [0.1, 0.2, 0.3]
    # delta_lengths = [10, 100]

    # c = pp.c.mzi_lattice(
    #     coupler_lengths=coupler_lengths,
    #     coupler_gaps=coupler_gaps,
    #     delta_lengths=delta_lengths,
    # )
    # n = c.get_netlist()
    # print(n.placements)
    # print(n.connections)

    # plt.show()

    # plt.show()

    # test_netlist_simple()
    # test_netlist_complex()

    # c = pp.c.waveguide()
    # print(c.get_settings())
    # c = pp.c.dbr(n=1)

    # print(c.get_layers())

    # c = pp.c.bend_circular180()
    # c = pp.c.coupler()
    # c.add_labels()
    # pp.show(c)
    # test_same_uid()

    # c = pp.c.mmi1x2()
    # c = pp.c.mzi1x2()

    # print(c.hash_geometry())
    # print(c.get_json())
    # print(c.get_settings())
    # print(c.settings)

    # print(c.get_settings())
    # print(c.get_ports_array())

    # print(json.dumps(c.get_settings()))
    # print(c.get_json()['cells'].keys())
    # print(c.get_json())

    # from pp.routing import add_fiber_array
    # cc = add_fiber_array(c)
    # pp.write_component(cc)

    # from pprint import pprint

    # c = pp.c.mmi1x2()
    # cc = add_fiber_array(c)
    # cc.get_json()
    # pp.show(cc)
    # c.update_settings(
    #     analysis={
    #         "device_type": "loopback",
    #         "test_method": "spectrum",
    #         "analysis_keys": "[cell_name, category, 0]",
    #         "polarization_mix": "None",
    #         "device_group_type": "deembed, mapping",
    #         "optical_io_measure": "[[1,4],[2,3]]",
    #         "polarization_input": "TE",
    #         "deembed_device_group_id": "None",
    #     }
    # )

    # w = pp.c.waveguide()
    # c = demo_component(port=w.ports["E0"])
    # pprint(c.get_json())
    # pprint(c.get_settings())

    # c = pp.c.waveguide()
    # c = pp.routing.add_fiber_array(c)
    # c = pp.routing.add_electrical_pads_top(c)
    # print(c)
    # print(c.get_settings()["name"])
    # print(c.get_json())
    # print(c.get_settings(test="hi"))
