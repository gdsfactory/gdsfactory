import uuid
import json
import hashlib
from copy import deepcopy
import numpy as np
from numpy import pi, sin, cos, mod

import gdspy

from phidl.device_layout import Label
from phidl.device_layout import Device
from phidl.device_layout import DeviceReference
from phidl.device_layout import _parse_layer
from phidl.device_layout import Port as PortPhidl

from pp.ports import select_optical_ports
from pp.ports import select_electrical_ports
import pp
from pp.config import CONFIG
from pp.compare_cells import hash_cells

NAME_TO_DEVICE = {}

BBOX_LAYER_EXCLUDE = CONFIG["BBOX_LAYER_EXCLUDE"]


class Port(PortPhidl):
    """ Extends phidl port by adding layer and a port_type (optical, electrical)

    Args:
        name: we name ports according to orientation (S0, S1, W0, W1, N0 ...)
        midpoint: (0, 0)
        width: of the port
        orientation: 0
        parent: None, parent component (component to which this port belong to)
        layer: 1
        port_type: optical, dc, rf, detector, superconducting, trench

    """

    _next_uid = 0

    def __init__(
        self,
        name=None,
        midpoint=(0, 0),
        width=1,
        orientation=0,
        parent=None,
        layer=1,
        port_type="optical",
    ):
        self.name = name
        self.midpoint = np.array(midpoint, dtype="float64")
        self.width = width
        self.orientation = mod(orientation, 360)
        self.parent = parent
        self.info = {}
        self.uid = Port._next_uid
        self.layer = layer
        self.port_type = port_type

        if self.width < 0:
            raise ValueError("[PHIDL] Port creation error: width must be >=0")
        self._next_uid += 1

    def __repr__(self):
        return "Port (name {}, midpoint {}, width {}, orientation {}, layer {}, port_type {})".format(
            self.name,
            self.midpoint,
            self.width,
            self.orientation,
            self.layer,
            self.port_type,
        )

    @property
    def angle(self):
        """convenient alias"""
        return self.orientation

    @angle.setter
    def angle(self, a):
        self.orientation = a

    @property
    def position(self):
        return self.midpoint

    @position.setter
    def position(self, p):
        self.midpoint = np.array(p, dtype="float64")

    def move(self, vector):
        self.midpoint = self.midpoint + np.array(vector)

    def move_polar_copy(self, d, angle):
        port = self._copy()
        DEG2RAD = np.pi / 180
        dp = np.array((d * np.cos(DEG2RAD * angle), d * np.sin(DEG2RAD * angle)))
        self.move(dp)
        return port

    def flip(self):
        """ flips port """
        port = self._copy()
        port.angle = (port.angle + 180) % 360
        return port

    def _copy(self, new_uid=True):
        new_port = Port(
            name=self.name,
            midpoint=self.midpoint,
            width=self.width,
            orientation=self.orientation,
            parent=self.parent,
            layer=self.layer,
            port_type=self.port_type,
        )
        new_port.info = deepcopy(self.info)
        if not new_uid:
            new_port.uid = self.uid
            Port._next_uid -= 1
        return new_port


class SizeInfo:
    def __init__(self, bbox):
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


def _rotate_points(points, angle=45, center=(0, 0)):
    """ Rotates points around a centerpoint defined by ``center``.  ``points`` may be
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
        device,
        origin=(0, 0),
        rotation=0,
        magnification=None,
        x_reflection=False,
        visual_label="",
    ):
        super().__init__(
            device=device,
            origin=origin,
            rotation=rotation,
            magnification=magnification,
            x_reflection=x_reflection,
        )
        self.parent = device
        # The ports of a DeviceReference have their own unique id (uid),
        # since two DeviceReferences of the same parent Device can be
        # in different locations and thus do not represent the same port
        self._local_ports = {
            name: port._copy(new_uid=True) for name, port in device.ports.items()
        }
        self.visual_label = visual_label

    def __repr__(self):
        return (
            'DeviceReference (parent Device "%s", ports %s, origin %s, rotation %s, x_reflection %s)'
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
        """ This allows you to access an alias from the reference's parent, and receive
        a copy of the reference which is correctly rotated and translated"""
        try:
            alias_device = self.parent[val]
        except:
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

    def get_labels(self, recursive=True, associate_visual_labels=True):
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
    def ports(self):
        """ This property allows you to access myref.ports, and receive a copy
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
    def info(self):
        return self.parent.info

    # @property
    # def bbox(self):
    # bbox = self.get_bounding_box()
    # if bbox is None:
    # bbox = ((0, 0), (0, 0))
    # return np.array(bbox)

    # def get_bounding_box(self):
    # """
    # WARNING ! This does not work as well as GDSPY method. But is cheaper.
    # Comment this method if packing breaks

    # the bounding box of a reference is the transformed
    # bounding box of the parent cell
    # """

    # _bbox = self.parent.get_bounding_box()
    # bbox = np.array([self._transform_point(
    # p, self.origin, self.rotation, self.x_reflection
    # ) for p in _bbox])
    # return bbox

    def get_bounding_box(self):
        """
        Returns the bounding box for this reference.

        Returns
        -------
        out : Numpy array[2,2] or ``None``
            Bounding box of this cell [[x_min, y_min], [x_max, y_max]],
            or ``None`` if the cell is empty.
        """
        if not isinstance(self.ref_cell, gdspy.Cell):
            return None
        if (
            self.rotation is None
            and self.magnification is None
            and self.x_reflection is None
        ):
            key = self.ref_cell
        else:
            key = (self.ref_cell, self.rotation, self.magnification, self.x_reflection)

        if key not in gdspy._bounding_boxes:
            """
            The bounding boxes are all valid, but this specific transformation
            has not been registered yet. Fetch bbox, and apply transform
            """
            _bb = self.ref_cell.get_bounding_box()
            if _bb is not None:
                bb = np.array(
                    [
                        self._transform_point(
                            p, (0, 0), self.rotation, self.x_reflection
                        )
                        for p in _bb
                    ]
                )
                _x1, _x2 = bb[0, 0], bb[1, 0]
                _y1, _y2 = bb[0, 1], bb[1, 1]

                bb[0, 0] = min(_x1, _x2)
                bb[1, 0] = max(_x1, _x2)
                bb[0, 1] = min(_y1, _y2)
                bb[1, 1] = max(_y1, _y2)
                gdspy._bounding_boxes[key] = bb

            else:
                bb = None
        else:
            bb = gdspy._bounding_boxes[key]
        if self.origin is None or bb is None:
            return bb
        else:
            translation = np.array(
                [(self.origin[0], self.origin[1]), (self.origin[0], self.origin[1])]
            )
            return bb + translation

    @property
    def bbox(self):
        _bbox = self.ref_cell.get_bounding_box()
        bbox = np.array(
            [
                self._transform_point(p, self.origin, self.rotation, self.x_reflection)
                for p in _bbox
            ]
        )

        # Rearrange min,max
        bbox[1][0], bbox[0][0] = (
            max(bbox[1][0], bbox[0][0]),
            min(bbox[1][0], bbox[0][0]),
        )
        bbox[1][1], bbox[0][1] = (
            max(bbox[1][1], bbox[0][1]),
            min(bbox[1][1], bbox[0][1]),
        )

        return bbox

    @property
    def size_info(self):
        return SizeInfo(self.bbox)
        # if self.__size_info__ == None:
        # return self.__size_info__

    def _transform_port(
        self, point, orientation, origin=(0, 0), rotation=None, x_reflection=False
    ):
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

    def _transform_point(self, point, origin=(0, 0), rotation=None, x_reflection=False):
        # Apply GDS-type transformations to a port (x_ref)
        new_point = np.array(point)

        if x_reflection:
            new_point[1] = -new_point[1]
        if rotation is not None:
            new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
        if origin is not None:
            new_point = new_point + np.array(origin)

        return new_point

    def move(self, origin=(0, 0), destination=None, axis=None):
        """ Moves the DeviceReference from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this device_ref """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = (0, 0)

        if isinstance(origin, Port):
            o = origin.midpoint
        elif np.array(origin).size == 2:
            o = origin
        elif origin in self.ports:
            o = self.ports[origin].midpoint
        else:
            raise ValueError(
                "[ComponentReference.move()] ``origin`` not array-like, a port, or port name"
            )

        if isinstance(destination, Port):
            d = destination.midpoint
        elif np.array(destination).size == 2:
            d = destination
        elif destination in self.ports:
            d = self.ports[destination].midpoint
        else:
            raise ValueError(
                "[ComponentReference.move()] ``destination`` not array-like, a port, or port name"
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

    def rotate(self, angle=45, center=(0, 0)):
        if angle == 0:
            return self
        if type(center) == str or type(center) == int:
            center = self.ports[center].position

        if type(center) is Port:
            center = center.midpoint
        self.rotation += angle
        self.rotation = self.rotation % 360
        self.origin = _rotate_points(self.origin, angle, center)
        self._bb_valid = False
        return self

    def reflect_h(self, port_name=None, x0=None):
        """
        Perform horizontal mirror (w.r.t vertical axis)
        """
        if port_name is None and x0 is None:
            x0 = 0

        if port_name is not None:
            position = self.ports[port_name]
            x0 = position.x
        self.reflect((x0, 1), (x0, 0))

    def reflect_v(self, port_name=None, y0=None):
        """
        Perform vertical mirror (w.r.t horizontal axis)
        """
        if port_name is None and y0 is None:
            y0 = 0

        if port_name is not None:
            position = self.ports[port_name]
            y0 = position.y
        self.reflect((1, y0), (0, y0))

    def reflect(self, p1=(0, 1), p2=(0, 0)):
        if type(p1) is Port:
            p1 = p1.midpoint
        if type(p2) is Port:
            p2 = p2.midpoint
        p1 = np.array(p1)
        p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        self.origin = _rotate_points(self.origin, angle=-angle, center=[0, 0])
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle=angle, center=[0, 0])
        self.rotation += angle
        self.rotation = self.rotation % 360
        self.origin = self.origin + p1

        self._bb_valid = False
        return self

    def connect(self, port, destination, overlap=0):
        # ``port`` can either be a string with the name or an actual Port
        if port in self.ports:  # Then ``port`` is a key for the ports dict
            p = self.ports[port]
        elif type(port) is Port:
            p = port
        else:
            raise ValueError(
                f"{self}.connect({port}) valid ports are {self.ports.keys()}"
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

    def get_property(self, property):
        if hasattr(self, property):
            return self.property

        return self.ref_cell.get_property(property)


class Component(Device):
    """ inherits from phidl.Device and extends some functionality

    - get/write JSON metadata
    - get ports by type (optical, electrical ...)
    - set data_analysis and test_protocols

    """
    def __init__(self, *args, **kwargs):
        # Allow name to be set like Component('arc') or Component(name = 'arc')

        if "name" in kwargs:
            _internal_name = kwargs.pop("name")
        elif (len(args) == 1) and (len(kwargs) == 0):
            _internal_name = args[0]
        else:
            _internal_name = "Unnamed"

        # print(_internal_name, len(args), len(kwargs))

        self.data_analysis_protocol = kwargs.pop("data_analysis_protocol", {})
        self.test_protocol = kwargs.pop("test_protocol", {})

        self.settings = kwargs
        self.__ports__ = {}
        self.info = {}
        self.aliases = {}
        self.uid = str(uuid.uuid4())[:8]
        self._internal_name = _internal_name
        gds_name = _internal_name

        if "with_uuid" in kwargs or _internal_name == "Unnamed":
            gds_name += "_" + self.uid

        super(Component, self).__init__(name=gds_name, exclude_from_current=True)
        self.name = gds_name

    def get_optical_ports(self):
        """ returns a lit of optical ports """
        return list(select_optical_ports(self.ports).values())

    def get_electrical_ports(self):
        """ returns a list of optical ports """
        return list(select_electrical_ports(self.ports).values())

    def get_properties(self):
        """ returns name, uid, ports, aliases and numer of references """
        return f"name: {self._internal_name}, uid: {self.uid},  ports: {self.ports.keys()}, aliases {self.aliases.keys()}, number of references: {len(self.references)}"

    def ref(
        self, position=(0, 0), port_id=None, rotation=0, h_mirror=False, v_mirror=False
    ):
        """ returns a reference of the component """
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

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def update_settings(self, **kwargs):
        """ update settings dict """
        for key, value in kwargs.items():
            self.settings[key] = _clean_value(value)

    def get_property(self, property):
        if property in self.settings:
            return self.settings[property]
        if hasattr(self, property):
            return getattr(self, property)

        raise ValueError(
            "Component {} does not have property {}".format(self.name, property)
        )

    def get_settings(self, **kwargs):
        """ Returns settings dictionary
        """
        output = {}
        ignore = set(
            dir(Component())
            + ["path", "settings", "properties", "function_name", "type"]
        )
        params = set(dir(self)) - ignore
        output["name"] = self.name
        if hasattr(self, "function_name"):
            output["class"] = self.function_name

        for key, value in kwargs.items():
            output[key] = _clean_value(value)

        for key, value in self.settings.items():
            output[key] = _clean_value(value)

        for param in params:
            value = getattr(self, param)
            output[param] = _clean_value(value)

        output["hash"] = hashlib.md5(json.dumps(output).encode()).hexdigest()
        # output["hash_geometry"] = str(self.hash_geometry())
        return output

    def add_port(
        self,
        name=None,
        midpoint=(0, 0),
        width=1,
        orientation=45,
        port=None,
        layer=1,
        port_type="optical",
    ):
        """ Can be called to copy an existing port like add_port(port = existing_port) or
        to create a new port add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port with a new name like add_port(port = existing_port, name = new_name)"""
        if port:
            if not isinstance(port, Port):
                print(type(port))
                raise ValueError(
                    "[PHIDL] add_port() error: Argument `port` must be a Port for copying"
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
                '[DEVICE] add_port() error: Port name "%s" already exists in this Device (name "%s", uid %s)'
                % (p.name, self._internal_name, self.uid)
            )

        self.ports[p.name] = p
        return p

    def get_json(self, **kwargs):
        """ returns JSON metadata """
        jsondata = {
            "json_version": 6,
            "cells": recurse_structures(self),
            "test_protocol": self.test_protocol,
            "data_analysis_protocol": self.data_analysis_protocol,
            "git_hash": pp.CONFIG["git_hash"],
        }

        if hasattr(self, "analysis"):
            jsondata["analysis"] = self.analysis

        return jsondata

    def hash_geometry(self):
        """ returns geometrical hash
        """
        if self.references or self.polygons:
            return hash_cells(self, {})[self.name]
        else:
            return "empty_geometry"

    def remove_layers(
        self, layers=(), include_labels=True, invert_selection=False, recursive=True
    ):
        """ remove a list of layers """
        layers = [_parse_layer(l) for l in layers]
        all_D = list(self.get_dependencies(recursive))
        all_D += [self]
        for D in all_D:
            for polygonset in D.polygons:
                polygon_layers = zip(polygonset.layers, polygonset.datatypes)
                polygons_to_keep = [(pl in layers) for pl in polygon_layers]
                if invert_selection == False:
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

            if include_labels == True:
                new_labels = []
                for l in D.labels:
                    original_layer = (l.layer, l.texttype)
                    original_layer = _parse_layer(original_layer)
                    if invert_selection:
                        keep_layer = original_layer in layers
                    else:
                        keep_layer = original_layer not in layers
                    if keep_layer:
                        new_labels += [l]
                D.labels = new_labels
        return self

    @property
    def size_info(self):
        """ size info of the component """
        # if self.__size_info__ == None:
        # self.__size_info__  = SizeInfo(self.bbox)
        return SizeInfo(self.bbox)  # self.__size_info__

    def add_ref(self, D, alias=None):
        """ Takes a Device and adds it as a ComponentReference to the current
        Device.  """
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

    def get_bounding_box(self, layers_excl=BBOX_LAYER_EXCLUDE, force=False):
        """
        Calculate the bounding box for this cell.

        Returns
            out : Numpy array[2, 2] or None
                Bounding box of this cell [[x_min, y_min], [x_max, y_max]],
                or None if the cell is empty.
        """
        if (
            len(self.polygons) == 0
            and len(self.paths) == 0
            and len(self.references) == 0
        ):
            return None

        if force or (
            not (
                self._bb_valid
                and all(ref._bb_valid for ref in self.get_dependencies(True))
            )
        ):
            bb = None  # np.array(((1e300, 1e300), (-1e300, -1e300)))
            all_polygons = []
            for polygon in self.polygons:

                polygons = _filter_polys(polygon, layers_excl)

                all_polygons.extend(polygons)
            for path in self.paths:
                all_polygons.extend(path.to_polygonset().polygons)
            for reference in self.references:
                reference_bb = reference.get_bounding_box()
                if reference_bb is not None:
                    if bb is None:
                        bb = reference_bb.copy()
                    else:
                        bb[0, 0] = min(bb[0, 0], reference_bb[0, 0])
                        bb[0, 1] = min(bb[0, 1], reference_bb[0, 1])
                        bb[1, 0] = max(bb[1, 0], reference_bb[1, 0])
                        bb[1, 1] = max(bb[1, 1], reference_bb[1, 1])

            if len(all_polygons) > 0:
                all_points = np.concatenate(all_polygons).transpose()
                if bb is None:
                    bb = np.zeros([2, 2])
                    bb[0, 0] = all_points[0].min()
                    bb[0, 1] = all_points[1].min()
                    bb[1, 0] = all_points[0].max()
                    bb[1, 1] = all_points[1].max()
                else:
                    bb[0, 0] = min(bb[0, 0], all_points[0].min())
                    bb[0, 1] = min(bb[0, 1], all_points[1].min())
                    bb[1, 0] = max(bb[1, 0], all_points[0].max())
                    bb[1, 1] = max(bb[1, 1], all_points[1].max())
            self._bb_valid = True
            gdspy._bounding_boxes[self] = bb
        return gdspy._bounding_boxes[self]


def _filter_polys(polygons, layers_excl):
    return [
        p
        for p, l, d in zip(polygons.polygons, polygons.layers, polygons.datatypes)
        if (l, d) not in layers_excl
    ]


IGNORE_FUNCTION_NAMES = set()
IGNORE_STRUCTURE_NAME_PREFIXES = set(["zz_conn"])


def recurse_structures(structure):
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

    # if hasattr(structure, "function_name"):
    #     print(structure.function_name)
    output = {structure.name: structure.get_settings()}
    for element in structure.references:
        if (
            isinstance(element, ComponentReference)
            and element.ref_cell.name not in output
        ):
            output.update(recurse_structures(element.ref_cell))

    return output


def _clean_value(value):
    """ returns a clean value """
    if type(value) in [int, float, str]:
        value = value
    elif callable(value):
        value = value.__name__
    elif type(value) == pp.Component:
        value = value.name
    elif hasattr(value, "__iter__") and type(value) != str:
        value = "_".join(["{}".format(i) for i in value]).replace(".", "p")
    elif isinstance(value, np.int32):
        value = int(value)
    elif isinstance(value, np.int64):
        value = int(value)
    else:
        value = str(value)

    return value


def demo_component(port):
    c = pp.Component()
    c.add_port(name="p1", port=port)
    return c


if __name__ == "__main__":
    c = pp.c.waveguide()
    # c = pp.c.ring_double_bus()
    # print(c.hash_geometry())
    # print(c.get_json())
    # print(c.get_settings())
    print(c.settings)

    # from pprint import pprint
    # from pp.routing import add_io_optical

    # c = pp.c.mmi1x2()
    # cc = add_io_optical(c)
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

    # print(c.get_json())
    # print(c.get_settings())
    # print(c.get_settings(test="hi"))
    # pp.show(c)
