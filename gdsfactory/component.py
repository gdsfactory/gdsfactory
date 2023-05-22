"""Component is a canvas for geometry.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import tempfile
import datetime
import hashlib
import itertools
import math
import pathlib
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gdstk
import kfactory as kf
import numpy as np
import yaml
from kfactory import kdb
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Literal

from gdsfactory.component_layout import (
    Label,
    Polygon,
    _align,
    _distribute,
    _parse_layer,
)
from gdsfactory.config import CONF, logger, GDSDIR_TEMP
from gdsfactory.cross_section import CrossSection
from gdsfactory.port import (
    Port,
    port_to_kport,
    auto_rename_ports,
    auto_rename_ports_counter_clockwise,
    auto_rename_ports_layer_orientation,
    auto_rename_ports_orientation,
    map_ports_layer_to_orientation,
    map_ports_to_orientation_ccw,
    map_ports_to_orientation_cw,
    select_ports,
)
from gdsfactory.serialization import clean_dict
from gdsfactory.snap import snap_to_grid
from gdsfactory.technology import LayerView, LayerViews
from gdsfactory.generic_tech import LAYER

Plotter = Literal["holoviews", "matplotlib", "qt", "klayout"]
Axis = Literal["x", "y"]

Number = Union[float, int]
Coordinate = Union[Tuple[Number, Number], np.ndarray, List[Number]]


class SizeInfo:
    def __init__(self, bbox: np.ndarray) -> None:
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


class MutabilityError(ValueError):
    pass


def _get_dependencies(component, references_set) -> None:
    for ref in component.references:
        references_set.add(ref.ref_cell)
        _get_dependencies(ref.ref_cell, references_set)


mutability_error_message = """
You cannot modify a Component after creation as it will affect all of its instances.

Create a new Component and add a reference to it.

For example:

# BAD
c = gf.components.bend_euler()
c.add_ref(gf.components.mzi())

# GOOD
c = gf.Component()
c.add_ref(gf.components.bend_euler())
c.add_ref(gf.components.mzi())
"""

PathType = Union[str, Path]
Float2 = Tuple[float, float]
Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
LayerSpec = Union[str, int, Layer, None]

_timestamp2019 = datetime.datetime.fromtimestamp(1572014192.8273)
MAX_NAME_LENGTH = 32


def _rnd(arr, precision=1e-4):
    arr = np.ascontiguousarray(arr)
    ndigits = round(-math.log10(precision))
    return np.ascontiguousarray(arr.round(ndigits) / precision, dtype=np.int64)


class Instance(kf.Instance):
    def mirror(self):
        self.trans.mirror = False if self.trans.is_mirror() else True

    def rotate(self, angle, origin: Optional[Tuple[float]] = (0, 0)):
        self.transform(kf.kdb.DTrans(angle, False, *origin))

    @property
    def xsize(self):
        return self.cell._kdb_cell.dbbox().width()

    @property
    def ysize(self):
        return self.cell._kdb_cell.dbbox().height()

    @property
    def size_info(self):
        dbbox = self.cell._kdb_cell.dbbox()
        return SizeInfo(
            np.asarray([[dbbox.p1.x, dbbox.p2.x], [dbbox.p1.y, dbbox.p2.y]])
        )

    def connect(
        self,
        port: str,
        destination: ComponentReference | Port,
        destination_name: Optional[str] = None,
        *,
        mirror: bool = False,
        allow_width_mismatch: bool = False,
        allow_layer_mismatch: bool = False,
        allow_type_mismatch: bool = False,
    ) -> None:
        """Function to allow to transform this instance so that a port of this instance is connected (same position with 180° turn) to another instance.

        Args:
            portname: The name of the port of this instance to be connected
            other_instance: The other instance or a port
            other_port_name: The name of the other port. Ignored if :py:attr:`~other_instance` is a port.
            mirror: Instead of applying klayout.db.Trans.R180 as a connection transformation, use klayout.db.Trans.M90, which effectively means this instance will be mirrored and connected.
        """
        import gdsfactory as gf

        portname = port
        other = destination
        other_port_name = destination_name

        if isinstance(other, ComponentReference):
            if other_port_name is None:
                raise ValueError(
                    "portname cannot be None if an Instance Object is given"
                )
            op = other.ports[other_port_name]
        p = self.cell.ports[portname]
        if isinstance(p, gf.Port):
            p = port_to_kport(p, library=self.cell.kcl)
        if isinstance(other, gf.Port):
            op = port_to_kport(other, library=self.cell.kcl)
            print(op)
            if isinstance(port, kf.Port):
                op = kf.Port(port=op)
        else:
            op = other

        if p.width != op.width and not allow_width_mismatch:
            raise kf.kcell.PortWidthMismatch(
                self,
                other,
                p,
                op,
            )
        elif (
            gf.get_layer(p.layer) != gf.get_layer(op.layer) and not allow_layer_mismatch
        ):
            raise kf.kcell.PortLayerMismatch(self.cell.kcl, self, other, p, op)
        elif p.port_type != op.port_type and not allow_type_mismatch:
            raise kf.kcell.PortTypeMismatch(self, other, p, op)
        else:
            self.align(p, op)

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Returns list of ports.

        Keyword Args:
            layer: select ports with GDS layer.
            prefix: select ports with port name.
            orientation: select ports with orientation in degrees.
            width: select ports with port width.
            layers_excluded: List of layers to exclude.
            port_type: select ports with port_type (optical, electrical, vertical_te).
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return list(select_ports(self.ports, **kwargs).values())

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
                raise ValueError(f"{origin} not in {self.ports.get_all_named.keys()}")

            origin = self.ports[origin]
            # origin = cast(Port, origin)
            o = origin.center
        elif hasattr(origin, "center"):
            # origin = cast(Port, origin)
            o = origin.center
        elif np.array(origin).size == 2:
            o = origin
        else:
            raise ValueError(
                f"move(origin={origin})\n"
                f"Invalid origin = {origin!r} needs to be"
                f"a coordinate, port or port name {list(self.ports.copy().get_all_named().keys())}"
            )

        if isinstance(destination, str):
            if destination not in self.ports:
                raise ValueError(
                    f"{destination} not in {self.ports.get_all_named.keys()}"
                )

            destination = self.ports[destination]
            d = destination.center
        if hasattr(destination, "center"):
            d = destination.center
        elif np.array(destination).size == 2:
            d = destination

        else:
            raise ValueError(
                f"{self.parent.name}.move(destination={destination!r}) \n"
                f"Invalid destination = {destination!r} needs to be"
                f"a coordinate, a port, or a valid port name {list(self.ports.get_all_named.keys())}"
            )

        # Lock one axis if necessary
        if axis == "x":
            d = (d[0], o[1])
        if axis == "y":
            d = (o[0], d[1])

        dxdy = np.array(d) - np.array(o)

        self = self.transform(
            kf.kdb.Trans(0, False, dxdy[0] / self.kcl.dbu, dxdy[1] / self.kcl.dbu)
        )
        return self

    @classmethod
    def __get_validators__(cls):
        """Get validators for the Component object."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Pydantic assumes component is valid if the following are true."""
        MAX_NAME_LENGTH = 100
        assert isinstance(
            v, ComponentReference
        ), f"TypeError, Got {type(v)}, expecting Instance"
        assert (
            len(v.name) <= MAX_NAME_LENGTH
        ), f"name `{v.name}` {len(v.name)} > {MAX_NAME_LENGTH} "
        return v


ComponentReference = Instance


class Component(kf.KCell):
    """A Component is an empty canvas where you add polygons, references and ports \
            (to connect to other components).

    - stores settings that you use to build the component
    - stores info that you want to use
    - can return ports by type (optical, electrical ...)
    - can return netlist for circuit simulation
    - can write to GDS, OASIS
    - can show in KLayout, matplotlib, 3D, QT viewer, holoviews
    - can return copy, mirror, flattened (no references)

    Args:
        name: component_name. Use @cell decorator for auto-naming.
        with_uuid: adds unique identifier.

    Properties:
        info: dictionary that includes
            - derived properties
            - external metadata (test_protocol, docs, ...)
            - simulation_settings
            - function_name
            - name: for the component

        settings:
            full: full settings passed to the function to create component.
            changed: changed settings.
            default: default component settings.
            child: dict info from the children, if any.
    """

    # def __init__(
    #     self,
    #     name: str = "Unnamed",
    #     with_uuid: bool = False,
    # ) -> None:
    #     """Initialize the Component object."""
    #     self.uid = str(uuid.uuid4())[:8]
    #     if with_uuid or name == "Unnamed":
    #         name += f"_{self.uid}"
    #     self.name = name
    #     self.info: Dict[str, Any] = {} # user added
    #     self.settings: Dict[str, Any] = {} # cell decorator adds this
    #     self._locked = False
    #     self.get_child_name = False
    #     self._reference_names_counter = Counter()
    #     self._reference_names_used = set()
    #     self._references = []
    #     self.ports = {}
    #     self._named_references = {}

    def copy(self) -> "Component":
        """Copy the full component.

        Returns:
            cell: exact copy of the current cell.
        """
        kdb_copy = self._kdb_cell.dup()
        c = Component(library=self.kcl, kdb_cell=kdb_copy)
        c.ports = self.ports
        for inst in self.insts:
            c.create_inst(inst.cell, inst.instance.trans)
        c._locked = False
        return c

    def create_inst(self, cell: kf.KCell, trans: kdb.Trans = kdb.Trans()) -> Instance:
        """Add an instance of another KCell.

        Args:
            cell: The cell to be added.
            trans: The transformation applied to the reference.

        Returns:
            :py:class:`~Instance`: The created instance.
        """
        from gdsfactory.pdk import get_layer

        ca = (
            self.insert(kdb.CellInstArray(cell._kdb_cell, trans))
            if not isinstance(cell, Label)
            else kf.Instance(
                self.kcl,
                self.kcl.insert(
                    self._kdb_cell.cell_index(),
                    self.kcl.layer(get_layer("TEXT")[0], get_layer("TEXT")[1]),
                    kdb.Texts([cell.to_Text()]),
                ),
            )
        )
        kcl = cell.kcl if not isinstance(cell, Label) else self.kcl
        inst = Instance(kcl, ca)
        self.insts.append(inst)
        return inst

    def __lshift__(self, cell: kf.KCell) -> Instance:
        """Convenience function for :py:attr:"~create_inst(cell)`.

        Args:
            cell: The cell to be added as an instance.
        """
        return self.create_inst(cell)

    @property
    def references(self):
        return self.insts

    # @property
    # def polygons(self) -> List[Polygon]:
    #     return self._cell.polygons

    @property
    def area(self) -> float:
        return self.bbox().area()

    # @property
    # def labels(self) -> List[Label]:
    #     return self._cell.labels

    # @property
    # def paths(self):
    #     return self._cell.paths

    # @property
    # def name(self) -> str:
    #     return self._cell.name

    # @name.setter
    # def name(self, value):
    #     self._cell.name = value

    def __iter__(self):
        """You can iterate over polygons, paths, labels and references."""
        return itertools.chain(self.polygons, self.paths, self.labels, self.references)

    def get_polygons(
        self,
        by_spec: Union[bool, Tuple[int, int]] = False,
        recursive: bool = True,
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
            recursive: integer or None
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
        if recursive:
            if by_spec:
                layer = self.kcl.layer(*by_spec)
                return list(kdb.Region(self.begin_shapes_rec(layer)).each())

            else:
                return {
                    (layer_info.layer, layer_info.datatype): list(
                        kdb.Region(self.begin_shapes_rec(layer_index)).each()
                    )
                    for layer_index, layer_info in zip(
                        self.kcl.layer_indexes(), self.kcl.layer_infos()
                    )
                }
        else:
            if by_spec:
                return [
                    p.polygon
                    for p in self.shapes(self.kcl.layer(*by_spec)).each(
                        kdb.Shapes.SRegions
                    )
                ]

            else:
                return {
                    (info.layer, info.datatype): [
                        p.polygon for p in self.shapes(index).each(kdb.Shapes.SRegions)
                    ]
                    for info, index in zip(
                        self.kcl.layer_infos(), self.kcl.layer_indexes()
                    )
                }

    def get_dependencies(self, recursive: bool = False) -> List[Component]:
        """Return a set of the cells included in this cell as references.

        Args:
            recursive: If True returns dependencies recursively.

        Returns:
            out: list of Components referenced by this Component.
        """
        if not recursive:
            return list({ref.parent for ref in self.references})

        references_set = set()
        _get_dependencies(self, references_set=references_set)
        return list(references_set)

    def get_component_spec(self):
        return (
            {
                "component": self.settings.function_name,
                "settings": self.settings.changed,
            }
            if self.settings
            else {"component": self.name, "settings": {}}
        )

    def __getitem__(self, key):
        """Access reference ports."""
        if key not in self.ports:
            ports = list(self.ports.get_all_named.keys())
            raise ValueError(f"{key!r} not in {ports}")

        return self.ports[key]

    def unlock(self) -> None:
        """Only do this if you know what you are doing."""
        self._locked = False

    def lock(self) -> None:
        """Makes sure components can't add new elements or move existing ones.

        Components lock automatically when going into the CACHE to
        ensure one component does not change others
        """
        self._locked = True

    def __setitem__(self, key, element):
        """Allow adding polygons and cell references.

        like D['arc3'] = pg.arc()

        Args:
            key: Alias name.
            element: Object that will be accessible by alias name.

        """
        if isinstance(element, (ComponentReference, Polygon)):
            self.named_references[key] = element
        else:
            raise ValueError(
                f"Tried to assign alias {key!r} in Component {self.name!r},"
                "but failed because the item was not a ComponentReference"
            )

    @classmethod
    def __get_validators__(cls):
        """Get validators for the Component object."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Pydantic assumes component is valid if the following are true.

        - name characters < MAX_NAME_LENGTH
        - is not empty (has references or polygons)
        """
        MAX_NAME_LENGTH = 99
        assert isinstance(
            v, Component
        ), f"TypeError, Got {type(v)}, expecting Component"
        assert (
            len(v.name) <= MAX_NAME_LENGTH
        ), f"name `{v.name}` {len(v.name)} > {MAX_NAME_LENGTH} "
        return v

    @property
    def named_references(self):
        return self.references

    def add_label(
        self,
        text: str = "hello",
        position: Tuple[float, float] = (0.0, 0.0),
        height: float = 1.0,
        rotation: float = 0,
        layer="TEXT",
        x_reflection=False,
    ) -> Label:
        """Adds Label to the Component.

        Args:
            text: Label text.
            position: x-, y-coordinates of the Label location.
            height: height the Label text.
            rotation: Angle rotation of the Label text.
            layer: Specific layer(s) to put Label on.
            x_reflection: True reflects across the horizontal axis before rotation.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        gds_layer, gds_datatype = layer

        if not isinstance(text, str):
            text = text

        x, y = position
        trans = kdb.DTrans(int(rotation / 90), x_reflection, x, y)
        label = kdb.DText(text, trans)
        label.height = height
        layer = self.kcl.layer(*layer)
        self.shapes(layer).insert(label)
        return label

    @property
    def bbox(self):
        """Returns the bounding box of the ComponentReference.

        it snaps to 3 decimals in um (0.001um = 1nm precision)
        """
        bbox = [
            [self._kdb_cell.dbbox().p1.x, self._kdb_cell.dbbox().p2.x],
            [self._kdb_cell.dbbox().p1.y, self._kdb_cell.dbbox().p2.y],
        ]
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.round(bbox, 3)

    # @property
    # def ports(self):
    #     """Returns ports dict."""
    #     return {port.name: port_to_kport(port) for port in self.ports.values()}
    # return self.ports.get_all(self)

    @property
    def ports_layer(self) -> Dict[str, str]:
        """Returns a mapping from layer0_layer1_E0: portName."""
        return map_ports_layer_to_orientation(self.ports)

    def port_by_orientation_cw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise."""
        m = map_ports_to_orientation_cw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def port_by_orientation_ccw(self, key: str, **kwargs):
        """Returns port by indexing them clockwise."""
        m = map_ports_to_orientation_ccw(self.ports, **kwargs)
        if key not in m:
            raise KeyError(f"{key} not in {list(m.keys())}")
        key2 = m[key]
        return self.ports[key2]

    def get_ports_xsize(self, **kwargs) -> float:
        """Returns xdistance from east to west ports.

        Keyword Args:
            layer: port GDS layer.
            prefix: with in port name.
            orientation: in degrees.
            width: port width.
            layers_excluded: List of layers to exclude.
            port_type: optical, electrical, ...
        """
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return snap_to_grid(ports_ccw[0].x - ports_cw[0].x)

    def get_ports_ysize(self, **kwargs) -> float:
        """Returns ydistance from east to west ports.

        Keyword Args:
            layer: port GDS layer.
            prefix: with in port name.
            orientation: in degrees.
            width: port width (um).
            layers_excluded: List of layers to exclude.
            port_type: optical, electrical, ...
        """
        ports_cw = self.get_ports_list(clockwise=True, **kwargs)
        ports_ccw = self.get_ports_list(clockwise=False, **kwargs)
        return snap_to_grid(ports_ccw[0].y - ports_cw[0].y)

    def plot_netlist(
        self, with_labels: bool = True, font_weight: str = "normal", **kwargs
    ):
        """Plots a netlist graph with networkx.

        Args:
            with_labels: add label to each node.
            font_weight: normal, bold.
            **kwargs: keyword arguments for the get_netlist function
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure()
        netlist = self.get_netlist(**kwargs)
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

    def plot_netlist_flat(
        self, with_labels: bool = True, font_weight: str = "normal", **kwargs
    ):
        """Plots a netlist graph with networkx.

        Args:
            flat: if true, will plot the flat netlist
            with_labels: add label to each node.
            font_weight: normal, bold.
            **kwargs: keyword arguments for the get_netlist function
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure()
        netlist = self.get_netlist_flat(**kwargs)
        connections = netlist["connections"]
        placements = netlist["placements"]
        connections_list = []
        for k, v_list in connections.items():
            connections_list.extend(
                (",".join(k.split(",")[:-1]), ",".join(v.split(",")[:-1]))
                for v in v_list
            )
        G = nx.Graph()
        G.add_edges_from(connections_list)
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

    def get_netlist_yaml(self, **kwargs) -> Dict[str, Any]:
        from gdsfactory.get_netlist import get_netlist_yaml

        return get_netlist_yaml(self, **kwargs)

    def write_netlist(self, filepath: str) -> None:
        """Write netlist in YAML."""
        netlist = self.get_netlist()
        OmegaConf.save(netlist, filepath)

    def write_netlist_dot(self, filepath: Optional[str] = None) -> None:
        """Write netlist graph in DOT format."""
        from networkx.drawing.nx_agraph import write_dot

        filepath = filepath or f"{self.name}.dot"

        G = self.plot_netlist()
        write_dot(G, filepath)

    def get_netlist(self, **kwargs) -> Dict[str, Any]:
        """From Component returns instances, connections and placements dict.

        Keyword Args:
            component: to extract netlist.
            full_settings: True returns all, false changed settings.
            tolerance: tolerance in nm to consider two ports connected.
            exclude_port_types: optional list of port types to exclude from netlisting.
            get_instance_name: function to get instance name.
            allow_multiple: False to raise an error if more than two ports share the same connection.
                if True, will return key: [value] pairs with [value] a list of all connected instances.

        Returns:
            Netlist dict (instances, connections, placements, ports)
                instances: Dict of instance name and settings.
                connections: Dict of Instance1Name,portName: Instance2Name,portName.
                placements: Dict of instance names and placements (x, y, rotation).
                ports: Dict portName: ComponentName,port.
                name: name of component.
        """
        from gdsfactory.get_netlist import get_netlist

        return get_netlist(component=self, **kwargs)

    def get_netlist_recursive(self, **kwargs) -> Dict[str, DictConfig]:
        """Returns recursive netlist for a component and subcomponents.

        Keyword Args:
            component: to extract netlist.
            component_suffix: suffix to append to each component name.
                useful if to save and reload a back-annotated netlist.
            get_netlist_func: function to extract individual netlists.
            full_settings: True returns all, false changed settings.
            tolerance: tolerance in nm to consider two ports connected.
            exclude_port_types: optional list of port types to exclude from netlisting.
            get_instance_name: function to get instance name.
            allow_multiple: False to raise an error if more than two ports share the same connection.
                if True, will return key: [value] pairs with [value] a list of all connected instances.

        Returns:
            Dictionary of netlists, keyed by the name of each component.
        """
        from gdsfactory.get_netlist import get_netlist_recursive

        return get_netlist_recursive(component=self, **kwargs)

    def get_netlist_flat(self, **kwargs) -> Dict[str, DictConfig]:
        """Returns a netlist where all subinstances are exposed and independently named.

        Keyword Args:
            component: to extract netlist.
            component_suffix: suffix to append to each component name.
                useful if to save and reload a back-annotated netlist.
            get_netlist_func: function to extract individual netlists.
            full_settings: True returns all, false changed settings.
            tolerance: tolerance in nm to consider two ports connected.
            exclude_port_types: optional list of port types to exclude from netlisting.
            get_instance_name: function to get instance name.
            allow_multiple: False to raise an error if more than two ports share the same connection.
                if True, will return key: [value] pairs with [value] a list of all connected instances.

        Returns:
            Dictionary of netlists, keyed by the name of each component.
        """
        from gdsfactory.get_netlist_flat import get_netlist_flat

        return get_netlist_flat(component=self, **kwargs)

    def assert_ports_on_grid(self, nm: int = 1) -> None:
        """Asserts that all ports are on grid."""
        for port in self.ports.copy()._ports:
            port.assert_on_grid(nm=nm)

    def get_ports(self, depth=None):
        """Returns copies of all the ports of the Component, rotated and \
                translated so that they're in their top-level position.

        The Ports returned are copies of the originals, but each copy has the same
        ``uid`` as the original so that they can be traced back to the original if needed.

        Args:
            depth : int or None
                If not None, defines from how many reference levels to
                retrieve Ports from.

        Returns:
            port_list : list of Port List of all Ports in the Component.
        """
        port_list = [p._copy() for p in self.ports.copy()._ports]

        if depth is None or depth > 0:
            for r in self.references:
                new_depth = None if depth is None else depth - 1
                ref_ports = r.parent.get_ports(depth=new_depth)

                # Transform ports that came from a reference
                ref_ports_transformed = []
                for rp in ref_ports:
                    new_port = rp._copy()
                    new_center, new_orientation = r._transform_port(
                        rp.center,
                        rp.orientation,
                        r.origin,
                        r.rotation,
                        r.x_reflection,
                    )
                    new_port.center = new_center
                    new_port.new_orientation = new_orientation
                    ref_ports_transformed.append(new_port)
                port_list += ref_ports_transformed

        return port_list

    def get_ports_dict(self, **kwargs) -> Dict[str, Port]:
        """Returns a dict of ports.

        Keyword Args:
            layer: port GDS layer.
            prefix: for example "E" for east, "W" for west ...
        """
        return select_ports(self.ports, **kwargs)

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Returns list of ports.

        Keyword Args:
            layer: select ports with GDS layer.
            prefix: select ports with port name.
            orientation: select ports with orientation in degrees.
            width: select ports with port width.
            layers_excluded: List of layers to exclude.
            port_type: select ports with port_type (optical, electrical, vertical_te).
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return list(select_ports(self.ports, **kwargs).values())

    def ref(
        self,
        position: Coordinate = (0, 0),
        port_id: Optional[str] = None,
        rotation: float = 0,
        h_mirror: bool = False,
        v_mirror: bool = False,
    ) -> ComponentReference:
        """Returns Component reference.

        Args:
            position: x, y position.
            port_id: name of the port.
            rotation: in degrees.
            h_mirror: horizontal mirror using y axis (x, 1) (1, 0).
                This is the most common mirror.
            v_mirror: vertical mirror using x axis (1, y) (0, y).
        """
        c = Component()

        _ref = c << self
        _ref = Instance(self.kcl, _ref)

        if port_id:
            if port_id not in self.ports.get_all_named():
                raise ValueError(
                    f"port {port_id} not in {self.ports.get_all_named().keys()}"
                )
            else:
                port = self.ports[port_id]

        origin = (port.d.x, port.d.y) if port_id else (0, 0)
        if h_mirror:
            _ref.trans.M90

        if v_mirror:
            _ref.mirror()

        if rotation != 0:
            _ref.rotate(rotation, origin)
        _ref.move(origin, position)

        return _ref

    def ref_center(self, position=(0, 0)):
        """Returns a reference of the component centered at (x=0, y=0)."""
        si = self.size_info
        yc = si.south + si.height / 2
        xc = si.west + si.width / 2
        center = (xc, yc)
        _ref = ComponentReference(self)
        _ref.move(center, position)
        return _ref

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        refs = [
            f"parent: {inst.cell.name}, ports; {inst.ports}, transformation: {inst.trans}"
            for inst in self.insts
        ]
        return f"{self.name}: uid {self.cell_index()}, ports {list(self.ports)}, references {refs}, {len(self.get_polygons(recursive=False))} polygons"
        # return (
        #     f"{self.name}: uid {self.uid}, "
        #     f"ports {list(self.ports.get_all_named.keys())}, "
        #     f"references {list(self.named_references.keys())}, "
        #     f"{len(self.polygons)} polygons"
        # )

    def pprint(self) -> None:
        """Prints component info."""
        try:
            from rich import pretty

            pretty.install()
            pretty.pprint(self.to_dict())
        except ImportError:
            print(yaml.dump(self.to_dict()))

    def pprint_ports(self) -> None:
        """Prints component netlists."""
        ports_list = self.get_ports_list()
        for port in ports_list:
            print(port)

    @property
    def metadata_child(self) -> DictConfig:
        """Returns metadata from child if any, Otherwise returns component own.

        metadata Great to access the children metadata at the bottom of the
        hierarchy.
        """
        settings = dict(self.settings)

        while settings.get("child"):
            settings = settings.get("child")

        return DictConfig(dict(settings))

    @property
    def metadata(self) -> DictConfig:
        return DictConfig(dict(self.settings))

    def add_port(
        self,
        name: Optional[Union[str, object]] = None,
        center: Optional[Tuple[float, float]] = None,
        width: Optional[float] = None,
        orientation: Optional[float] = None,
        port: Optional[Port] = None,
        layer: LayerSpec = None,
        port_type: str = "optical",
        cross_section: Optional[CrossSection] = None,
    ) -> kf.Port:
        """Add port to component.

        You can copy an existing port like add_port(port = existing_port) or
        create a new port add_port(myname, mycenter, mywidth, myorientation).
        You can also copy an existing port
        with a new name add_port(port = existing_port, name = new_name)

        Args:
            name: port name.
            center: x, y.
            width: in um.
            orientation: in deg.
            port: optional port.
            layer: port layer.
            port_type: optical, electrical, vertical_dc, vertical_te, vertical_tm.
            cross_section: port cross_section.
        """
        from gdsfactory.pdk import get_layer, get_cross_section

        if port:
            name = name if name is not None else port.name

            if isinstance(port, kf.Port):
                kf.KCell.add_port(self, port=port, name=name)
                return port

            elif isinstance(port, Port):
                port.orientation = float(port.orientation)
                print(port.orientation, name)
                p = kf.Port(
                    name=name,
                    position=port.center,
                    width=port.width,
                    angle=int(port.orientation // 90),
                    layer=self.layer(*get_layer(port.layer)),
                    port_type=port.port_type,
                )
                kf.KCell.add_port(self, p)
                return p

            else:
                raise ValueError(
                    f"add_port() needs a Port or kf.Port, got {type(port)}"
                )

        if layer is None:
            if cross_section is None:
                raise ValueError(
                    f"You need to define layer or cross_section. Got {layer!r} and {cross_section}"
                )
            else:
                xs = get_cross_section(cross_section)
                layer = xs.layer

        layer = get_layer(layer)

        p = kf.Port(
            name=name,
            position=center,
            width=width,
            angle=int(orientation // 90),
            layer=self.layer(*layer),
            port_type=port_type,
        )
        kf.KCell.add_port(self, p)
        return p

    def add_ports(
        self,
        ports: Union[List[Port], Dict[str, Port], kf.Ports, kf.kcell.InstancePorts],
        prefix: str = "",
    ) -> None:
        """Add a list or dict of ports.

        you can include a prefix to add to the new port names to avoid name conflicts.

        Args:
            ports: list or dict of ports.
            prefix: to prepend to each port name.
        """
        if isinstance(ports, (kf.kcell.InstancePorts, kf.Ports)):
            ports = ports.copy()._ports

        for port in list(ports):
            name = f"{prefix}{port.name}" if prefix else port.name
            self.add_port(name=name, port=port)

    def snap_ports_to_grid(self, nm: int = 1) -> None:
        for port in self.ports.copy()._ports:
            port.snap_to_grid(nm=nm)

    def remove_layers(
        self,
        layers: List[LayerSpec],
        recursive: bool = True,
    ) -> Component:
        """Remove a list of layers and returns the same Component.

        Args:
            layers: list of layers to remove.
            recursive: operate on the cells included in this cell.
        """
        from gdsfactory import get_layer

        for layer in layers:
            layer = get_layer(layer)
            self.shapes(self.kcl.layer(layer[0], layer[1])).clear()

        if recursive:
            for cell in self.child_cells():
                cell.remove_layers(layers, recursive=False)

        return self

    @property
    def xmin(self):
        return self.bbox[0][0]

    @property
    def xmax(self):
        return self.bbox[0][1]

    @property
    def ymin(self):
        return self.bbox[1][0]

    @property
    def ymax(self):
        return self.bbox[1][1]

    def extract(
        self,
        layers: List[Union[Tuple[int, int], str]],
    ) -> Component:
        """Extract polygons from a Component and returns a new Component."""
        from gdsfactory.pdk import get_layer

        if type(layers) not in (list, tuple):
            raise ValueError(f"layers {layers!r} needs to be a list or tuple")

        layers = [get_layer(layer) for layer in layers]
        # component = self.copy()
        # component._cell.filter(spec=layers, remove=False)

        component = Component()
        poly_dict = self.get_polygons(by_spec=True, include_paths=False)

        for layer in layers:
            if layer in poly_dict:
                polygons = poly_dict[layer]
                for polygon in polygons:
                    component.add_polygon(polygon, layer)

        for layer in layers:
            for path in self._cell.get_paths(layer=layer):
                component.add(path)

        return component

    def add_polygon(self, points: Union[np.ndarray, kdb.Polygon], layer: LayerSpec):
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        if not isinstance(points, kdb.DPolygon):
            points = kdb.DPolygon([kdb.DPoint(point[0], point[1]) for point in points])

        self.shapes(self.kcl.layer(layer[0], layer[1])).insert(points)

    def _add_polygons(self, *polygons: List[Polygon]) -> None:
        self.is_unlocked()
        self._cell.add(*polygons)

    def copy_child_info(self, component: Component) -> None:
        """Copy and settings info from child component into parent.

        Parent components can access child cells settings.
        """
        if not isinstance(component, (Component, ComponentReference)):
            raise ValueError(
                f"{type(component)}" "is not a Component or ComponentReference"
            )

        self.get_child_name = True
        self.child = component
        self.info.update(component.info)
        self.settings.update(component.settings)

    @property
    def size_info(self) -> SizeInfo:
        """Size info of the component."""
        return SizeInfo(self.bbox)

    def get_setting(self, setting: str) -> Union[str, int, float]:
        return (
            self.info.get(setting)
            or self.settings.full.get(setting)
            or self.metadata_child.get(setting)
        )

    def is_unlocked(self) -> None:
        """Raises error if Component is locked."""
        if self._locked:
            raise MutabilityError(
                f"Component {self.name!r} cannot be modified as it's already on cache. "
                + mutability_error_message
            )

    def _add(self, element) -> None:
        """Add a new element or list of elements to this Component.

        Args:
            element: Polygon, ComponentReference or iterable
                The element or iterable of elements to be inserted in this cell.

        Raises:
            MutabilityError: if component is locked.
        """
        self.is_unlocked()
        if isinstance(element, (ComponentReference, kdb.Instance)):
            self.insts.append(element)
        else:
            self << element

    def add(self, element) -> None:
        """Add a new element or list of elements to this Component.

        Args:
            element: Polygon, ComponentReference or iterable
                The element or iterable of elements to be inserted in this cell.

        Raises:
            MutabilityError: if component is locked.
        """
        if isinstance(element, ComponentReference):
            self._add(element)
        elif isinstance(element, Iterable):
            for subelement in element:
                self.add(subelement)
        else:
            self._add(element)

    def add_array(
        self,
        component: Component,
        columns: int = 2,
        rows: int = 2,
        spacing: Tuple[float, float] = (100, 100),
        alias: Optional[str] = None,
    ) -> ComponentReference:
        """Creates a ComponentReference reference to a Component.

        Args:
            component: The referenced component.
            columns: Number of columns in the array.
            rows: Number of rows in the array.
            spacing: array-like[2] of int or float.
                Distance between adjacent columns and adjacent rows.
            alias: str or None. Alias of the referenced Component.

        Returns
            a: ComponentReference containing references to the Component.
        """
        if not isinstance(component, Component):
            raise TypeError("add_array() needs a Component object.")
        return Instance(
            self.kcl,
            self.insert(
                kdb.DCellInstArray(
                    component.cell_index(),
                    # kdb.DTrans.R0,
                    kdb.DVector(),
                    kdb.DVector(spacing[0], 0),
                    kdb.DVector(0, spacing[1]),
                    columns,
                    rows,
                )
            ),
        )

    def distribute(
        self, elements="all", direction="x", spacing=100, separation=True, edge="center"
    ):
        """Distributes the specified elements in the Component.

        Args:
            elements : array-like of objects or 'all'
                Elements to distribute.
            direction : {'x', 'y'}
                Direction of distribution; either a line in the x-direction or
                y-direction.
            spacing : int or float
                Distance between elements.
            separation : bool
                If True, guarantees elements are separated with a fixed spacing
                between; if  False, elements are spaced evenly along a grid.
            edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
                Which edge to perform the distribution along (unused if
                separation == True)

        """
        if elements == "all":
            elements = self.polygons + self.references
        _distribute(
            elements=elements,
            direction=direction,
            spacing=spacing,
            separation=separation,
            edge=edge,
        )
        return self

    def align(self, elements="all", alignment="ymax"):
        """Align elements in the Component.

        Args:
            elements : array-like of objects, or 'all'
                Elements in the Component to align.
            alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
                Which edge to align along (e.g. 'ymax' will move the elements such
                that all of their topmost points are aligned).
        """
        if elements == "all":
            elements = self.polygons + self.references
        _align(elements, alignment=alignment)
        return self

    def flatten(self, single_layer: Optional[LayerSpec] = None):
        """Returns a flattened copy of the component.

        Flattens the hierarchy of the Component such that there are no longer
        any references to other Components. All polygons and labels from
        underlying references are copied and placed in the top-level Component.
        If single_layer is specified, all polygons are moved to that layer.

        Args:
            single_layer: move all polygons are moved to the specified (optional).
        """
        component_flat = Component()

        _cell = self._cell.copy(name=component_flat.name)
        _cell = _cell.flatten()
        component_flat._cell = _cell
        if single_layer is not None:
            from gdsfactory import get_layer

            layer, datatype = get_layer(single_layer)
            for polygon in _cell.polygons:
                polygon.layer = layer
                polygon.datatype = datatype
            for path in _cell.paths:
                path.set_layers(layer)
                path.set_datatypes(datatype)

        component_flat.info = self.info.copy()
        component_flat.add_ports(self.ports)
        return component_flat

    def flatten_reference(self, ref: ComponentReference) -> None:
        """From existing cell replaces reference with a flatten reference \
        which has the transformations already applied.

        Transformed reference keeps the original name.

        Args:
            ref: the reference to flatten into a new cell.

        """
        from gdsfactory.functions import transformed

        self.remove(ref)
        new_component = transformed(ref, decorator=None)
        self.add_ref(new_component, alias=ref.name)

    def add_ref(
        self, component: Component, alias: Optional[str] = None, **kwargs
    ) -> ComponentReference:
        """Add ComponentReference to the current Component.

        Args:
            component: Component.
            alias: named_references.

        Keyword Args:
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
        if not isinstance(component, Component):
            raise TypeError(f"type = {type(Component)} needs to be a Component.")
        # if alias: FIXME
        #     raise NotImplementedError("not yet")

        return self.create_inst(component)

    @property
    def layers(self) -> Set[Tuple[int, int]]:
        """Returns a set of the Layers in the Component."""
        return self.get_layers()

    def get_layers(self) -> Set[Tuple[int, int]]:
        """Return a set of (layer, datatype).

        .. code ::

            import gdsfactory as gf
            gf.components.straight().get_layers() == {(1, 0), (111, 0)}
        """
        return {(info.layer, info.datatype) for info in self.kcl.layer_infos()}

    def _repr_html_(self) -> None:
        """Show geometry in KLayout and in matplotlib for Jupyter Notebooks."""

        self.show(show_ports=True)  # show in klayout
        self.__repr__()
        self.plot_klayout()

    def plot_klayout(self) -> None:
        """Returns ipython widget for klayout visualization.

        Defaults to matplotlib if it fails to import ipywidgets.
        """
        try:
            from gdsfactory.pdk import get_layer_views
            from gdsfactory.widgets.layout_viewer import LayoutViewer
            from IPython.display import display

            gdspath = self.write_gds(logging=False)
            lyp_path = gdspath.with_suffix(".lyp")

            layer_views = get_layer_views()
            layer_views.to_lyp(filepath=lyp_path)
            layout = LayoutViewer(gdspath, lyp_path)
            display(layout.image)
        except ImportError:
            print(
                "You can install `pip install gdsfactory[full]` for better visualization"
            )
            self.plot(plotter="matplotlib")

    def plot_jupyter(self):
        """Shows current gds in klayout. Uses Kweb if server running.

        if not tries using Klayout widget and finally defaults to matplotlib.
        """
        try:
            import os
            from gdsfactory.config import PATH
            from gdsfactory.pdk import get_layer_views
            from IPython.display import IFrame
            import kweb.server_jupyter as kj
            from html import escape

            gdspath = self.write_gds(gdsdir=PATH.gdslib / "extra", logging=False)

            dirpath = GDSDIR_TEMP
            dirpath.mkdir(exist_ok=True, parents=True)
            lyp_path = dirpath / "layers.lyp"

            layer_props = get_layer_views()
            layer_props.to_lyp(filepath=lyp_path)

            port = kj.port if hasattr(kj, "port") else 8000

            src = f"http://127.0.0.1:{port}/gds?gds_file={escape(str(gdspath))}&layer_props={escape(str(lyp_path))}"
            logger.debug(src)

            if kj.jupyter_server and not os.environ.get("DOCS", False):
                return IFrame(
                    src=src,
                    width=1400,
                    height=600,
                )
            else:
                return self.plot_klayout()
        except ImportError:
            print(
                "You can install `pip install gdsfactory[full]` for better visualization"
            )
            return self.plot_klayout()

    def plot_matplotlib(self, **kwargs) -> None:
        """Plot component using matplotlib.

        Keyword Args:
            show_ports: Sets whether ports are drawn.
            show_subports: Sets whether subports (ports that belong to references) are drawn.
            label_aliases: Sets whether aliases are labeled with a text name.
            new_window: If True, each call to quickplot() will generate a separate window.
            blocking: If True, calling quickplot() will pause execution of ("block") the
                remainder of the python code until the quickplot() window is closed.
                If False, the window will be opened and code will continue to run.
            zoom_factor: Sets the scaling factor when zooming the quickplot window with the
                mousewheel/trackpad.
            interactive_zoom: Enables using mousewheel/trackpad to zoom.
            fontsize: for labels.
            layers_excluded: list of layers to exclude.
            layer_views: layer_views colors loaded from Klayout.
            min_aspect: minimum aspect ratio.
        """
        from gdsfactory.quickplotter import quickplot

        quickplot(self, **kwargs)

    def plot(self, plotter: Optional[Plotter] = None, **kwargs) -> None:
        """Returns component plot using klayout, matplotlib, holoviews or qt.

        We recommend using klayout.

        Args:
            plotter: plot backend ('holoviews', 'matplotlib', 'qt', 'klayout').
        """
        plotter = plotter or CONF.get("plotter", "klayout")

        if plotter == "klayout":
            self.plot_klayout()
            return

        elif plotter == "matplotlib":
            from gdsfactory.quickplotter import quickplot

            quickplot(self, **kwargs)
            return

        elif plotter == "holoviews":
            try:
                import holoviews as hv

                hv.extension("bokeh")
            except ImportError as e:
                print("you need to `pip install holoviews`")
                raise e
            return self.plot_holoviews(**kwargs)

        elif plotter == "qt":
            from gdsfactory.quickplotter import quickplot2

            quickplot2(self)
            return
        else:
            raise ValueError(f"{plotter!r} not in {Plotter}")

    def plot_holoviews(
        self,
        layers_excluded: Optional[Layers] = None,
        layer_views: Optional[LayerViews] = None,
        min_aspect: float = 0.25,
        padding: float = 0.5,
    ):
        """Plot component in holoviews.

        Args:
            layers_excluded: list of layers to exclude.
            layer_views: layer_views colors loaded from Klayout.
            min_aspect: minimum aspect ratio.
            padding: around bounding box.

        Returns:
            Holoviews Overlay to display all polygons.
        """
        from gdsfactory.add_pins import get_pin_triangle_polygon_tip
        from gdsfactory.generic_tech import LAYER_VIEWS

        if layer_views is None:
            layer_views = LAYER_VIEWS

        try:
            import holoviews as hv

            hv.extension("bokeh")
        except ImportError as e:
            print("you need to `pip install holoviews`")
            raise e

        self._bb_valid = False  # recompute the bounding box
        b = self.bbox + ((-padding, -padding), (padding, padding))
        b = np.array(b.flat)
        center = np.array((np.sum(b[::2]) / 2, np.sum(b[1::2]) / 2))
        size = np.array((np.abs(b[2] - b[0]), np.abs(b[3] - b[1])))
        dx = np.array(
            (
                np.maximum(min_aspect * size[1], size[0]) / 2,
                np.maximum(size[1], min_aspect * size[0]) / 2,
            )
        )
        b = np.hstack((center - dx, center + dx))

        plots_to_overlay = []
        layers_excluded = [] if layers_excluded is None else layers_excluded

        for layer, polygon in self.get_polygons(by_spec=True).items():
            if layer in layers_excluded:
                continue

            try:
                layer_view = layer_views.get_from_tuple(layer)
            except ValueError:
                layers = list(layer_views.get_layer_views().keys())
                warnings.warn(f"{layer!r} not defined in {layers}", stacklevel=3)
                layer_view = LayerView(layer=layer)
            # TODO: Match up options with LayerViews
            plots_to_overlay.append(
                hv.Polygons(polygon, label=str(layer_view.name)).opts(
                    data_aspect=1,
                    frame_width=500,
                    ylim=(b[1], b[3]),
                    xlim=(b[0], b[2]),
                    fill_color=layer_view.fill_color.as_rgb() or "",
                    line_color=layer_view.frame_color.as_rgb() or "",
                    fill_alpha=layer_view.get_alpha() or "",
                    line_alpha=layer_view.get_alpha() or "",
                    tools=["hover"],
                )
            )
        for name, port in self.ports.items():
            name = str(name)
            polygon, ptip = get_pin_triangle_polygon_tip(port=port)

            plots_to_overlay.append(
                hv.Polygons(polygon, label=name).opts(
                    data_aspect=1,
                    frame_width=500,
                    fill_alpha=0,
                    ylim=(b[1], b[3]),
                    xlim=(b[0], b[2]),
                    color="red",
                    line_alpha=layer_view.get_alpha() or "",
                    tools=["hover"],
                )
                * hv.Text(ptip[0], ptip[1], name)
            )

        return hv.Overlay(plots_to_overlay).opts(
            show_legend=True, shared_axes=False, ylim=(b[1], b[3]), xlim=(b[0], b[2])
        )

    def show(
        self,
        show_ports: bool = False,
        show_subports: bool = False,
        **kwargs,
    ) -> None:
        """Show component in KLayout.

        returns a copy of the Component, so the original component remains intact.
        with pins markers on each port show_ports = True, and optionally also
        the ports from the references (show_subports=True)

        Args:
            show_ports: shows component with port markers and labels.
            port_marker_layer: for the ports.

        Keyword Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/.
            unit: unit size for objects in library. 1um by default.
            precision: for object dimensions in the library (m). 1nm by default.
            timestamp: Defaults to 2019-10-25. If None uses current time.
        """
        from gdsfactory.show import show

        if show_ports:
            self.draw_ports()

        show(self, **kwargs)

    def to_gmsh(
        self,
        type,
        z=None,
        xsection_bounds=None,
        layer_stack=None,
        wafer_padding=0.0,
        wafer_layer=LAYER.WAFER,
        *args,
        **kwargs,
    ):
        """Returns a gmsh msh of the component for finite element simulation.

        Arguments:
            type: one of "xy", "uz", or "3D". Determines the type of mesh to return.
            z: used to define z-slice for xy meshing
            xsection_bounds: used to define in-plane line for uz meshing
            wafer_padding: padding beyond bbox to add to WAFER layers.

        Keyword Args:
            Arguments for the target meshing function in gdsfactory.simulation.gmsh
        """

        padded_component = Component()
        padded_component << self
        (xmin, ymin), (xmax, ymax) = self.bbox
        points = [
            [xmin - wafer_padding, ymin - wafer_padding],
            [xmax + wafer_padding, ymin - wafer_padding],
            [xmax + wafer_padding, ymax + wafer_padding],
            [xmin - wafer_padding, ymax + wafer_padding],
        ]
        padded_component.add_polygon(points, layer=wafer_layer)

        if layer_stack is None:
            raise ValueError(
                'A LayerStack must be provided through argument "layer_stack".'
            )
        if type == "xy":
            if z is None:
                raise ValueError(
                    'For xy-meshing, a z-value must be provided via the float argument "z".'
                )
            from gdsfactory.simulation.gmsh.xy_xsection_mesh import xy_xsection_mesh

            return xy_xsection_mesh(padded_component, z, layer_stack, **kwargs)
        elif type == "uz":
            if xsection_bounds is None:
                raise ValueError(
                    "For uz-meshing, you must provide a line in the xy-plane "
                    "via the Tuple argument [[x1,y1], [x2,y2]] xsection_bounds."
                )
            from gdsfactory.simulation.gmsh.uz_xsection_mesh import uz_xsection_mesh

            return uz_xsection_mesh(
                padded_component, xsection_bounds, layer_stack, **kwargs
            )
        elif type == "3D":
            from gdsfactory.simulation.gmsh.xyz_mesh import xyz_mesh

            return xyz_mesh(padded_component, layer_stack, **kwargs)
        else:
            raise ValueError(
                'Required argument "type" must be one of "xy", "uz", or "3D".'
            )

    def write_gds(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: Optional[PathType] = None,
    ) -> Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
        """
        gdsdir = (
            gdsdir or pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
        )
        gdsdir = pathlib.Path(gdsdir)
        gdspath = gdspath or gdsdir / f"{self.name}.gds"
        gdspath = pathlib.Path(gdspath)
        gdsdir = gdspath.parent
        gdsdir.mkdir(exist_ok=True, parents=True)
        self.write(filename=gdspath)
        return gdspath

    def write_oas(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: Optional[PathType] = None,
        **kwargs,
    ) -> Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            unit: unit size for objects in library. 1um by default.
        """
        gdsdir = (
            gdsdir or pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
        )
        gdsdir = pathlib.Path(gdsdir)
        gdspath = gdspath or gdsdir / f"{self.name}.oas"
        gdspath = pathlib.Path(gdspath)
        gdsdir = gdspath.parent
        gdsdir.mkdir(exist_ok=True, parents=True)
        self.write(filename=gdspath)
        return gdspath

    def write_gds_with_metadata(self, *args, **kwargs) -> Path:
        """Write component in GDS and metadata (component settings) in YAML."""
        gdspath = self.write(*args, **kwargs)
        metadata = gdspath.with_suffix(".yml")
        metadata.write_text(self.to_yaml(with_cells=True, with_ports=True))
        logger.info(f"Write YAML metadata to {str(metadata)!r}")
        return gdspath

    def to_dict(
        self,
        ignore_components_prefix: Optional[List[str]] = None,
        ignore_functions_prefix: Optional[List[str]] = None,
        with_cells: bool = False,
        with_ports: bool = False,
    ) -> Dict[str, Any]:
        """Returns Dict representation of a component.

        Args:
            ignore_components_prefix: for components to ignore when exporting.
            ignore_functions_prefix: for functions to ignore when exporting.
            with_cells: write cells recursively.
            with_ports: write port information dict.
        """
        d = {}
        if with_ports:
            ports = {port.name: port.to_dict() for port in self.get_ports_list()}
            d["ports"] = ports

        if with_cells:
            cells = recurse_structures(
                self,
                ignore_functions_prefix=ignore_functions_prefix,
                ignore_components_prefix=ignore_components_prefix,
            )
            d["cells"] = clean_dict(cells)

        d["name"] = self.name
        d["settings"] = clean_dict(dict(self.settings))
        return d

    def to_yaml(self, **kwargs) -> str:
        """Write Dict representation of a component in YAML format.

        Args:
            ignore_components_prefix: for components to ignore when exporting.
            ignore_functions_prefix: for functions to ignore when exporting.
            with_cells: write cells recursively.
            with_ports: write port information.
        """
        return OmegaConf.to_yaml(self.to_dict(**kwargs))

    def to_dict_polygons(self) -> Dict[str, Any]:
        """Returns a dict representation of the flattened component."""
        d = {}
        polygons = {}
        layer_to_polygons = self.get_polygons(by_spec=True)

        for layer, polygons_layer in layer_to_polygons.items():
            layer_name = f"{layer[0]}_{layer[1]}"
            for polygon in polygons_layer:
                polygons[layer_name] = [tuple(snap_to_grid(v)) for v in polygon]

        ports = {port.name: port.settings for port in self.get_ports_list()}
        clean_dict(ports)
        clean_dict(polygons)
        d.info = self.info
        d.polygons = polygons
        d.ports = ports
        return d

    def auto_rename_ports(self, **kwargs) -> None:
        """Rename ports by orientation NSEW (north, south, east, west).

        Keyword Args:
            function: to rename ports.
            select_ports_optical: to select optical ports.
            select_ports_electrical: to select electrical ports.
            prefix_optical: prefix.
            prefix_electrical: prefix.

        .. code::

                  3  4
                 _|__|_
             2 -|      |- 5
                |      |
             1 -|______|- 6
                  |  |
                  8  7
        """
        self.is_unlocked()
        auto_rename_ports(self, **kwargs)

    def auto_rename_ports_counter_clockwise(self, **kwargs) -> None:
        self.is_unlocked()
        auto_rename_ports_counter_clockwise(self, **kwargs)

    def auto_rename_ports_layer_orientation(self, **kwargs) -> None:
        self.is_unlocked()
        auto_rename_ports_layer_orientation(self, **kwargs)

    def auto_rename_ports_orientation(self, **kwargs) -> None:
        """Rename ports by orientation NSEW (north, south, east, west).

        Keyword Args:
            function: to rename ports.
            select_ports_optical: to select ports.
            select_ports_electrical:
            prefix_optical:
            prefix_electrical:

        .. code::

                 N0  N1
                 |___|_
            W1 -|      |- E1
                |      |
            W0 -|______|- E0
                 |   |
                S0   S1
        """
        self.is_unlocked()
        auto_rename_ports_orientation(self, **kwargs)

    def move(
        self,
        origin: Float2 = (0, 0),
        destination: Optional[Float2] = None,
        axis: Optional[Axis] = None,
    ) -> Component:
        """Returns new Component with a moved reference to the original.

        component.

        Args:
            origin: of component.
            destination: x, y.
            axis: x or y.
        """
        from gdsfactory.functions import move

        return move(component=self, origin=origin, destination=destination, axis=axis)

    def mirror(
        self,
        p1: Float2 = (0, 1),
        p2: Float2 = (0, 0),
    ) -> Component:
        """Returns new Component with a mirrored reference.

        Args:
            p1: first point to define mirror axis.
            p2: second point to define mirror axis.
        """
        from gdsfactory.functions import mirror

        return mirror(component=self, p1=p1, p2=p2)

    def rotate(self, angle: float = 90) -> Component:
        """Returns new component with a rotated reference to the original.

        Args:
            angle: in degrees.
        """
        from gdsfactory.functions import rotate

        return rotate(component=self, angle=angle)

    def add_padding(self, **kwargs) -> Component:
        """Returns same component with padding.

        Keyword Args:
            component: for padding.
            layers: list of layers.
            suffix for name.
            default: default padding (50um).
            top: north padding.
            bottom: south padding.
            right: east padding.
            left: west padding.
        """
        from gdsfactory.add_padding import add_padding

        return add_padding(component=self, **kwargs)

    def absorb(self, reference) -> Component:
        """Absorbs polygons from ComponentReference into Component.

        Destroys the reference in the process but keeping the polygon geometry.

        Args:
            reference: ComponentReference to be absorbed into the Component.
        """
        if reference not in self.references:
            raise ValueError(
                "The reference you asked to absorb does not exist in this Component."
            )
        reference.flatten()
        return self

    def remove(self, items):
        """Removes items from a Component, which can include Ports, PolygonSets \
        CellReferences, ComponentReferences and Labels.

        Args:
            items: list of Items to be removed from the Component.
        """
        if not hasattr(items, "__iter__"):
            items = [items]
        for item in items:
            if isinstance(item, Port):
                self.ports = {k: v for k, v in self.ports.items() if v != item}
            elif isinstance(item, gdstk.Reference):
                self._cell.remove(item)
                item.owner = None
            elif isinstance(item, ComponentReference):
                self.references.remove(item)
                self._cell.remove(item._reference)
                item.owner = None
                self._named_references.pop(item.name)
            else:
                self._cell.remove(item)

        self._bb_valid = False
        return self

    def hash_geometry(self, precision: float = 1e-4) -> str:
        """Returns an SHA1 hash of the geometry in the Component.

        For each layer, each polygon is individually hashed and then the polygon hashes
        are sorted, to ensure the hash stays constant regardless of the ordering
        the polygons.  Similarly, the layers are sorted by (layer, datatype).

        Args:
            precision: Rounding precision for the the objects in the Component.
                For instance, a precision of 1e-2 will round a point at
                (0.124, 1.748) to (0.12, 1.75).

        """
        polygons_by_spec = self.get_polygons(by_spec=True, as_array=False)
        layers = np.array(list(polygons_by_spec.keys()))
        sorted_layers = layers[np.lexsort((layers[:, 0], layers[:, 1]))]

        final_hash = hashlib.sha1()
        for layer in sorted_layers:
            layer_hash = hashlib.sha1(layer.astype(np.int64)).digest()
            polygons = polygons_by_spec[tuple(layer)]
            polygons = [_rnd(p.points, precision) for p in polygons]
            polygon_hashes = np.sort([hashlib.sha1(p).digest() for p in polygons])
            final_hash.update(layer_hash)
            for ph in polygon_hashes:
                final_hash.update(ph)

        return final_hash.hexdigest()

    def get_labels(
        self, apply_repetitions=True, depth: Optional[int] = None, layer=None
    ) -> List[Label]:
        """Return labels.

        Args:
            apply_repetitions:.
            depth: None returns all labels and 0 top level.
            layer: layerspec.
        """
        from gdsfactory.pdk import get_layer

        if layer:
            layer, texttype = get_layer(layer)
        else:
            texttype = None
        return self._cell.get_labels(
            apply_repetitions=apply_repetitions,
            depth=depth,
            layer=layer,
            texttype=texttype,
        )

    def remove_labels(self) -> None:
        """Remove labels."""
        self._cell.remove(*self.labels)

    # Deprecated
    def get_info(self):
        """Gathers the .info dictionaries from every sub-Component and returns them in a list.

        Args:
            depth: int or None
                If not None, defines from how many reference levels to
                retrieve Ports from.

        Returns:
            list of dictionaries
                List of the ".info" property dictionaries from all sub-Components
        """
        D_list = self.get_dependencies(recursive=True)
        return [D.info.copy() for D in D_list]

    def remap_layers(
        self, layermap, include_labels: bool = True, include_paths: bool = True
    ) -> Component:
        """Returns a copy of the component with remapped layers.

        Args:
            layermap: Dictionary of values in format {layer_from : layer_to}.
            include_labels: Selects whether to move Labels along with polygons.
            include_paths: Selects whether to move Paths along with polygons.
        """
        component = self.copy()
        layermap = {_parse_layer(k): _parse_layer(v) for k, v in layermap.items()}

        all_D = list(component.get_dependencies(True))
        all_D.append(component)
        for D in all_D:
            for p in D.polygons:
                layer = (p.layer, p.datatype)
                if layer in layermap:
                    new_layer = layermap[layer]
                    p.layer = new_layer[0]
                    p.datatype = new_layer[1]
            if include_labels:
                for label in D.labels:
                    original_layer = (label.layer, label.texttype)
                    original_layer = _parse_layer(original_layer)
                    if original_layer in layermap:
                        new_layer = layermap[original_layer]
                        label.layer = new_layer[0]
                        label.texttype = new_layer[1]

            if include_paths:
                for path in D.paths:
                    new_layers = list(path.layers)
                    new_datatypes = list(path.datatypes)
                    for layer_number in range(len(new_layers)):
                        original_layer = _parse_layer(
                            (new_layers[layer_number], new_datatypes[layer_number])
                        )
                        if original_layer in layermap:
                            new_layer = layermap[original_layer]
                            new_layers[layer_number] = new_layer[0]
                            new_datatypes[layer_number] = new_layer[1]
                    path.set_layers(*new_layers)
                    path.set_datatypes(*new_datatypes)
        return component

    def to_3d(
        self,
        layer_views: Optional[LayerViews] = None,
        layer_stack: Optional = None,
        exclude_layers: Optional[Tuple[Layer, ...]] = None,
    ):
        """Return Component 3D trimesh Scene.

        Args:
            component: to extrude in 3D.
            layer_views: layer colors from Klayout Layer Properties file.
                Defaults to active PDK.layer_views.
            layer_stack: contains thickness and zmin for each layer.
                Defaults to active PDK.layer_stack.
            exclude_layers: layers to exclude.

        """
        from gdsfactory.export.to_3d import to_3d

        return to_3d(
            self,
            layer_views=layer_views,
            layer_stack=layer_stack,
            exclude_layers=exclude_layers,
        )

    def to_np(
        self,
        nm_per_pixel: int = 20,
        layers: Layers = ((1, 0),),
        values: Optional[Tuple[float, ...]] = None,
        pad_width: int = 1,
    ) -> np.ndarray:
        """Returns a pixelated numpy array from Component polygons.

        Args:
            component: Component.
            nm_per_pixel: you can go from 20 (coarse) to 4 (fine).
            layers: to convert. Order matters (latter overwrite former).
            values: associated to each layer (defaults to 1).
            pad_width: padding pixels around the image.

        """
        from gdsfactory.export.to_np import to_np

        return to_np(
            self,
            nm_per_pixel=nm_per_pixel,
            layers=layers,
            values=values,
            pad_width=pad_width,
        )

    def write_stl(
        self,
        filepath: str,
        layer_stack: Optional = None,
        exclude_layers: Optional[Tuple[Layer, ...]] = None,
    ) -> np.ndarray:
        """Write a Component to STL for 3D printing.

        Args:
            filepath: to write STL to.
            layer_stack: contains thickness and zmin for each layer.
            exclude_layers: layers to exclude.
            use_layer_name: If True, uses LayerLevel names in output filenames rather than gds_layer and gds_datatype.
            hull_invalid_polygons: If True, replaces invalid polygons (determined by shapely.Polygon.is_valid) with its convex hull.
            scale: Optional factor by which to scale meshes before writing.

        """
        from gdsfactory.export.to_stl import to_stl

        return to_stl(
            self,
            filepath=filepath,
            layer_stack=layer_stack,
            exclude_layers=exclude_layers,
        )

    def offset(
        self,
        distance: float = 0.1,
        polygons=None,
        use_union: bool = True,
        precision: float = 1e-4,
        join: str = "miter",
        tolerance: int = 2,
        layer: LayerSpec = "WG",
    ) -> Component:
        """Returns new Component with polygons eroded or dilated by an offset.

        Args:
            distance: Distance to offset polygons. Positive values expand, negative shrink.
            precision: Desired precision for rounding vertex coordinates.
            join: {'miter', 'bevel', 'round'} Type of join used to create polygon offset
            tolerance: For miter joints, this number must be at least 2 represents the
              maximal distance in multiples of offset between new vertices and their
              original position before beveling to avoid spikes at acute joints. For
              round joints, it indicates the curvature resolution in number of
              points per full circle.
            layer: Specific layer for new polygons.

        """
        import gdsfactory as gf

        gds_layer, gds_datatype = gf.get_layer(layer)
        p = gdstk.offset(
            polygons or self.get_polygons(),
            distance=distance,
            join=join,
            tolerance=tolerance,
            precision=precision,
            use_union=use_union,
            layer=gds_layer,
            datatype=gds_datatype,
        )

        component = gf.Component()
        component.add_polygon(p, layer=layer)
        return component


def copy(
    D: Component,
    references=None,
    ports=None,
    polygons=None,
    paths=None,
    name=None,
    labels=None,
) -> Component:
    """Returns a Component copy.

    Args:
        D: component to copy.
    """
    D_copy = Component()
    D_copy.info = D.info
    # D_copy._cell = D._cell.copy(name=D_copy.name)

    for ref in references if references is not None else D.references:
        D_copy.add(copy_reference(ref))
    for port in (ports if ports is not None else D.ports).values():
        D_copy.add_port(port=port)
    for poly in polygons if polygons is not None else D.polygons:
        D_copy.add_polygon(poly)
    for path in paths if paths is not None else D.paths:
        D_copy.add(path)
    for label in labels if labels is not None else D.labels:
        D_copy.add_label(
            text=label.text,
            position=label.origin,
            layer=(label.layer, label.texttype),
        )

    if name is not None:
        D_copy.name = name

    return D_copy


def copy_reference(
    ref,
    parent=None,
    columns=None,
    rows=None,
    spacing=None,
    origin=None,
    rotation=None,
    magnification=None,
    x_reflection=None,
    name=None,
    v1=None,
    v2=None,
) -> ComponentReference:
    return ComponentReference(
        component=parent or ref.parent,
        columns=columns or ref.columns,
        rows=rows or ref.rows,
        spacing=spacing or ref.spacing,
        origin=origin or ref.origin,
        rotation=rotation or ref.rotation,
        magnification=magnification or ref.magnification,
        x_reflection=x_reflection or ref.x_reflection,
        name=name or ref.name,
        v1=v1 or ref.v1,
        v2=v2 or ref.v2,
    )


def test_get_layers() -> Component:
    import gdsfactory as gf

    c1 = gf.components.straight(
        length=10,
        width=0.5,
        layer=(2, 0),
        bbox_layers=[(111, 0)],
        bbox_offsets=[3],
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    assert c1.get_layers() == {(2, 0), (111, 0)}, c1.get_layers()
    # return c1
    c2 = c1.remove_layers([(111, 0)])
    assert c2.get_layers() == {(2, 0)}, c2.get_layers()
    return c2


def _filter_polys(polygons, layers_excl):
    return [
        polygon
        for polygon, layer, datatype in zip(
            polygons.polygons, polygons.layers, polygons.datatypes
        )
        if (layer, datatype) not in layers_excl
    ]


def recurse_structures(
    component: Component,
    ignore_components_prefix: Optional[List[str]] = None,
    ignore_functions_prefix: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Recurse component and components references recursively.

    Args:
        component: component to recurse.
        ignore_components_prefix: list of prefix to ignore.
        ignore_functions_prefix: list of prefix to ignore.
    """
    ignore_functions_prefix = ignore_functions_prefix or []
    ignore_components_prefix = ignore_components_prefix or []

    if (
        hasattr(component, "function_name")
        and component.function_name in ignore_functions_prefix
    ):
        return {}

    if hasattr(component, "name") and any(
        component.name.startswith(i) for i in ignore_components_prefix
    ):
        return {}

    output = {component.name: dict(component.settings)}
    for reference in component.references:
        if (
            isinstance(reference, ComponentReference)
            and reference.ref_cell.name not in output
        ):
            output.update(recurse_structures(reference.ref_cell))

    return output


def flatten_invalid_refs_recursive(
    component: Component, grid_size: Optional[float] = None
) -> Component:
    """Returns new Component with flattened references.

    Args:
        component: to flatten invalid references.
        grid_size: optional grid size in um.
    """
    from gdsfactory.decorators import is_invalid_ref
    from gdsfactory.functions import transformed
    import networkx as nx

    def _create_dag(component):
        """DAG where components point to references which then point to components again."""
        nodes = {}
        edges = {}

        def _add_nodes_recursive(g, component):
            g.add_node(component.name)
            nodes[component.name] = component
            for ref in component.references:
                edge_name = f"{component.name}:{ref.name}"
                g.add_edge(component.name, edge_name)
                g.add_edge(edge_name, ref.parent.name)
                edges[edge_name] = ref
                _add_nodes_recursive(g, ref.parent)

        g = nx.DiGraph()
        _add_nodes_recursive(g, component)

        return g, nodes, edges

    def _find_leaves(g):
        leaves = [n for n, d in g.out_degree() if d == 0]
        return leaves

    def _prune_leaves(g):
        """Prune components AND references pointing to them at the bottom of the DAG.
        Helper function
        """
        comps = _find_leaves(g)
        for component in comps:
            g.remove_node(component)
        refs = _find_leaves(g)
        for r in refs:
            g.remove_node(r)
        return g, comps, refs

    finished_comps = {}
    g, comps, refs = _create_dag(component)
    while True:
        g, comp_leaves, ref_leaves = _prune_leaves(g)
        if not comp_leaves:
            break
        new_comps = {}
        for ref_name in ref_leaves:
            r = refs[ref_name]
            comp_name, _ = ref_name.split(":")
            if comp_name in finished_comps:
                continue
            new_comps[comp_name] = comps[comp_name] = new_comps.get(
                comp_name
            ) or Component(name=comp_name)
            if is_invalid_ref(r, grid_size):
                comp = transformed(r, cache=False, decorator=None)  # type: ignore
                comps[comp.name] = comp
                r = refs[ref_name] = ComponentReference(comp)
            comps[comp_name].add(
                copy_reference(refs[ref_name], parent=comps[r.parent.name])
            )
        finished_comps.update(new_comps)
    return finished_comps[component.name]


def test_same_uid() -> None:
    import gdsfactory as gf

    c = Component()
    c << gf.components.rectangle()
    c << gf.components.rectangle()

    r1 = c.references[0].parent
    r2 = c.references[1].parent

    assert r1.uid == r2.uid, f"{r1.uid} must equal {r2.uid}"


def test_netlist_simple() -> None:
    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=2)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect(port="o1", destination=c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 2


def test_netlist_simple_width_mismatch_throws_error() -> None:
    import pytest

    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.components.straight(length=1, width=1)
    c2 = c << gf.components.straight(length=2, width=2)
    c2.connect(port="o1", destination=c1.ports["o2"])
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    with pytest.raises(ValueError):
        c.get_netlist()


def test_netlist_complex() -> None:
    import gdsfactory as gf

    c = gf.components.mzi_arms()
    netlist = c.get_netlist()
    # print(netlist.pretty())
    assert len(netlist["instances"]) == 4, len(netlist["instances"])


def test_extract() -> Component:
    import gdsfactory as gf

    c = gf.components.straight(
        length=10,
        width=0.5,
        bbox_layers=[gf.LAYER.WGCLAD],
        bbox_offsets=[3],
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    c2 = c.extract(layers=[gf.LAYER.WGCLAD])

    assert len(c.polygons) == 2, len(c.polygons)
    assert len(c2.polygons) == 1, len(c2.polygons)
    assert gf.LAYER.WGCLAD in c2.layers
    return c2


def hash_file(filepath):
    md5 = hashlib.md5()
    md5.update(filepath.read_bytes())
    return md5.hexdigest()


def test_bbox_reference() -> Component:
    import gdsfactory as gf

    c = gf.Component("component_with_offgrid_polygons")
    c1 = c << gf.components.rectangle(size=(1.5e-3, 1.5e-3), port_type=None)
    c2 = c << gf.components.rectangle(size=(1.5e-3, 1.5e-3), port_type=None)
    c2.xmin = c1.xmax

    assert c2.xsize == 2e-3
    return c2


def test_bbox_component() -> None:
    import gdsfactory as gf

    c = gf.components.rectangle(size=(1.5e-3, 1.5e-3), port_type=None)
    assert c.xsize == 2e-3


def test_remap_layers() -> None:
    import gdsfactory as gf

    c = gf.components.straight(layer=(2, 0))
    remap = c.remap_layers(layermap={(2, 0): gf.LAYER.WGN})
    hash_geometry = "83fbc6a8289505eaed3a2e3ab279cc03f5e4d00c"

    assert (
        remap.hash_geometry() == hash_geometry
    ), f"hash_geometry = {remap.hash_geometry()!r}"


def test_remove_labels() -> None:
    import gdsfactory as gf

    c = gf.c.straight()
    c.remove_labels()

    assert len(c.labels) == 0


def test_import_gds_settings() -> None:
    import gdsfactory as gf

    c = gf.components.mzi()
    gdspath = c.write_gds_with_metadata()
    c2 = gf.import_gds(gdspath, name="mzi_sample", read_metadata=True)
    c3 = gf.routing.add_fiber_single(c2)
    assert c3


def test_flatten_invalid_refs_recursive() -> None:
    import gdsfactory as gf

    @gf.cell
    def flat():
        c = gf.Component()
        mmi1 = (c << gf.components.mmi1x2()).move((0, 1e-4))
        mmi2 = (c << gf.components.mmi1x2()).rotate(90)
        mmi2.move((40, 20))
        route = gf.routing.get_route(mmi1.ports["o2"], mmi2.ports["o1"], radius=5)
        c.add(route.references)
        return c

    @gf.cell
    def hierarchy():
        c = gf.Component()
        (c << flat()).rotate(33)
        (c << flat()).move((100, 0))
        return c

    c_orig = hierarchy()
    c_new = flatten_invalid_refs_recursive(c_orig)
    assert c_new is not c_orig
    assert c_new != c_orig
    assert c_orig.references[0].parent.name != c_new.references[0].parent.name
    assert (
        c_orig.references[1].parent.references[0].parent.name
        != c_new.references[1].parent.references[0].parent.name
    )


if __name__ == "__main__":
    c = Component("parent")
    c2 = Component("child")
    length = 10
    width = 0.5
    layer = (1, 0)
    c2.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    c.add_port(name="o1", center=(0, 0), width=0.5, orientation=180, layer=(1, 0))
    c.add_port(name="o2", center=(length, 0), width=0.5, orientation=180, layer=(1, 0))

    c << c2

    # print(c2.get_polygons())
    # print(c2.get_polygons((1, 0)))
    # print(c2.get_polygons(recursive=False))
    # print(c2.get_polygons((1, 0), recursive=False))
    # print(c2)
    # c2.show(show_ports=True)

    # kf.show('a.gds')
    # c2.write('a.gds')
    # layer = (2, 0)
    # width = 1
    # c2.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)

    # ref = c << c2
    # ref.y = 10
    # c2.show()

    # c = gf.c.mzi()
    # c = gf.c.bend_circular()
    # c = gf.c.mzi()
    c.show()
    # import gdsfactory as gf
    # c2 = gf.Component()
    # r = c.ref()
    # c2.copy_child_info(c.named_references["sxt"])
    # test_remap_layers()
    # c = test_get_layers()
    # c.plot_qt()
    # c.ploth()
    # c = test_extract()
    # c.show()
