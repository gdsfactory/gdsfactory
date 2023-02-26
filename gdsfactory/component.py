"""Component is a canvas for geometry.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import datetime
import hashlib
import itertools
import math
import pathlib
import tempfile
import uuid
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gdstk
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Literal

from gdsfactory.component_layout import (
    Label,
    Polygon,
    _align,
    _distribute,
    _GeometryHelper,
    _parse_layer,
    get_polygons,
)
from gdsfactory.component_reference import ComponentReference, Coordinate, SizeInfo
from gdsfactory.config import CONF, logger
from gdsfactory.cross_section import CrossSection
from gdsfactory.port import (
    Port,
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
from gdsfactory.technology import LayerView, LayerViews, LayerStack
from gdsfactory.generic_tech import LAYER

Plotter = Literal["holoviews", "matplotlib", "qt", "klayout"]
Axis = Literal["x", "y"]


class MutabilityError(ValueError):
    pass


def _get_dependencies(component, references_set):
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

tmp = pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
tmp.mkdir(exist_ok=True, parents=True)
_timestamp2019 = datetime.datetime.fromtimestamp(1572014192.8273)
MAX_NAME_LENGTH = 32


def _rnd(arr, precision=1e-4):
    arr = np.ascontiguousarray(arr)
    ndigits = round(-math.log10(precision))
    return np.ascontiguousarray(arr.round(ndigits) / precision, dtype=np.int64)


class Component(_GeometryHelper):
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

    def __init__(
        self,
        name: str = "Unnamed",
        with_uuid: bool = False,
    ) -> None:
        """Initialize the Component object."""
        self.uid = str(uuid.uuid4())[:8]
        if with_uuid or name == "Unnamed":
            name += f"_{self.uid}"

        self._cell = gdstk.Cell(name=name)
        self.name = name
        self.info: Dict[str, Any] = {}

        self.settings: Dict[str, Any] = {}
        self._locked = False
        self.get_child_name = False
        self._reference_names_counter = Counter()
        self._reference_names_used = set()
        self._named_references = {}
        self._references = []

        self.ports = {}

    @property
    def references(self):
        return self._references

    @property
    def polygons(self) -> List[Polygon]:
        return self._cell.polygons

    @property
    def area(self) -> float:
        return self._cell.area

    @property
    def labels(self) -> List[Label]:
        return self._cell.labels

    @property
    def paths(self):
        return self._cell.paths

    @property
    def name(self) -> str:
        return self._cell.name

    @name.setter
    def name(self, value):
        self._cell.name = value

    def __iter__(self):
        """You can iterate over polygons, paths, labels and references."""
        return itertools.chain(self.polygons, self.paths, self.labels, self.references)

    def get_polygons(
        self,
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
        return get_polygons(
            instance=self,
            by_spec=by_spec,
            depth=depth,
            include_paths=include_paths,
            as_array=as_array,
        )

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
            ports = list(self.ports.keys())
            raise ValueError(f"{key!r} not in {ports}")

        return self.ports[key]

    def __lshift__(self, element):
        """Convenience operator equivalent to add_ref()."""
        return self.add_ref(element)

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
        return self._named_references

    def add_label(
        self,
        text: str = "hello",
        position: Tuple[float, float] = (0.0, 0.0),
        magnification: float = 1.0,
        rotation: float = 0,
        anchor: str = "o",
        layer="TEXT",
        x_reflection=False,
    ) -> Label:
        """Adds Label to the Component.

        Args:
            text: Label text.
            position: x-, y-coordinates of the Label location.
            magnification: Magnification factor for the Label text.
            rotation: Angle rotation of the Label text.
            anchor: {'n', 'e', 's', 'w', 'o', 'ne', 'nw', ...}
                Position of the anchor relative to the text.
            layer: Specific layer(s) to put Label on.
            x_reflection: True reflects across the horizontal axis before rotation.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        gds_layer, gds_datatype = layer

        if not isinstance(text, str):
            text = text
        label = Label(
            text=text,
            origin=position,
            anchor=anchor,
            magnification=magnification,
            rotation=rotation,
            layer=gds_layer,
            texttype=gds_datatype,
            x_reflection=x_reflection,
        )
        self.add(label)
        return label

    @property
    def bbox(self):
        """Returns the bounding box of the ComponentReference.

        it snaps to 3 decimals in um (0.001um = 1nm precision)
        """
        bbox = self._cell.bounding_box()
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.round(bbox, 3)

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
        for port in self.ports.values():
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
        port_list = [p._copy() for p in self.ports.values()]

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
            prefix: select ports with prefix in port name.
            suffix: select ports with port name suffix.
            orientation: select ports with orientation in degrees.
            width: select ports with port width.
            layers_excluded: List of layers to exclude.
            port_type: select ports with port_type (optical, electrical, vertical_te).
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return select_ports(self.ports, **kwargs)

    def get_ports_list(self, **kwargs) -> List[Port]:
        """Returns list of ports.

        Keyword Args:
            layer: select ports with GDS layer.
            prefix: select ports with prefix in port name.
            suffix: select ports with port name suffix.
            orientation: select ports with orientation in degrees.
            orientation: select ports with orientation in degrees.
            width: select ports with port width.
            layers_excluded: List of layers to exclude.
            port_type: select ports with port_type (optical, electrical, vertical_te).
            clockwise: if True, sort ports clockwise, False: counter-clockwise.
        """
        return list(select_ports(self.ports, **kwargs).values())

    def get_ports_pandas(self):
        import pandas as pd

        col_spec = [
            "name",
            "width",
            "center",
            "orientation",
            "layer",
            "port_type",
            "shear_angle",
        ]

        return pd.DataFrame(
            [port.to_dict() for port in self.get_ports_list()], columns=col_spec
        )

    def get_ports_polars(self):
        import polars as pl

        col_spec = {
            "name": pl.Utf8,
            "width": pl.Float64,
            "center": pl.List(pl.Float64),
            "orientation": pl.Float64,
            "layer": pl.List(pl.UInt16),
            "port_type": pl.Utf8,
            "shear_angle": pl.Float64,
        }

        return pl.DataFrame(
            [port.to_dict() for port in self.get_ports_list()], schema=col_spec
        )

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
        _ref = ComponentReference(self)

        if port_id and port_id not in self.ports:
            raise ValueError(f"port {port_id} not in {self.ports.keys()}")

        origin = self.ports[port_id].center if port_id else (0, 0)
        if h_mirror:
            _ref.mirror_x(port_id)

        if v_mirror:
            _ref.mirror_y(port_id)

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
        return (
            f"{self.name}: uid {self.uid}, "
            f"ports {list(self.ports.keys())}, "
            f"references {list(self.named_references.keys())}, "
            f"{len(self.polygons)} polygons"
        )

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
    def metadata_child(self) -> Dict:
        """Returns metadata from child if any, Otherwise returns component own.

        metadata Great to access the children metadata at the bottom of the
        hierarchy.
        """
        settings = dict(self.settings)

        while settings.get("child"):
            settings = settings.get("child")

        return dict(settings)

    @property
    def metadata(self) -> Dict:
        return dict(self.settings)

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
    ) -> Port:
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
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        if port:
            if not isinstance(port, Port):
                raise ValueError(f"add_port() needs a Port, got {type(port)}")
            p = port.copy()
            if name is not None:
                p.name = name
            p.parent = self

        elif isinstance(name, Port):
            p = name.copy()
            p.parent = self
            name = p.name
        elif center is None:
            raise ValueError("Port needs center parameter (x, y) um.")

        else:
            p = Port(
                name=name,
                center=center,
                width=width,
                orientation=orientation,
                parent=self,
                layer=layer,
                port_type=port_type,
                cross_section=cross_section,
            )
        if name is not None:
            p.name = name
        if p.name in self.ports:
            raise ValueError(f"add_port() Port name {p.name!r} exists in {self.name!r}")

        self.ports[p.name] = p
        return p

    def add_ports(
        self,
        ports: Union[List[Port], Dict[str, Port]],
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """Add a list or dict of ports.

        you can include a prefix to add to the new port names to avoid name conflicts.

        Args:
            ports: list or dict of ports.
            prefix: to prepend to each port name.
        """
        ports = ports if isinstance(ports, list) else ports.values()
        for port in list(ports):
            name = f"{prefix}{port.name}{suffix}"
            self.add_port(name=name, port=port)

    def snap_ports_to_grid(self, nm: int = 1) -> None:
        for port in self.ports.values():
            port.snap_to_grid(nm=nm)

    def remove_layers(
        self,
        layers: List[LayerSpec],
        include_labels: bool = True,
        invert_selection: bool = False,
        recursive: bool = True,
    ) -> Component:
        """Remove a list of layers and returns the same Component.

        Args:
            layers: list of layers to remove.
            include_labels: remove labels on those layers.
            invert_selection: removes all layers except layers specified.
            recursive: operate on the cells included in this cell.
        """
        from gdsfactory import get_layer

        component = self.flatten() if recursive and self.references else self
        layers = [get_layer(layer) for layer in layers]

        should_remove = not invert_selection
        component._cell.filter(
            spec=layers,
            remove=should_remove,
            polygons=True,
            paths=True,
            labels=include_labels,
        )
        return component

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

    def add_polygon(self, points, layer=np.nan):
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        if layer is None:
            return None

        try:
            if isinstance(layer, set):
                return [self.add_polygon(points, ly) for ly in layer]
            elif all(isinstance(ly, (Layer)) for ly in layer):
                return [self.add_polygon(points, ly) for ly in layer]
            elif len(layer) > 2:  # Someone wrote e.g. layer = [1,4,5]
                raise ValueError(
                    """ [PHIDL] If specifying multiple layers
                you must use set notation, e.g. {1,5,8} """
                )
        except Exception:
            pass

        if isinstance(points, gdstk.Polygon):
            # if layer is unspecified or matches original polygon, just add it as-is
            polygon = points
            if layer is np.nan or (
                isinstance(layer, tuple) and (polygon.layer, polygon.datatype) == layer
            ):
                polygon = Polygon(polygon.points, polygon.layer, polygon.datatype)
            else:
                layer, datatype = _parse_layer(layer)
                polygon = Polygon(polygon.points, layer, datatype)
            self._add_polygons(polygon)
            return polygon

        points = np.asarray(points)
        if points.ndim == 1:
            return [self.add_polygon(poly, layer=layer) for poly in points]
        if layer is np.nan:
            layer = 0

        if points.ndim == 2:
            # add single polygon from points
            if len(points[0]) > 2:
                # Convert to form [[1,2],[3,4],[5,6]]
                points = np.column_stack(points)
            layer, datatype = _parse_layer(layer)
            polygon = Polygon(points, layer=layer, datatype=datatype)
            self._add_polygons(polygon)
            return polygon
        elif points.ndim == 3:
            layer, datatype = _parse_layer(layer)
            polygons = [
                Polygon(ppoints, layer=layer, datatype=datatype) for ppoints in points
            ]
            self._add_polygons(*polygons)
            return polygons
        else:
            raise ValueError(f"Unable to add {points.ndim}-dimensional points object")

    def _add_polygons(self, *polygons: List[Polygon]):
        self.is_unlocked()
        self._cell.add(*polygons)

    def copy(self) -> Component:
        return copy(self)

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
        if isinstance(element, ComponentReference):
            self._cell.add(element._reference)
            self._references.append(element)
        else:
            self._cell.add(element)

    def add(self, element) -> None:
        """Add a new element or list of elements to this Component.

        Args:
            element: Polygon, ComponentReference or iterable
                The element or iterable of elements to be inserted in this cell.

        Raises:
            MutabilityError: if component is locked.
        """
        if isinstance(element, ComponentReference):
            self._register_reference(element)
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
        ref = ComponentReference(
            component=component,
            columns=int(round(columns)),
            rows=int(round(rows)),
            spacing=spacing,
        )
        ref.name = None
        self._add(ref)
        self._register_reference(reference=ref, alias=alias)
        return ref

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

    def flatten_reference(self, ref: ComponentReference):
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
        ref = ComponentReference(component, **kwargs)
        self._add(ref)
        self._register_reference(reference=ref, alias=alias)
        return ref

    def _register_reference(
        self, reference: ComponentReference, alias: Optional[str] = None
    ) -> None:
        component = reference.parent
        reference.owner = self

        if alias is None:
            if reference.name is not None:
                alias = reference.name
            else:
                prefix = (
                    component.settings.function_name
                    if hasattr(component, "settings")
                    and hasattr(component.settings, "function_name")
                    else component.name
                )
                self._reference_names_counter.update({prefix: 1})
                alias = f"{prefix}_{self._reference_names_counter[prefix]}"

                while alias in self._named_references:
                    self._reference_names_counter.update({prefix: 1})
                    alias = f"{prefix}_{self._reference_names_counter[prefix]}"

        reference.name = alias
        self._named_references[alias] = reference

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
        polygons = self._cell.get_polygons(depth=None)
        return {(polygon.layer, polygon.datatype) for polygon in polygons}

    def _ipython_display_(self) -> None:
        """Show geometry in KLayout and in matplotlib for Jupyter Notebooks."""
        self.show(show_ports=True)  # show in klayout
        self.plot_klayout()
        print(self)

    def add_pins_triangle(
        self,
        port_marker_layer: Layer = (1, 10),
    ):
        """Returns component with triangular pins."""
        from gdsfactory.add_pins import add_pins_triangle

        component = self.copy()
        component.name = self.name
        add_pins_triangle(component=component, layer=port_marker_layer)
        return component

    def plot_widget(
        self,
        show_ports: bool = True,
        port_marker_layer: Layer = (1, 10),
    ):
        """Returns ipython widget for klayout visualization.

        Args:
            show_ports: shows component with port markers and labels.
            port_marker_layer: for the ports.
        """
        from gdsfactory.pdk import get_layer_views
        from gdsfactory.widgets.layout_viewer import LayoutViewer
        from IPython.display import display

        component = (
            self.add_pins_triangle(port_marker_layer=port_marker_layer)
            if show_ports
            else self
        )

        gdspath = component.write_gds(logging=False)
        lyp_path = gdspath.with_suffix(".lyp")

        layer_views = get_layer_views()
        layer_views.to_lyp(filepath=lyp_path)

        layout = LayoutViewer(gdspath, lyp_path)
        display(layout.widget)

    def plot_klayout(
        self,
        show_ports: bool = True,
        port_marker_layer: Layer = (1, 10),
    ) -> None:
        """Returns klayout image.

        If it fails to import klayout defaults to matplotlib.

        Args:
            show_ports: shows component with port markers and labels.
            port_marker_layer: for the ports.
        """

        component = (
            self.add_pins_triangle(port_marker_layer=port_marker_layer)
            if show_ports
            else self
        )

        try:
            import klayout.db as db  # noqa: F401
            import klayout.lay as lay
            from gdsfactory.pdk import get_layer_views
            from IPython.display import display
            from ipywidgets import Image

            gdspath = component.write_gds(logging=False)
            lyp_path = gdspath.with_suffix(".lyp")

            layer_views = get_layer_views()
            layer_views.to_lyp(filepath=lyp_path)

            layout_view = lay.LayoutView()
            layout_view.load_layout(str(gdspath.absolute()))
            layout_view.max_hier()
            layout_view.load_layer_props(str(lyp_path))

            pixel_buffer = layout_view.get_pixels_with_options(800, 600)
            png_data = pixel_buffer.to_png_data()
            image = Image(value=png_data, format="png")
            display(image)

        except ImportError:
            print(
                "You can install `pip install gdsfactory[full]` for better visualization"
            )
            component.plot(plotter="matplotlib")

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

            dirpath = pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
            dirpath.mkdir(exist_ok=True, parents=True)
            lyp_path = dirpath / "layers.lyp"

            layer_props = get_layer_views()
            layer_props.to_lyp(filepath=lyp_path)

            src = f"http://127.0.0.1:8000/gds?gds_file={escape(str(gdspath))}&layer_props={escape(str(lyp_path))}"
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
        plotter = plotter or CONF.get("plotter", "matplotlib")

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
                warnings.warn(f"{layer!r} not defined in {layers}")
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
        port_marker_layer: Layer = (1, 10),
        **kwargs,
    ) -> None:
        """Show component in KLayout.

        returns a copy of the Component, so the original component remains intact.
        with pins markers on each port show_ports = True, and optionally also
        the ports from the references (show_subports=True)

        Args:
            show_ports: shows component with port markers and labels.
            show_subports: add ports markers and labels to references.
            port_marker_layer: for the ports.

        Keyword Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/.
            unit: unit size for objects in library. 1um by default.
            precision: for object dimensions in the library (m). 1nm by default.
            timestamp: Defaults to 2019-10-25. If None uses current time.
        """
        from gdsfactory.add_pins import add_pins_triangle
        from gdsfactory.show import show

        component = (
            self.add_pins_triangle(port_marker_layer=port_marker_layer)
            if show_ports
            else self
        )

        if show_subports:
            component = self.copy()
            component.name = self.name
            for reference in component.references:
                if isinstance(component, ComponentReference):
                    add_pins_triangle(
                        component=component,
                        reference=reference,
                        layer=port_marker_layer,
                    )

        show(component, **kwargs)

    def _write_library(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: Optional[PathType] = None,
        timestamp: Optional[datetime.datetime] = _timestamp2019,
        logging: bool = True,
        with_oasis: bool = False,
        **kwargs,
    ) -> Path:
        """Write component to GDS or OASIS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            timestamp: Defaults to 2019-10-25 for consistent hash.
                If None uses current time.
            logging: disable GDS path logging, for example for showing it in KLayout.
            with_oasis: If True, file will be written to OASIS. Otherwise, file will be written to GDS.

        Keyword Args:
            Keyword arguments will override the active PDK's default GdsWriteSettings and OasisWriteSettings.

            Gds settings:
                unit: unit size for objects in library. 1um by default.
                precision: for dimensions in the library (m). 1nm by default.
                on_duplicate_cell: specify how to resolve duplicate-named cells. Choose one of the following:
                    "warn" (default): overwrite all duplicate cells with one of the duplicates (arbitrarily).
                    "error": throw a ValueError when attempting to write a gds with duplicate cells.
                    "overwrite": overwrite all duplicate cells with one of the duplicates, without warning.
                    None: do not try to resolve (at your own risk!)
                flatten_invalid_refs: flattens component references which have invalid transformations.
                max_points: Maximal number of vertices per polygon. Polygons with more vertices that this are automatically fractured.
            Oasis settings:
                compression_level: Level of compression for cells (between 0 and 9).
                    Setting to 0 will disable cell compression, 1 gives the best speed and 9, the best compression.
                detect_rectangles: Store rectangles in compressed format.
                detect_trapezoids: Store trapezoids in compressed format.
                circle_tolerance: Tolerance for detecting circles. If less or equal to 0, no detection is performed. Circles are stored in compressed format.
                validation ("crc32", "checksum32", None): type of validation to include in the saved file.
                standard_properties: Store standard OASIS properties in the file.

        """

        from gdsfactory.pdk import get_active_pdk

        if gdspath and gdsdir:
            warnings.warn(
                "gdspath and gdsdir have both been specified. gdspath will take precedence and gdsdir will be ignored."
            )

        default_settings = get_active_pdk().gds_write_settings
        default_oasis_settings = get_active_pdk().oasis_settings

        explicit_gds_settings = {
            k: v
            for k, v in kwargs.items()
            if v is not None and k in default_settings.dict()
        }
        explicit_oas_settings = {
            k: v
            for k, v in kwargs.items()
            if v is not None and k in default_oasis_settings.dict()
        }
        # update the write settings with any settings explicitly passed
        write_settings = default_settings.copy(update=explicit_gds_settings)
        oasis_settings = default_oasis_settings.copy(update=explicit_oas_settings)

        if write_settings.flatten_invalid_refs:
            top_cell = flatten_invalid_refs_recursive(self)
        else:
            top_cell = self

        gdsdir = (
            gdsdir or pathlib.Path(tempfile.TemporaryDirectory().name) / "gdsfactory"
        )
        gdsdir = pathlib.Path(gdsdir)
        if with_oasis:
            gdspath = gdspath or gdsdir / f"{top_cell.name}.oas"
        else:
            gdspath = gdspath or gdsdir / f"{top_cell.name}.gds"
        gdspath = pathlib.Path(gdspath)
        gdsdir = gdspath.parent
        gdsdir.mkdir(exist_ok=True, parents=True)

        cells = top_cell.get_dependencies(recursive=True)
        cell_names = [cell.name for cell in list(cells)]
        cell_names_unique = set(cell_names)

        if len(cell_names) != len(set(cell_names)):
            for cell_name in cell_names_unique:
                cell_names.remove(cell_name)

            if write_settings.on_duplicate_cell == "error":
                raise ValueError(
                    f"Duplicated cell names in {top_cell.name!r}: {cell_names!r}"
                )
            elif write_settings.on_duplicate_cell in {"warn", "overwrite"}:
                if write_settings.on_duplicate_cell == "warn":
                    warnings.warn(
                        f"Duplicated cell names in {top_cell.name!r}:  {cell_names}",
                    )
                cells_dict = {cell.name: cell._cell for cell in cells}
                cells = cells_dict.values()
            elif write_settings.on_duplicate_cell is not None:
                raise ValueError(
                    f"on_duplicate_cell: {write_settings.on_duplicate_cell!r} not in (None, warn, error, overwrite)"
                )

        all_cells = [top_cell._cell] + sorted(cells, key=lambda cc: cc.name)

        no_name_cells = [
            cell.name for cell in all_cells if cell.name.startswith("Unnamed")
        ]

        if no_name_cells:
            warnings.warn(
                f"Component {top_cell.name!r} contains {len(no_name_cells)} Unnamed cells"
            )

        # for cell in all_cells:
        #     print(cell.name, type(cell))

        lib = gdstk.Library(
            unit=write_settings.unit, precision=write_settings.precision
        )
        lib.add(top_cell._cell)
        lib.add(*top_cell._cell.dependencies(True))

        if with_oasis:
            lib.write_oas(gdspath, **oasis_settings.dict())
        else:
            lib.write_gds(
                gdspath, timestamp=timestamp, max_points=write_settings.max_points
            )
        if logging:
            logger.info(f"Wrote to {str(gdspath)!r}")
        return gdspath

    def write_gds(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: Optional[PathType] = None,
        unit: Optional[float] = None,
        precision: Optional[float] = None,
        logging: bool = True,
        on_duplicate_cell: Optional[str] = None,
        flatten_invalid_refs: Optional[bool] = None,
        max_points: Optional[int] = None,
    ) -> Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            unit: unit size for objects in library. 1um by default.
            precision: for dimensions in the library (m). 1nm by default.
            logging: disable GDS path logging, for example for showing it in KLayout.
            on_duplicate_cell: specify how to resolve duplicate-named cells. Choose one of the following:
                "warn" (default): overwrite all duplicate cells with one of the duplicates (arbitrarily).
                "error": throw a ValueError when attempting to write a gds with duplicate cells.
                "overwrite": overwrite all duplicate cells with one of the duplicates, without warning.
            flatten_invalid_refs: flattens component references which have invalid transformations.
            max_points: Maximal number of vertices per polygon.
                Polygons with more vertices that this are automatically fractured.
        """

        return self._write_library(
            gdspath=gdspath,
            gdsdir=gdsdir,
            unit=unit,
            precision=precision,
            logging=logging,
            on_duplicate_cell=on_duplicate_cell,
            flatten_invalid_refs=flatten_invalid_refs,
            max_points=max_points,
        )

    def write_oas(
        self,
        gdspath: Optional[PathType] = None,
        gdsdir: Optional[PathType] = None,
        unit: Optional[float] = None,
        precision: Optional[float] = None,
        logging: bool = True,
        on_duplicate_cell: Optional[str] = "warn",
        flatten_invalid_refs: Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            unit: unit size for objects in library. 1um by default.
            precision: for dimensions in the library (m). 1nm by default.
            logging: disable GDS path logging, for example for showing it in KLayout.
            on_duplicate_cell: specify how to resolve duplicate-named cells. Choose one of the following:
                "warn" (default): overwrite all duplicate cells with one of the duplicates (arbitrarily).
                "error": throw a ValueError when attempting to write a gds with duplicate cells.
                "overwrite": overwrite all duplicate cells with one of the duplicates, without warning.
                None: do not try to resolve (at your own risk!)
            flatten_invalid_refs: flattens component references which have invalid transformations.

        Keyword Args:
            compression_level: Level of compression for cells (between 0 and 9).
                Setting to 0 will disable cell compression, 1 gives the best speed and 9, the best compression.
            detect_rectangles: Store rectangles in compressed format.
            detect_trapezoids: Store trapezoids in compressed format.
            circle_tolerance: Tolerance for detecting circles. If less or equal to 0, no detection is performed.
                Circles are stored in compressed format.
            validation ("crc32", "checksum32", None) – type of validation to include in the saved file.
            standard_properties: Store standard OASIS properties in the file.
        """
        return self._write_library(
            gdspath=gdspath,
            gdsdir=gdsdir,
            unit=unit,
            precision=precision,
            logging=logging,
            on_duplicate_cell=on_duplicate_cell,
            with_oasis=True,
            flatten_invalid_refs=flatten_invalid_refs,
            **kwargs,
        )

    def write_gds_with_metadata(self, *args, **kwargs) -> Path:
        """Write component in GDS and metadata (component settings) in YAML."""
        gdspath = self.write_gds(*args, **kwargs)
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

    def mirror(self, p1: Float2 = (0, 1), p2: Float2 = (0, 0), **kwargs) -> Component:
        """Returns new Component with a mirrored reference.

        Args:
            p1: first point to define mirror axis.
            p2: second point to define mirror axis.
        """
        from gdsfactory.functions import mirror

        return mirror(component=self, p1=p1, p2=p2, **kwargs)

    def rotate(self, angle: float = 90, **kwargs) -> Component:
        """Returns new component with a rotated reference to the original.

        Args:
            angle: in degrees.
        """
        from gdsfactory.functions import rotate

        return rotate(component=self, angle=angle, **kwargs)

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
        ref_polygons = reference.get_polygons(
            by_spec=False, include_paths=False, as_array=False
        )
        self._add_polygons(*ref_polygons)

        self.add(reference.get_labels())
        self.add(reference.get_paths())
        self.remove(reference)
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
        layer_stack: Optional[LayerStack] = None,
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

    def to_stl(
        self,
        filepath: str,
        layer_views: Optional[LayerViews] = None,
        layer_stack: Optional[LayerStack] = None,
        exclude_layers: Optional[Tuple[Layer, ...]] = None,
    ) -> np.ndarray:
        """Exports a Component into STL.

        Args:
            component: to export.
            filepath: to write STL to.
            layer_views: layer colors from Klayout Layer Properties file.
            layer_stack: contains thickness and zmin for each layer.
            exclude_layers: layers to exclude.

        """
        from gdsfactory.export.to_stl import to_stl

        return to_stl(
            self,
            filepath=filepath,
            layer_views=layer_views,
            layer_stack=layer_stack,
            exclude_layers=exclude_layers,
        )

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
        # Add WAFER layer:
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
    component: Component,
    grid_size: Optional[float] = None,
    updated_components=None,
    traversed_components=None,
):
    """Recursively flattens component references which have invalid transformations (i.e. non-90 deg rotations or sub-grid translations) and returns a copy if any subcells have been modified.

    WARNING: this function will produce same-name copies of cells. It is strictly meant to be used on write of the GDS file and
    should not be mixed with other cells, or you will likely experience issues with duplicate cells

    Args:
        component: the component to fix (in place).
        grid_size: the GDS grid size, in um, defaults to active PDK.get_grid_size()
            any translations with higher resolution than this are considered invalid.
        updated_components: the running dictionary of components which have been modified by this transformation. Should always be None, except for recursive invocations.
        traversed_components: the set of component names which have been traversed. Should always be None, except for recursive invocations.
    """
    from gdsfactory.decorators import is_invalid_ref

    invalid_refs = []
    refs = component.references
    subcell_modified = False
    if updated_components is None:
        updated_components = {}
    if traversed_components is None:
        traversed_components = set()
    for ref in refs:
        # mark any invalid refs for flattening
        # otherwise, check if there are any modified cells beneath (we need not do this if the ref will be flattened anyways)
        if is_invalid_ref(ref, grid_size):
            invalid_refs.append(ref.name)
        else:
            # otherwise, recursively flatten refs if the subcell has not already been traversed
            if ref.parent.name not in traversed_components:
                flatten_invalid_refs_recursive(
                    ref.parent,
                    grid_size=grid_size,
                    updated_components=updated_components,
                    traversed_components=traversed_components,
                )
            # now, if the ref's cell been modified, mark it as such
            if ref.parent.name in updated_components:
                subcell_modified = True
    if invalid_refs or subcell_modified:
        new_component = component.copy()
        new_component.name = component.name
        # make sure all modified cells have their references updated
        new_refs = new_component.references.copy()
        for ref in new_refs:
            if ref.name in invalid_refs:
                new_component.flatten_reference(ref)
            elif (
                ref.parent.name in updated_components
                and ref.parent is not updated_components[ref.parent.name]
            ):
                ref.parent = updated_components[ref.parent.name]
        component = new_component
        updated_components[component.name] = new_component
    traversed_components.add(component.name)
    return component


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


def test_import_gds_settings():
    import gdsfactory as gf

    c = gf.components.mzi()
    gdspath = c.write_gds_with_metadata()
    c2 = gf.import_gds(gdspath, name="mzi_sample", read_metadata=True)
    c3 = gf.routing.add_fiber_single(c2)
    assert c3


def test_flatten_invalid_refs_recursive():
    import gdsfactory as gf
    from gdsfactory.difftest import run_xor
    from gdsfactory.routing.all_angle import get_bundle_all_angle

    @gf.cell
    def flat():
        c = gf.Component()
        mmi1 = (c << gf.components.mmi1x2()).move((0, -1.0005))
        mmi2 = (c << gf.components.mmi1x2()).rotate(80)
        mmi2.move((40, 20))
        bundle = get_bundle_all_angle([mmi1.ports["o2"]], [mmi2.ports["o1"]])
        for route in bundle:
            c.add(route.references)
        return c

    @gf.cell
    def hierarchy():
        c = gf.Component()
        (c << flat()).rotate(33)
        (c << flat()).rotate(33).move((0, 100))
        (c << flat()).move((100, 0))
        return c

    c_orig = hierarchy()
    c_new = flatten_invalid_refs_recursive(c_orig)
    assert c_new is not c_orig
    invalid_refs_filename = "invalid_refs.gds"
    invalid_refs_fixed_filename = "invalid_refs_fixed.gds"
    # gds files should still be same to 1nm tolerance
    c_orig.write_gds(invalid_refs_filename)
    c_new.write_gds(invalid_refs_fixed_filename)
    run_xor(invalid_refs_filename, invalid_refs_fixed_filename)


if __name__ == "__main__":
    import gdsfactory as gf

    # c2 = gf.Component()
    c = gf.components.mzi(delta_length=20)
    # r = c.ref()
    # c2.copy_child_info(c.named_references["sxt"])
    # test_remap_layers()
    # c = test_get_layers()
    # c.plot_qt()
    # c.ploth()
    # c = test_extract()
    # gdspath = c.write_gds()
    # gf.show(gdspath)
    # c.show(show_ports=True)
    c.plot_klayout()
