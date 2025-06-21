"""Component is a canvas for geometry."""

from __future__ import annotations

import pathlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias, cast, overload

import kfactory as kf
import klayout.lay as lay
import networkx as nx
import numpy as np
import numpy.typing as npt
import yaml
from graphviz import Digraph
from kfactory import (
    DInstance,
    DInstances,
    DPort,
    DPorts,
    VInstance,
    cell,
    kdb,
    save_layout_options,
)
from kfactory.exceptions import LockedError
from kfactory.kcell import BaseKCell, ProtoKCell
from kfactory.port import ProtoPort
from matplotlib.figure import Figure
from pydantic import Field
from trimesh.scene.scene import Scene

from gdsfactory.config import CONF, GDSDIR_TEMP
from gdsfactory.serialization import clean_value_json, convert_tuples_to_lists
from gdsfactory.utils import to_kdb_dpoints

if TYPE_CHECKING:
    from gdsfactory.cross_section import CrossSection, CrossSectionSpec
    from gdsfactory.technology.layer_stack import LayerStack
    from gdsfactory.technology.layer_views import LayerViews
    from gdsfactory.typings import (
        AngleInDegrees,
        ComponentSpec,
        Coordinates,
        Layer,
        LayerSpec,
        LayerSpecs,
        PathType,
        Port,
        Position,
    )

cell_without_validator = cell

_PolygonPoints: TypeAlias = "npt.NDArray[np.floating[Any]] | kdb.DPolygon | kdb.Polygon | kdb.DSimplePolygon | kdb.Region | Coordinates"


def ensure_tuple_of_tuples(points: Any) -> tuple[tuple[float, float], ...]:
    # Convert a single NumPy array to a tuple of tuples
    if isinstance(points, np.ndarray):
        points = tuple(map(tuple, points.tolist()))
    elif isinstance(points, list):
        # If it's a list, check if the first element is an np.ndarray or a list to decide on conversion
        if len(points) > 0 and isinstance(points[0], np.ndarray | list):
            points = tuple(tuple(point) for point in points)
    return cast(tuple[tuple[float, float], ...], points)


def points_to_polygon(
    points: _PolygonPoints,
) -> kdb.Polygon | kdb.DPolygon | kdb.DSimplePolygon | kdb.Region:
    if isinstance(points, tuple | list | np.ndarray):
        points = ensure_tuple_of_tuples(points)
        polygon = kdb.DPolygon()
        polygon.assign_hull(to_kdb_dpoints(points))
    elif isinstance(
        points, kdb.Polygon | kdb.DPolygon | kdb.DSimplePolygon | kdb.Region
    ):
        return points
    return kdb.DPolygon(to_kdb_dpoints(points))


def size(region: kdb.Region, offset: float, dbu: float = 1e3) -> kdb.Region:
    return region.dup().size(int(offset * dbu))


def boolean_or(region1: kdb.Region, region2: kdb.Region) -> kdb.Region:
    return (region1.__or__(region2)).merge()


def boolean_not(region1: kdb.Region, region2: kdb.Region) -> kdb.Region:
    return kdb.Region.__sub__(region1, region2)


def boolean_xor(region1: kdb.Region, region2: kdb.Region) -> kdb.Region:
    return kdb.Region.__xor__(region1, region2)


def boolean_and(region1: kdb.Region, region2: kdb.Region) -> kdb.Region:
    return kdb.Region.__and__(region1, region2)


boolean_operations = {
    "or": boolean_or,
    "|": boolean_or,
    "not": boolean_not,
    "-": boolean_not,
    "^": boolean_xor,
    "xor": boolean_xor,
    "&": boolean_and,
    "and": boolean_and,
    "A-B": boolean_not,
}


def copy(region: kdb.Region) -> kdb.Region:
    return region.dup()


ComponentReference: TypeAlias = DInstance
ComponentReferences: TypeAlias = DInstances


class ComponentBase(ProtoKCell[float, BaseKCell], ABC):
    """Canvas where you add polygons, instances and ports.

    - stores settings that you use to build the component
    - stores info that you want to use
    - can return ports by type (optical, electrical ...)
    - can return netlist for circuit simulation
    - can write to GDS, OASIS
    - can show in KLayout, matplotlib or 3D

    Properties:
        info: dictionary that includes derived properties, simulation_settings, settings (test_protocol, docs, ...)
    """

    @property
    def layers(self) -> list[Layer]:
        return [
            (info.layer, info.datatype)
            for info in self.kcl.layout.layer_infos()
            if not self.bbox(self.kcl.layout.layer(info)).empty()
        ]

    @abstractmethod
    def add_polygon(
        self, points: _PolygonPoints, layer: LayerSpec
    ) -> kdb.Shape | None: ...

    def bbox_np(self) -> npt.NDArray[np.float64]:
        """Returns the bounding box of the Component as a numpy array."""
        return np.array(
            [[self.xmin, self.ymin], [self.xmax, self.ymax]], dtype=np.float64
        )

    def add_port(
        self,
        name: str | None = None,
        *,
        port: ProtoPort[Any] | None = None,
        center: Position | kdb.DPoint | None = None,
        width: float | None = None,
        orientation: AngleInDegrees = 0,
        layer: LayerSpec | None = None,
        port_type: str = "optical",
        keep_mirror: bool = False,
        cross_section: "CrossSectionSpec | None" = None,
    ) -> DPort:
        """Adds a Port to the Component.

        Args:
            name: name of the port.
            port: port to add.
            center: center of the port.
            width: width of the port.
            orientation: orientation of the port.
            layer: layer spec to add port on.
            port_type: port type (optical, electrical, ...)
            keep_mirror: if True, keeps the mirror of the port.
            cross_section: cross_section of the port.
        """
        if self.locked:
            raise LockedError(self)

        if port:
            return DPort(
                base=super()
                .add_port(port=port, name=name, keep_mirror=keep_mirror)
                .base
            )

        from gdsfactory.config import CONF
        from gdsfactory.pdk import get_cross_section, get_layer

        if port_type not in CONF.port_types:
            warnings.warn(
                f"Port type {port_type} not in {CONF.port_types}. "
                "Please add it to the port_types list in the config gf.CONF.port_types.",
                stacklevel=3,
            )

        if layer is None:
            if cross_section is None:
                raise ValueError("Must specify layer or cross_section")
            xs = get_cross_section(cross_section)
            layer = xs.layer

        if width is None:
            if cross_section is None:
                raise ValueError("Must specify width or cross_section")
            xs = get_cross_section(cross_section)
            width = xs.width

        if center is None:
            raise ValueError("Must specify center")

        elif isinstance(center, kdb.DPoint):
            layer = get_layer(layer)
            trans = kdb.DCplxTrans(1, orientation, False, center.to_v())
        else:
            layer = get_layer(layer)
            x = float(center[0])
            y = float(center[1])
            trans = kdb.DCplxTrans(1, float(orientation), False, x, y)

        _port = DPorts(kcl=self.kcl, bases=self.ports.bases).create_port(
            name=name, width=width, layer=layer, port_type=port_type, dcplx_trans=trans
        )
        if cross_section:
            xs = get_cross_section(cross_section)
            _port.info["cross_section"] = xs.name

        return _port

    def copy(self) -> Self:
        """Copy the full cell."""
        return self.dup()

    def add_label(
        self,
        text: str = "hello",
        position: Position | kf.kdb.DPoint = (0.0, 0.0),
        layer: LayerSpec = "TEXT",
    ) -> None:
        """Adds Label to the Component.

        Args:
            text: Label text.
            position: x-, y-coordinates of the Label location.
            layer: Specific layer(s) to put Label on.
        """
        from gdsfactory.pdk import get_layer

        if self.locked:
            raise LockedError(self)

        layer = get_layer(layer)
        if isinstance(position, kf.kdb.DPoint):
            x, y = position.x, position.y
        else:
            x, y = position

        trans = kdb.DTrans(0, False, x, y)
        self.shapes(layer).insert(kf.kdb.DText(text, trans))

    def get_ports_list(self, **kwargs: Any) -> "list[Port]":
        """Returns list of ports.

        Args:
            kwargs: Additional kwargs.

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
        from gdsfactory.port import select_ports

        return select_ports(ports=self.ports.to_dtype(), **kwargs)

    def add_route_info(
        self,
        cross_section: "CrossSection | str",
        length: float,
        length_eff: float | None = None,
        taper: bool = False,
        **kwargs: Any,
    ) -> None:
        """Adds route information to a component.

        Args:
            cross_section: CrossSection or name of the cross_section.
            length: length of the route.
            length_eff: effective length of the route.
            taper: if True adds taper information.
            kwargs: extra information to add to the component.
        """
        from gdsfactory.pdk import get_active_pdk

        if self.locked:
            raise LockedError(self)

        pdk = get_active_pdk()

        length_eff = length_eff or length
        xs_name = (
            cross_section
            if isinstance(cross_section, str)
            else pdk.get_cross_section_name(cross_section)
        )

        info = self.info
        if taper:
            info[f"route_info_{xs_name}_taper_length"] = length

        info["route_info_type"] = xs_name
        info["route_info_length"] = length_eff
        info["route_info_weight"] = length_eff
        info[f"route_info_{xs_name}_length"] = length_eff
        for key, value in kwargs.items():
            info[f"route_info_{key}"] = value

    def copy_child_info(self, component: kf.ProtoTKCell[Any]) -> None:
        """Copy and settings info from child component into parent.

        Parent components can access child cells settings.
        """
        if self.locked:
            raise LockedError(self)

        info = dict(component.info)

        for k, v in info.items():
            if k not in self.info:
                self.info[k] = v

    def write_gds(
        self,
        gdspath: "PathType | None" = None,
        gdsdir: "PathType | None" = None,
        save_options: "kdb.SaveLayoutOptions | None" = None,
        with_metadata: bool = True,
    ) -> pathlib.Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            save_options: klayout save options.
            with_metadata: if True, writes metadata (ports, settings) to the GDS file.
        """
        if gdspath and gdsdir:
            warnings.warn(
                "gdspath and gdsdir have both been specified. "
                "gdspath will take precedence and gdsdir will be ignored.",
                stacklevel=3,
            )
        gdsdir = gdsdir or GDSDIR_TEMP
        gdsdir = pathlib.Path(gdsdir)
        gdsdir.mkdir(parents=True, exist_ok=True)
        name = self.name or ""
        gdspath = gdspath or gdsdir / f"{name[: CONF.max_cellname_length]}.gds"
        gdspath = pathlib.Path(gdspath)

        gdspath.parent.mkdir(parents=True, exist_ok=True)

        if save_options is None:
            save_options = save_layout_options()

        if not with_metadata:
            save_options.write_context_info = False

        self.write(filename=gdspath, save_options=save_options)
        return pathlib.Path(gdspath)

    def pprint_ports(self, **kwargs: Any) -> None:
        """Pretty prints ports.

        Args:
            kwargs: keyword arguments to filter ports.

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
        ports = self.get_ports_list(**kwargs)
        from gdsfactory.port import pprint_ports

        pprint_ports(ports)

    def write_netlist(
        self, netlist: dict[str, Any], filepath: str | pathlib.Path | None = None
    ) -> str:
        """Returns netlist as YAML string.

        Args:
            netlist: netlist to write.
            filepath: Optional file path to write to.
        """
        netlist_converted = convert_tuples_to_lists(netlist)
        yaml_string = yaml.dump(netlist_converted)
        if filepath:
            filepath = pathlib.Path(filepath)
            filepath.write_text(yaml_string)
        return str(yaml_string)

    def to_dict(self, with_ports: bool = False) -> dict[str, Any]:
        """Returns a dictionary representation of the Component."""
        from gdsfactory.port import to_dict

        d = {
            "name": self.name,
            "info": self.info.model_dump(exclude_none=True),
            "settings": self.settings.model_dump(exclude_none=True),
        }

        if with_ports:
            d["ports"] = {
                port.name: to_dict(port) for port in self.ports if port.name is not None
            }
        res = clean_value_json(d)
        assert isinstance(res, dict)
        return res

    def get_netlist(self, recursive: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Returns a place-aware netlist for circuit simulation.

        It includes not only the connectivity information (nodes and connections) but also the specific placement coordinates for each component or cell in the layout.

        Args:
            recursive: if True, returns a recursive netlist.
            kwargs: keyword arguments to get_netlist.
        """
        from gdsfactory.get_netlist import get_netlist, get_netlist_recursive

        if recursive:
            return get_netlist_recursive(self, **kwargs)

        return get_netlist(self, **kwargs)  # type: ignore[arg-type]


Route: TypeAlias = (
    kf.routing.generic.ManhattanRoute | kf.routing.aa.optical.OpticalAllAngleRoute
)


class Component(ComponentBase, kf.DKCell):
    """Canvas where you add polygons, instances and ports.

    - stores settings that you use to build the component
    - stores info that you want to use
    - can return ports by type (optical, electrical ...)
    - can return netlist for circuit simulation
    - can write to GDS, OASIS
    - can show in KLayout, matplotlib or 3D

    Properties:
        info: dictionary that includes derived properties, simulation_settings, settings (test_protocol, docs, ...)
    """

    routes: "dict[str, Route]" = Field(default_factory=dict)

    @property
    def layers(self) -> list[Layer]:
        return [
            (info.layer, info.datatype)
            for info in self.kcl.layout.layer_infos()
            if not self.bbox(self.kcl.layout.layer(info)).empty()
        ]

    def add(self, instances: Iterable[ComponentReference] | ComponentReference) -> None:
        if self.locked:
            raise LockedError(self)

        if not isinstance(instances, Iterable):
            instance_list = [instances]
        else:
            instance_list = list(instances)

        for instance in instance_list:
            self.kdb_cell.insert(instance.instance)

    def absorb(self, reference: ComponentReference) -> Self:
        """Absorbs polygons from ComponentReference into Component.

        Destroys the reference in the process but keeping the polygon geometry.

        Args:
            reference: Instance to be absorbed into the Component.

        """
        if self.locked:
            raise LockedError(self)
        if reference not in self.insts:
            raise ValueError(
                "The reference you asked to absorb does not exist in this Component."
            )
        reference.flatten()
        return self

    def trim(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        flatten: bool = False,
    ) -> None:
        """Trims the Component to a bounding box.

        Args:
            left: left coordinate of the bounding box.
            bottom: bottom coordinate of the bounding box.
            right: right coordinate of the bounding box.
            top: top coordinate of the bounding box.
            flatten: if True, flattens the Component.
        """
        if self.locked:
            raise LockedError(self)

        c = self

        domain_box = kdb.DBox(left, bottom, right, top)
        if not c.dbbox().inside(domain_box):
            kdb_cell = c.kcl.layout.clip(c.kdb_cell, kdb.DBox(left, bottom, right, top))
            c.kdb_cell.clear()
            c.kdb_cell.copy_tree(kdb_cell)
            kdb_cell.delete()
            if flatten:
                c.flatten()

    def add_ref(
        self,
        component: kf.ProtoTKCell[Any],
        name: str | None = None,
        columns: int = 1,
        rows: int = 1,
        column_pitch: float = 0.0,
        row_pitch: float = 0.0,
    ) -> ComponentReference:
        """Adds a component instance reference to a Component.

        Args:
            component: The referenced component.
            name: Name of the reference.
            columns: Number of columns in the array.
            rows: Number of rows in the array.
            column_pitch: column pitch.
            row_pitch: row pitch.
        """
        if self.locked:
            raise LockedError(self)

        if rows > 1 or columns > 1:
            if rows > 1 and row_pitch == 0:
                raise ValueError(f"rows = {rows} > 1 require {row_pitch=} > 0")

            if columns > 1 and column_pitch == 0:
                raise ValueError(f"columns = {columns} > 1 require {column_pitch} > 0")

            a = kf.kdb.DVector(column_pitch, 0)
            b = kf.kdb.DVector(0, row_pitch)

            inst = self.create_inst(component, na=columns, nb=rows, a=a, b=b)
        else:
            inst = self.create_inst(component)
        if name is not None:
            inst.name = name
        return ComponentReference(kcl=self.kcl, instance=inst.instance)

    def get_paths(
        self, layer: "LayerSpec", recursive: bool = True
    ) -> list[kf.kdb.DPath]:
        """Returns a list of paths.

        Args:
            layer: layer to get paths from.
            recursive: if True, gets paths recursively.
        """
        from gdsfactory import get_layer

        paths: list[kf.kdb.DPath] = []

        layer = get_layer(layer)

        if recursive:
            iterator = self.kdb_cell.begin_shapes_rec(layer)
            iterator.shape_flags = kdb.Shapes.SPaths
            paths.extend(
                it.shape().dpath.transformed(it.dtrans()) for it in iterator.each()
            )
        else:
            paths.extend(
                shape.dpath
                for shape in self.kdb_cell.shapes(layer).each(kdb.Shapes.SPaths)
            )
        return paths

    def get_boxes(
        self, layer: "LayerSpec", recursive: bool = True
    ) -> list[kf.kdb.DBox]:
        """Returns a list of boxes.

        Args:
            layer: layer to get boxes from.
            recursive: if True, gets boxes recursively.
        """
        from gdsfactory import get_layer

        boxes: list[kf.kdb.DBox] = []

        layer = get_layer(layer)

        if recursive:
            iterator = self.kdb_cell.begin_shapes_rec(layer)
            iterator.shape_flags = kdb.Shapes.SBoxes
            boxes.extend(
                it.shape().dbox.transformed(it.dtrans()) for it in iterator.each()
            )
        else:
            boxes.extend(
                shape.dbox
                for shape in self.kdb_cell.shapes(layer).each(kdb.Shapes.SBoxes)
            )
        return boxes

    def get_labels(
        self, layer: "LayerSpec", recursive: bool = True
    ) -> list[kf.kdb.DText]:
        """Returns a list of labels from the Component.

        Args:
            layer: layer to get labels from.
            recursive: if True, gets labels recursively.
        """
        from gdsfactory import get_layer

        texts: list[kf.kdb.DText] = []
        layer_enum = get_layer(layer)

        if recursive:
            iterator = self.kdb_cell.begin_shapes_rec(layer_enum)
            iterator.shape_flags = kdb.Shapes.STexts
            texts.extend(
                it.shape().dtext.transformed(it.dtrans()) for it in iterator.each()
            )
        else:
            texts.extend(
                shape.dtext
                for shape in self.kdb_cell.shapes(layer_enum).each(kdb.Shapes.STexts)
            )
        return texts

    def area(self, layer: "LayerSpec") -> float:
        """Returns the area of the Component in um2."""
        from gdsfactory import get_layer

        layer_index = get_layer(layer)
        r = kdb.Region(self.kdb_cell.begin_shapes_rec(layer_index))
        r.merge()
        return float(sum(p.area2() / 2 * self.kcl.dbu**2 for p in r.each()))

    def get_polygons(
        self,
        merge: bool = False,
        by: Literal["index", "name", "tuple"] = "index",
        layers: "LayerSpecs | None" = None,
        smooth: float | None = None,
    ) -> dict[tuple[int, int] | str | int, list[kf.kdb.Polygon]]:
        """Returns a dict of Polygons per layer.

        Args:
            merge: if True, merges the polygons.
            by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple')
            layers: list of layers to get polygons from. Defaults to all layers.
            smooth: if True, smooths the polygons.
        """
        if merge and self.locked:
            raise LockedError(self)

        from gdsfactory.functions import get_polygons

        return get_polygons(self, merge=merge, by=by, layers=layers, smooth=smooth)

    def get_region(
        self, layer: "LayerSpec", merge: bool = False, smooth: float | None = None
    ) -> kdb.Region:
        """Returns a Region of the Component.

        Note that all operations that you do with the Region will be done in the database units.

        Where for most processes 1 dbu = 1 nm.

        Args:
            layer: layer to get region from.
            merge: if True, merges the region.
            smooth: if True, smooths the region by the specified amount (in um).
        """
        from gdsfactory import get_layer

        layer_index = get_layer(layer)
        r = kdb.Region(self.kdb_cell.begin_shapes_rec(layer_index))
        if smooth:
            r.smooth(self.kcl.to_dbu(smooth))
        if merge:
            r.merge()
        return r

    def get_polygons_points(
        self,
        merge: bool = False,
        scale: float | None = None,
        by: Literal["index", "name", "tuple"] = "index",
        layers: "LayerSpecs | None" = None,
    ) -> dict[int | str | tuple[int, int], list[npt.NDArray[np.floating[Any]]]]:
        """Returns a dict with list of points per layer.

        Args:
            merge: if True, merges the polygons.
            scale: if True, scales the points.
            by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple')
            layers: list of layers to get polygons from. Defaults to all layers.
        """
        if merge and self.locked:
            raise LockedError(self)

        from gdsfactory.functions import get_polygons_points

        return get_polygons_points(self, merge=merge, scale=scale, by=by, layers=layers)

    def extract(
        self,
        layers: "LayerSpecs",
        recursive: bool = True,
    ) -> Component:
        """Extracts a list of layers and adds them to a new Component.

        Args:
            layers: list of layers to extract.
            recursive: if True, extracts layers recursively and returns a flattened Component.
        """
        from gdsfactory.functions import extract

        return extract(self, layers=layers, recursive=recursive)

    def copy_layers(
        self, layer_map: "dict[LayerSpec, LayerSpec]", recursive: bool = False
    ) -> Self:
        """Remaps a list of layers and returns the same Component.

        Args:
            layer_map: dictionary of layers to copy.
            recursive: if True, remaps layers recursively.
        """
        from gdsfactory import get_layer

        if self.locked:
            raise LockedError(self)

        for layer, new_layer in layer_map.items():
            src_layer_index = get_layer(layer)
            dst_layer_index = get_layer(new_layer)
            self.kdb_cell.copy(src_layer_index, dst_layer_index)

            if recursive:
                for ci in self.kdb_cell.called_cells():
                    self.kcl[ci].kdb_cell.copy(src_layer_index, dst_layer_index)
        return self

    def remove_layers(
        self,
        layers: "LayerSpecs",
        recursive: bool = True,
    ) -> Self:
        """Removes a list of layers and returns the same Component.

        Args:
            layers: list of layers to remove.
            recursive: if True, removes layers recursively.
        """
        from gdsfactory import get_layer

        if self.locked:
            raise LockedError(self)

        layers = [get_layer(layer) for layer in layers]
        for layer_index in layers:
            assert isinstance(layer_index, int)
            self.kdb_cell.shapes(layer_index).clear()
            if recursive:
                [
                    self.kcl[ci].kdb_cell.shapes(layer).clear()
                    for ci in self.kdb_cell.called_cells()
                    for layer in layers
                    if isinstance(layer, int)
                ]
        return self

    def remap_layers(
        self, layer_map: "dict[LayerSpec, LayerSpec]", recursive: bool = False
    ) -> Self:
        """Remaps a list of layers and returns the same Component.

        Args:
            layer_map: dictionary of layers to remap.
            recursive: if True, remaps layers recursively.
        """
        from gdsfactory import get_layer

        if self.locked:
            raise LockedError(self)

        for layer, new_layer in layer_map.items():
            src_layer_index = get_layer(layer)
            dst_layer_index = get_layer(new_layer)
            self.kdb_cell.move(src_layer_index, dst_layer_index)

            if recursive:
                for ci in self.kdb_cell.called_cells():
                    self.kcl[ci].kdb_cell.move(src_layer_index, dst_layer_index)
        return self

    def to_3d(
        self,
        layer_views: "LayerViews | None" = None,
        layer_stack: "LayerStack | None" = None,
        exclude_layers: "Sequence[Layer] | None " = None,
    ) -> Scene:
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

    def over_under(
        self, layer: "LayerSpec", distance: float = 0.001, remove_old_layer: bool = True
    ) -> None:
        """Returns a Component over-under on a layer in the Component.

        For big components use tiled version.

        Args:
            layer: layer to perform over-under on.
            distance: distance to perform over-under in um.
            remove_old_layer: if True, removes the old layer.
        """
        from gdsfactory import get_layer

        if self.locked:
            raise LockedError(self)

        distance_dbu = self.kcl.to_dbu(distance)

        layer_index = get_layer(layer)
        region = kdb.Region(self.kdb_cell.begin_shapes_rec(layer_index))
        region.size(+distance_dbu).size(-distance_dbu)
        if remove_old_layer:
            self.remove_layers([layer])
        self.kdb_cell.shapes(layer_index).insert(region)

        self.kcl.layout.end_changes()

    def offset(self, layer: "LayerSpec", distance: float) -> None:
        """Offsets a Component layer by a distance in um.

        Args:
            layer: layer to offset the Component on.
            distance: distance to offset the Component in um.
        """
        from gdsfactory import get_layer

        if self.locked:
            raise LockedError(self)

        distance_dbu = self.kcl.to_dbu(distance)

        layer_index = get_layer(layer)
        region = kdb.Region(self.kdb_cell.begin_shapes_rec(layer_index))
        region.size(distance_dbu)
        self.remove_layers([layer])
        self.kdb_cell.shapes(layer_index).insert(region)

        self.kcl.layout.end_changes()

    def add_polygon(
        self, points: _PolygonPoints, layer: "LayerSpec"
    ) -> kdb.Shape | None:
        """Adds a Polygon to the Component and returns a klayout Shape.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        if self.locked:
            raise LockedError(self)

        _layer = get_layer(layer)

        polygon = points_to_polygon(points)

        return self.kdb_cell.shapes(_layer).insert(polygon)

    @overload
    def plot(
        self,
        lyrdb: pathlib.Path | str | None = None,
        display_type: Literal["image", "widget"] | None = None,
        *,
        show_labels: bool = True,
        show_ruler: bool = True,
        return_fig: Literal[True] = True,
    ) -> Figure: ...

    @overload
    def plot(
        self,
        lyrdb: pathlib.Path | str | None = None,
        display_type: Literal["image", "widget"] | None = None,
        *,
        show_labels: bool = True,
        show_ruler: bool = True,
        return_fig: Literal[False] = False,
    ) -> None: ...

    def plot(
        self,
        lyrdb: pathlib.Path | str | None = None,
        display_type: Literal["image", "widget"] | None = None,
        *,
        show_labels: bool = True,
        show_ruler: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plots the Component using klayout.

        Args:
            lyrdb: path to layer properties file.
            display_type: if "image", displays the image.
            show_labels: if True, shows labels.
            show_ruler: if True, shows ruler.
            return_fig: if True, returns the figure.
        """
        from io import BytesIO

        import matplotlib.pyplot as plt

        from gdsfactory.pdk import get_layer_views

        self.insert_vinsts()

        lyp_path = GDSDIR_TEMP / "layer_properties.lyp"
        layer_views = get_layer_views()
        layer_views.to_lyp(filepath=lyp_path)

        layout_view = lay.LayoutView()
        cell_view_index = layout_view.create_layout(True)
        layout_view.active_cellview_index = cell_view_index
        cell_view = layout_view.cellview(cell_view_index)
        layout = cell_view.layout()
        layout.assign(kf.kcl.layout)

        assert self.name is not None, "Component name is None"

        cell_view.cell = layout.cell(self.name)

        layout_view.max_hier()
        layout_view.load_layer_props(str(lyp_path))

        layout_view.add_missing_layers()
        layout_view.zoom_fit()

        layout_view.set_config("text-visible", "true" if show_labels else "false")
        layout_view.set_config("grid-show-ruler", "true" if show_ruler else "false")

        pixel_buffer = layout_view.get_pixels_with_options(800, 600)
        png_data = pixel_buffer.to_png_data()

        # Convert PNG data to NumPy array and display with matplotlib
        with BytesIO(png_data) as f:
            img_array = plt.imread(f)

        # Compute the figure dimensions based on the image size and desired DPI
        dpi = 80
        fig_width = img_array.shape[1] / dpi
        fig_height = img_array.shape[0] / dpi

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Remove margins and display the image
        ax.imshow(img_array)
        ax.axis("off")  # Hide axes
        ax.set_position((0, 0, 1, 1))  # Set axes to occupy the full figure space

        plt.subplots_adjust(
            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
        )  # Remove any padding
        plt.tight_layout(pad=0)  # Ensure no space is wasted
        return fig if return_fig else None

    def plot_netlist(
        self,
        recursive: bool = False,
        with_labels: bool = True,
        font_weight: str = "normal",
        **kwargs: Any,
    ) -> nx.Graph:
        """Plots a netlist graph with networkx.

        Args:
            recursive: if True, returns a recursive netlist.
            with_labels: add label to each node.
            font_weight: normal, bold.
            kwargs: keyword arguments to get_netlist.

        Keyword Args:
            tolerance: tolerance in grid_factor to consider two ports connected.
            exclude_port_types: optional list of port types to exclude from netlisting.
            get_instance_name: function to get instance name.
            allow_multiple: False to raise an error if more than two ports share the same connection. \
                    if True, will return key: [value] pairs with [value] a list of all connected instances.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        from gdsfactory.get_netlist import nets_to_connections

        plt.figure()
        netlist = self.get_netlist(recursive=recursive, **kwargs)
        G = nx.Graph()

        if recursive:
            pos: dict[str, tuple[float, float]] = {}
            labels: dict[str, str] = {}
            for net in netlist.values():
                nets = net.get("nets", [])
                connections = net.get("connections", {})
                connections = nets_to_connections(nets, connections)
                placements = net["placements"]
                G.add_edges_from(
                    [
                        (",".join(k.split(",")[:-1]), ",".join(v.split(",")[:-1]))
                        for k, v in connections.items()
                    ]
                )
                pos |= {k: (v["x"], v["y"]) for k, v in placements.items()}
                labels |= {k: ",".join(k.split(",")[:1]) for k in placements.keys()}

        else:
            nets = netlist.get("nets", [])
            connections = netlist.get("connections", {})
            connections = nets_to_connections(nets, connections)
            placements = netlist["placements"]
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

    def plot_netlist_graphviz(
        self, recursive: bool = False, interactive: bool = False, splines: str = "ortho"
    ) -> None:
        """Plots a netlist graph with graphviz.

        Args:
            recursive: if True, returns a recursive netlist.
            interactive: if True, opens the graph in a browser.
            splines: ortho, spline, polyline, line, curved.
        """
        from gdsfactory.schematic import plot_graphviz

        n = self.to_graphviz(
            recursive=recursive,
        )
        plot_graphviz(n, splines=splines, interactive=interactive)

    def to_graphviz(self, recursive: bool = False) -> Digraph:
        """Returns a netlist graph with graphviz.

        Args:
            recursive: if True, returns a recursive netlist.
        """
        from gdsfactory.schematic import to_graphviz

        netlist = self.get_netlist(recursive=recursive)
        return to_graphviz(
            netlist["instances"],
            placements=netlist["placements"],
            nets=netlist["nets"],
        )


class ComponentAllAngle(ComponentBase, kf.VKCell):
    def plot(self, **kwargs: Any) -> None:
        """Plots the Component using klayout."""
        c = Component()
        if self.name is not None:
            c.name = self.name

        VInstance(self).insert_into_flat(c, levels=0)
        c.plot(**kwargs)

    def dup(self, new_name: str | None = None) -> Self:
        """Copy the full cell."""
        c = self.__class__(
            kcl=self.kcl, name=new_name or self.name + "$1" if self.name else None
        )
        c.ports = self.ports.copy()

        c.settings = self.settings.model_copy()
        c.settings_units = self.settings_units.model_copy()
        c.info = self.info.model_copy()
        for layer, shapes in self.shapes().items():
            for shape in shapes:
                c.shapes(layer).insert(shape)

        return c

    def add_polygon(self, points: _PolygonPoints, layer: "LayerSpec") -> None:
        """Adds a Polygon to the Component and returns a klayout Shape.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        if self.locked:
            raise LockedError(self)

        _layer = get_layer(layer)

        polygon = points_to_polygon(points)

        return self.shapes(_layer).insert(polygon)

    def get_polygons(self, layer: "LayerSpec") -> list[kf.kdb.DPolygon]:
        """Returns a list of polygons from the Component."""
        from gdsfactory import get_layer

        return [x for x in self.shapes(get_layer(layer)) if isinstance(x, kdb.DPolygon)]


def container(
    component: ComponentSpec,
    function: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Component:
    """Returns new component with a component reference.

    Args:
        component: to add to container.
        function: function to apply to component.
        kwargs: keyword arguments to pass to function.
    """
    import gdsfactory as gf

    component = gf.get_component(component)
    c = Component()
    cref = c << component
    c.add_ports(cref.ports)
    if function:
        function(component=c, **kwargs)

    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.generic_tech import LAYER

    c = gf.components.circle()
    c2 = gf.Component()
    region = c.get_region(layer=LAYER.WG, smooth=1)
    region2 = region.sized(100)
    region3 = region2 - region

    c2.add_polygon(region3, layer=LAYER.WG)
    c2.show()

    # polygons = c.get_polygons(smooth=1)[LAYER.WG]
    # c2.add_polygon(region, layer=LAYER.WG)
    # c2
