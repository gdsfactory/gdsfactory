"""Component is a canvas for geometry."""

from __future__ import annotations

import pathlib
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import kfactory as kf
import numpy as np
from kfactory import Instance, kdb
from kfactory.kcell import cell, save_layout_options

from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.port import pprint_ports, select_ports, to_dict
from gdsfactory.serialization import clean_value_json

if TYPE_CHECKING:
    from gdsfactory.typings import (
        CrossSection,
        Layer,
        LayerSpec,
        LayerStack,
        LayerViews,
        PathType,
    )

cell_without_validator = cell
ComponentReference = Instance


def ensure_tuple_of_tuples(points) -> tuple[tuple[float, float]]:
    # Convert a single NumPy array to a tuple of tuples
    if isinstance(points, np.ndarray):
        points = tuple(map(tuple, points.tolist()))
    elif isinstance(points, list):
        # If it's a list, check if the first element is an np.ndarray or a list to decide on conversion
        if len(points) > 0 and isinstance(points[0], np.ndarray | list):
            points = tuple(
                tuple(point) if isinstance(point, np.ndarray) else tuple(point)
                for point in points
            )
    return points


def size(region: kdb.Region, offset: float, dbu=1e3) -> kdb.Region:
    return region.dup().size(int(offset * dbu))


def boolean_or(region1: kdb.Region, region2: kdb.Region) -> kdb.Region:
    return region1.__or__(region2)


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
    "-": boolean_or,
    "^": boolean_xor,
    "xor": boolean_xor,
    "&": boolean_and,
    "and": boolean_and,
}


def copy(region: kdb.Region) -> kdb.Region:
    return region.dup()


class Region(kdb.Region):
    def __iadd__(self, offset) -> kdb.Region:
        """Adds an offset to the layer."""
        return size(self, offset)

    def __isub__(self, offset) -> kdb.Region:
        """Adds an offset to the layer."""
        return size(self, -offset)

    def __add__(self, element) -> kdb.Region:
        if isinstance(element, float | int):
            return size(self, element)

        elif isinstance(element, kdb.Region):
            return boolean_or(self, element)
        else:
            raise ValueError(f"Cannot add type {type(element)} to region")

    def __sub__(self, element) -> kdb.Region | None:
        if isinstance(element, float | int):
            return size(self, -element)

        elif isinstance(element, kdb.Region):
            return boolean_not(self, element)

    def copy(self) -> kdb.Region:
        return self.dup()


class Component(kf.KCell):
    """A Component is an empty canvas where you add polygons, instances and ports \
            (to connect to other components).

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
    def layers(self) -> list[tuple[int, int]]:
        return list(self.get_polygons().keys())

    def add_port(  # type: ignore[override]
        self,
        name: str | None = None,
        port: kf.Port | None = None,
        center: tuple[float, float] | kf.kdb.DPoint | None = None,
        width: float | None = None,
        orientation: float | None = None,
        layer: LayerSpec | None = None,
        port_type: str = "optical",
        cross_section: CrossSection | None = None,
    ) -> kf.Port:
        """Adds a Port to the Component.

        Args:
            name: name of the port.
            port: port to add.
            center: center of the port.
            width: width of the port.
            orientation: orientation of the port.
            layer: layer spec to add port on.
            port_type: port type (optical, electrical, ...)
            cross_section: cross_section of the port.
        """
        if port:
            kf.KCell.add_port(self, port=port, name=name)
            return port
        else:
            from gdsfactory.pdk import get_cross_section, get_layer

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

            if orientation is None:
                raise ValueError("Must specify orientation")

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

            return self.create_port(
                name=name,
                dwidth=round(width / self.kcl.dbu) * self.kcl.dbu,
                layer=layer,
                port_type=port_type,
                dcplx_trans=trans,
            )

    def from_kcell(self) -> Component:
        """Returns a Component from a KCell."""
        kdb_copy = self._kdb_copy()

        c = Component(kcl=self.kcl, kdb_cell=kdb_copy)
        c.ports = self.ports.copy()

        c._settings = self.settings.model_copy()
        c.info = self.info.model_copy()
        return c

    def copy(self) -> Component:
        return self.dup()

    def dup(self) -> Component:
        """Copy the full cell.

        Sets `_locked` to `False`

        Returns:
            cell: Exact copy of the current cell.
                The name will have `$1` as duplicate names are not allowed
        """
        kdb_copy = self._kdb_copy()

        c = Component(kcl=self.kcl, kdb_cell=kdb_copy)
        c.ports = self.ports.copy()

        c._settings = self.settings.model_copy()
        c.info = self.info.model_copy()
        return c

    def add_polygon(
        self,
        points: np.ndarray | kdb.DPolygon | kdb.Polygon | Region | list[list[float]],
        layer: LayerSpec,
    ) -> kdb.DPolygon | kdb.Polygon | Region:
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)
        if len(points) == 2:
            points = tuple(zip(points[0], points[1]))

        if isinstance(points, tuple | list | np.ndarray):
            points = ensure_tuple_of_tuples(points)
            polygon = kf.kdb.DPolygon()
            polygon.assign_hull(points)

        elif isinstance(
            points, kdb.Polygon | kdb.DPolygon | kdb.DSimplePolygon | kdb.Region
        ):
            polygon = points

        self.shapes(layer).insert(polygon)

        if not isinstance(polygon, kdb.Region):
            return Region(polygon.to_itype(self.kcl.dbu))
        else:
            return polygon

    def add_label(
        self,
        text: str = "hello",
        position: tuple[float, float] | kf.kdb.DPoint = (0.0, 0.0),
        layer: LayerSpec = "TEXT",
    ) -> kdb.Shape:
        """Adds Label to the Component.

        Args:
            text: Label text.
            position: x-, y-coordinates of the Label location.
            layer: Specific layer(s) to put Label on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)
        if isinstance(position, kf.kdb.DPoint):
            x, y = position.x, position.y

        elif isinstance(position, tuple | list):
            x, y = position

        else:
            raise ValueError(f"position {position} not supported")
        trans = kdb.DTrans(0, False, x, y)
        return self.shapes(layer).insert(kf.kdb.DText(text, trans))

    def add_array(
        self,
        component: Component,
        columns: int = 2,
        rows: int = 2,
        spacing: tuple[float, float] = (100, 100),
    ) -> ComponentReference:
        """Creates a ComponentReference reference to a Component.

        Args:
            component: The referenced component.
            columns: Number of columns in the array.
            rows: Number of rows in the array.
            spacing: x, y distance between adjacent columns and adjacent rows.

        """
        if not isinstance(component, Component):
            raise TypeError("add_array() needs a Component object.")

        return self.create_inst(
            component,
            na=columns,
            nb=rows,
            a=kf.kdb.Vector(spacing[0] / self.kcl.dbu, 0),
            b=kf.kdb.Vector(0, spacing[1] / self.kcl.dbu),
        )

    def get_ports_list(self, **kwargs) -> list[kf.Port]:
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
        return select_ports(self.ports, **kwargs)

    def add_route_info(
        self,
        cross_section: CrossSection | str,
        length: float,
        length_eff: float | None = None,
        taper: bool = False,
        **kwargs,
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

    def absorb(self, reference: Instance) -> Component:
        """Absorbs polygons from ComponentReference into Component.

        Destroys the reference in the process but keeping the polygon geometry.

        Args:
            reference: Instance to be absorbed into the Component.

        """
        if reference not in self.insts:
            raise ValueError(
                "The reference you asked to absorb does not exist in this Component."
            )
        reference.flatten()
        return self

    def add_ref(
        self, component: Component, name: str | None = None, alias: str | None = None
    ) -> kf.Instance:
        inst = self.create_inst(component)
        if alias:
            warnings.warn("alias is deprecated, use name instead")
            inst.name = alias
        elif name:
            inst.name = name
        return inst

    def add(self, instances: list[Instance] | Instance) -> None:
        if not hasattr(instances, "__iter__"):
            instances = [instances]

        for instance in instances:
            self._kdb_cell.insert(instance._instance)

    def get_polygons(
        self, merge: bool = False
    ) -> dict[tuple[int, int], list[kf.kdb.Polygon]]:
        """Returns a dict of Polygons per layer.

        Args:
            merge: if True, merges the polygons.
        """
        from gdsfactory import get_layer

        polygons = defaultdict(list)

        for layer in self.kcl.layers:
            layer_index = get_layer(layer)
            r = kdb.Region(self.begin_shapes_rec(layer_index))
            if merge:
                r.merge()
            for p in r.each():
                layer_tuple = (layer.layer, layer.datatype)
                polygons[layer_tuple].append(p)
        return polygons

    def get_polygons_points(
        self, merge: bool = False, scale: float | None = None
    ) -> dict[tuple[int, int], list[tuple[float, float]]]:
        """Returns a dict with list of points per layer.

        Args:
            merge: if True, merges the polygons.
        """
        polygons_dict = self.get_polygons(merge=merge)
        polygons_points = {}
        for layer_tuple, polygons in polygons_dict.items():
            all_points = []
            for polygon in polygons:
                if scale:
                    points = [
                        (point.x * scale, point.y * scale)
                        for point in polygon.to_simple_polygon()
                        .to_dtype(self.kcl.dbu)
                        .each_point()
                    ]
                else:
                    points = [
                        (point.x, point.y)
                        for point in polygon.to_simple_polygon()
                        .to_dtype(self.kcl.dbu)
                        .each_point()
                    ]
                all_points.append(points)
            polygons_points[layer_tuple] = all_points
        return polygons_points

    def area(self, layer: LayerSpec) -> float:
        """Returns the area of the Component in um2."""
        from gdsfactory import get_layer

        layer_index = get_layer(layer)
        r = kdb.Region(self.begin_shapes_rec(layer_index))
        r.merge()
        return sum(p.area2() / 2 * self.kcl.dbu**2 for p in r.each())

    @classmethod
    def __get_validators__(cls):
        """Get validators for the Component object."""
        yield cls.validate

    @classmethod
    def validate(cls, v, _info) -> Component:
        """Pydantic assumes component is valid if the following are true.

        - is not empty (has references or polygons)
        """
        from gdsfactory.pdk import get_active_pdk

        pdk = get_active_pdk()

        max_name_length = pdk.max_name_length
        assert isinstance(
            v, Component
        ), f"TypeError, Got {type(v)}, expecting Component"
        assert (
            len(v.name) <= max_name_length
        ), f"name `{v.name}` {len(v.name)} > {max_name_length} "
        return v

    def copy_child_info(self, component: Component) -> None:
        """Copy and settings info from child component into parent.

        Parent components can access child cells settings.
        """
        info = dict(component.info)

        for k, v in info.items():
            if k not in self.info:
                self.info[k] = v

    def write_gds(
        self,
        gdspath: PathType | None = None,
        gdsdir: PathType | None = None,
        save_options: kdb.SaveLayoutOptions = save_layout_options(),
        **kwargs,
    ) -> pathlib.Path:
        """Write component to GDS and returns gdspath.

        Args:
            gdspath: GDS file path to write to.
            gdsdir: directory for the GDS file. Defaults to /tmp/randomFile/gdsfactory.
            save_options: klayout save options.
            kwargs: deprecated.
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
        gdspath = gdspath or gdsdir / f"{self.name}.gds"
        gdspath = pathlib.Path(gdspath)

        if not gdspath.parent.is_dir():
            gdspath.parent.mkdir(parents=True, exist_ok=True)

        if kwargs:
            for k in kwargs:
                warnings.warn(f"{k} is deprecated", stacklevel=2)
        self.write(filename=gdspath, save_options=save_options)
        return pathlib.Path(gdspath)

    def extract(
        self,
        layers: list[LayerSpec],
    ) -> Component:
        """Extracts a list of layers and adds them to a new Component.

        Args:
            layers: list of layers to extract.
        """
        from gdsfactory import get_layer

        c = Component()

        for layer in layers:
            layer_index = get_layer(layer)
            for r in self._kdb_cell.begin_shapes_rec(layer_index):
                c.shapes(layer_index).insert(r)

        return c

    def remove_layers(
        self,
        layers: list[LayerSpec],
        recursive: bool = True,
    ) -> Component:
        """Removes a list of layers and returns the same Component.

        Args:
            layers: list of layers to remove.
            recursive: if True, removes layers recursively.
        """
        from gdsfactory import get_layer

        layers = [get_layer(layer) for layer in layers]
        for layer_index in layers:
            if recursive:
                self.kcl.clear_layer(layer_index)
            else:
                self.shapes(layer_index).clear()
        return self

    def remap_layers(
        self, layer_map: dict[LayerSpec, LayerSpec], recursive: bool = False
    ) -> Component:
        """Remaps a list of layers and returns the same Component.

        Args:
            layer_map: dictionary of layers to remap.
            recursive: if True, remaps layers recursively.
        """
        from gdsfactory import get_layer

        for layer, new_layer in layer_map.items():
            src_layer_index = get_layer(layer)
            dst_layer_index = get_layer(new_layer)
            self.move(src_layer_index, dst_layer_index)

            if recursive:
                for ci in self.called_cells():
                    self.kcl[ci].move(src_layer_index, dst_layer_index)

    def pprint_ports(self, **kwargs) -> None:
        """Pretty prints ports.

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
        pprint_ports(ports)

    def to_3d(
        self,
        layer_views: LayerViews | None = None,
        layer_stack: LayerStack | None = None,
        exclude_layers: tuple[Layer, ...] | None = None,
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

    def get_netlist(self) -> dict[str, Any]:
        """Returns a netlist for circuit simulation."""
        from gdsfactory.get_netlist import get_netlist

        return get_netlist(self)

    def plot_netlist(
        self, with_labels: bool = True, font_weight: str = "normal", **kwargs
    ):
        """Plots a netlist graph with networkx.

        Args:
            with_labels: add label to each node.
            font_weight: normal, bold.

        Keyword Args:
            tolerance: tolerance in grid_factor to consider two ports connected.
            exclude_port_types: optional list of port types to exclude from netlisting.
            get_instance_name: function to get instance name.
            allow_multiple: False to raise an error if more than two ports share the same connection. \
                    if True, will return key: [value] pairs with [value] a list of all connected instances.
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

    def over_under(self, layer: LayerSpec, distance: float = 1.0) -> None:
        """Flattens and performs over-under on a layer in the Component.
        For big components use tiled version.

        Args:
            layer: layer to perform over-under on.
            distance: distance to perform over-under in um.
        """
        from gdsfactory import get_layer

        self.flatten()
        layer = get_layer(layer)
        region = kdb.Region(self.shapes(layer))
        region.size(+distance).size(-distance)
        self.shapes(layer).clear()
        self.shapes(layer).insert(region)

    def to_dict(self, with_ports: bool = False) -> dict[str, Any]:
        """Returns a dictionary representation of the Component."""
        d = clean_value_json(
            {
                "name": self.name,
                "info": self.info.model_dump(exclude_none=True),
                "settings": self.settings.model_dump(exclude_none=True),
            }
        )
        if with_ports:
            d["ports"] = {port.name: to_dict(port) for port in self.ports}
        return d

    def plot(
        self,
        show_labels: bool = False,
        show_ruler: bool = True,
        return_fig: bool = False,
    ):
        """Plots the Component using klayout.

        Args:
            show_labels: if True, shows labels.
            show_ruler: if True, shows ruler.
            return_fig: if True, returns the figure.

        """
        from io import BytesIO

        import klayout.db as db  # noqa: F401
        import klayout.lay as lay
        import matplotlib.pyplot as plt

        from gdsfactory.pdk import get_layer_views

        gdspath = self.write_gds()
        lyp_path = gdspath.with_suffix(".lyp")

        layer_views = get_layer_views()
        layer_views.to_lyp(filepath=lyp_path)

        layout_view = lay.LayoutView()
        layout_view.load_layout(str(gdspath.absolute()))
        layout_view.max_hier()
        layout_view.load_layer_props(str(lyp_path))

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
        ax.set_position([0, 0, 1, 1])  # Set axes to occupy the full figure space

        plt.subplots_adjust(
            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
        )  # Remove any padding
        plt.tight_layout(pad=0)  # Ensure no space is wasted
        if return_fig:
            return fig

    # Deprecated methods
    @property
    def named_references(self):
        """Returns a dictionary of named references."""
        warnings.warn("named_references is deprecated. Use insts instead")
        return self.insts

    @property
    def references(self) -> list[ComponentReference]:
        """Returns a list of references."""
        warnings.warn("references is deprecated. Use insts instead")
        return list(self.insts)

    def ref(self, *args, **kwargs) -> kdb.DCellInstArray:
        """Returns a Component Instance."""
        raise ValueError("ref() is deprecated. Use add_ref() instead")


@kf.cell
def container(component, function, **kwargs) -> Component:
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
    function(c, **kwargs)
    c.ports = cref.ports
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.Component()
    # wg1 = c << gf.c.straight(length=10, cross_section="xs_rc")
    # wg2 = c << gf.c.straight(length=5, cross_section="xs_rc")
    # wg2.connect("o1", wg1["o2"])
    # wg2.d.movex(5)
    # p = c.get_polygons()
    # print(c.area(layer=(1, 0)))
    # print(c.get_ports_list(prefix="o"))

    # c = gf.Component()
    # b1 = gf.components.circle(radius=10)
    # b2 = gf.components.circle(radius=11)

    # ref = c << gf.c.bend_euler(cross_section="xs_rc")
    # c.add_ports(ref.ports)
    # p = c.get_ports_list(sort_ports=True)
    # print(c.get_ports_list(sort_ports=True))

    # c = gf.Component()
    # text = c << gf.components.text("hello")
    # text.mirror(
    #     p1=kf.kdb.Point(1, 1), p2=gf.kdb.Point(1, 3)
    # )  # Reflects across the line formed by p1 and p2
    # c.remap_layers({(1, 0): (2, 0)})
    # c.show()

    # c.add_polygon([(0, 0), (1, 1), (1, 3)], layer=(1, 0))
    # c = c.remove_layers(layers=[(1, 0), (2, 0)], recursive=True)
    # c = c.extract(layers=[(1, 0)])

    # c = Component()
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=(1, 0))
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer="SLAB150")
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=LAYER.WG)
    # c.create_port(name="o1", position=(10, 10), angle=1, layer=LAYER.WG, width=2000)
    # c.add_port(name="o1", center=(0, 0), orientation=270, layer=LAYER.WG, width=2.0)
    # c.add_label(text="hello", position=(2, 2), layer=LAYER.TEXT)
    # p = c.add_polygon(np.array(list(zip((-8, 6, 7, 9), (-6, 8, 17, 5)))), layer=(1, 0))

    # p = c.add_polygon(list(zip((-8, 6, 7, 9), (-6, 8, 17, 5))), layer=(1, 0))
    # p2 = p + 2
    # p2 = c.add_polygon(p2, layer=(1, 0))

    # p3 = p2 - p
    # p3 = c.add_polygon(p3, layer=(2, 0))

    # P = gf.path.straight(length=10)
    # s0 = gf.Section(
    #     width=1, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2")
    # )
    # s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
    # x1 = gf.CrossSection(sections=(s0, s1))
    # c1 = gf.path.extrude(P, x1)
    # ref = c.add_ref(c1)
    # c.add_ports(ref.ports)
    # scene = c.to_3d()
    # scene.show()

    c = gf.c.straight()
    # print(c.to_dict())
    # print(c.area(layer=(1, 0)))
    # stl = gf.export.to_stl(c)
    c.show()
