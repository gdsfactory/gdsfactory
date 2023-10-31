"""Component is a canvas for geometry."""
from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

import kfactory as kf
import numpy as np
from kfactory import Instance, kdb
from kfactory.kcell import save_layout_options

from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.port import pprint_ports, select_ports

if TYPE_CHECKING:
    from gdsfactory.typings import (
        CrossSection,
        Layer,
        LayerSpec,
        LayerStack,
        LayerViews,
        PathType,
    )

ComponentReference = Instance

ORPHANAGE = kf.KCell()


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

    def add_port(
        self,
        name: str,
        port: kf.Port | None = None,
        center: tuple[float, float] | kf.kdb.Point | None = None,
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

            layer = get_layer(layer)
            trans = kdb.DCplxTrans(1, orientation, False, center[0], center[1])
            self.create_port(
                name=name,
                dwidth=width,
                layer=layer,
                port_type=port_type,
                dcplx_trans=trans,
            )

    def add_polygon(
        self, points: np.ndarray | kdb.DPolygon | kdb.Polygon | Region, layer: LayerSpec
    ) -> kdb.DPolygon | kdb.Polygon | Region:
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        if isinstance(points, tuple | list | np.ndarray):
            polygon = kf.dpolygon_from_array(points)

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
            spacing: array-like[2] of int or float.
                Distance between adjacent columns and adjacent rows.

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

    def ref(self, *args, **kwargs) -> kdb.DCellInstArray:
        """Returns a Component Instance."""
        raise ValueError("ref() is deprecated. Use add_ref() instead")

    @classmethod
    def __get_validators__(cls):
        """Get validators for the Component object."""
        yield cls.validate

    @classmethod
    def validate(cls, v, _info) -> Component:
        """Pydantic assumes component is valid if the following are true.

        - name characters < pdk.cell_decorator_settings.max_name_length
        - is not empty (has references or polygons)
        """
        from gdsfactory.pdk import get_active_pdk

        pdk = get_active_pdk()

        max_name_length = pdk.cell_decorator_settings.max_name_length
        assert isinstance(
            v, Component
        ), f"TypeError, Got {type(v)}, expecting Component"
        assert (
            len(v.name) <= max_name_length
        ), f"name `{v.name}` {len(v.name)} > {max_name_length} "
        return v

    def show(self, **kwargs) -> None:
        """Shows the Component in Klayout.

        Args:
            **kwargs: extra arguments to pass to klayout.db.Database.show().
        """
        if kwargs:
            warnings.warn(
                f"{kwargs.keys()} is deprecated. Use the klayout extension to show ports"
            )
        super().show()

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

        if kwargs:
            warnings.warn(f"{kwargs.keys()} is deprecated")

        self.write(filename=str(gdspath), save_options=save_options)
        return gdspath

    @property
    def named_references(self):
        """Returns a dictionary of named references."""
        warnings.warn("named_references is deprecated. Use insts instead")
        return self.insts

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

    def remap_layers(self, layer_map: dict[LayerSpec, LayerSpec]) -> Component:
        """Remaps a list of layers and returns the same Component.

        Args:
            layer_map: dictionary of layers to remap.
        """
        from gdsfactory import get_layer

        for layer, new_layer in layer_map.items():
            layer_index = get_layer(layer)
            new_layer_index = get_layer(new_layer)
            for r in self._kdb_cell.begin_shapes_rec(layer_index):
                self.shapes(new_layer_index).insert(r)

            self.remove_layers([layer_index], recursive=True)
        return self

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


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.c.straight()
    print(c.get_ports_list(prefix="o"))

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

    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=(1, 0))
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
    c.show()
