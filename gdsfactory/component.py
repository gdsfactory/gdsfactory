"""Component is a canvas for geometry."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import kfactory as kf
import numpy as np
from kfactory import kdb

from gdsfactory.port import select_ports

if TYPE_CHECKING:
    from gdsfactory.typings import CrossSection, LayerSpec

ComponentReference = kf.Instance


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
        center: tuple[float, float] | None = None,
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

            layer = get_layer(layer)

            self.create_port(
                name=name,
                position=(
                    center[0] / self.kcl.dbu,
                    center[1] / self.kcl.dbu,
                ),
                width=int(width / self.kcl.dbu),
                angle=int(orientation // 90),
                layer=layer,
                port_type=port_type,
            )

    def add_polygon(self, points: np.ndarray | kdb.Polygon, layer: LayerSpec):
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        from gdsfactory.pdk import get_layer

        layer = get_layer(layer)

        if not isinstance(points, kdb.DPolygon):
            points = kdb.DPolygon([kdb.DPoint(point[0], point[1]) for point in points])

        self.shapes(layer).insert(points)

    def add_label(
        self,
        text: str = "hello",
        position: tuple[float, float] = (0.0, 0.0),
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
        x, y = position
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
        return list(select_ports(self.ports, **kwargs).values())

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
            **kwargs: extra information to add to the component.
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

    def absorb(self, reference) -> Component:
        """Absorbs polygons from ComponentReference into Component.

        Destroys the reference in the process but keeping the polygon geometry.

        Args:
            reference: ComponentReference to be absorbed into the Component.
        """
        if reference not in self.insts:
            raise ValueError(
                "The reference you asked to absorb does not exist in this Component."
            )
        reference.flatten()
        return self

    def add_ref(self, component: Component, **kwargs) -> kf.Instance:
        return self.create_inst(component)

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

    def auto_rename_ports(self) -> None:
        self.autorename_ports()


if __name__ == "__main__":
    c = Component()
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=(1, 0))
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer="SLAB150")
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=LAYER.WG)
    # c.create_port(name="o1", position=(10, 10), angle=1, layer=LAYER.WG, width=2000)
    # c.add_port(name="o1", center=(0, 0), orientation=270, layer=LAYER.WG, width=2.0)
    # c.add_label(text="hello", position=(2, 2), layer=LAYER.TEXT)

    import gdsfactory as gf

    P = gf.path.straight(length=10)
    s0 = gf.Section(
        width=1, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2")
    )
    s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
    x1 = gf.CrossSection(sections=(s0, s1))
    c1 = gf.path.extrude(P, x1)
    ref = c.add_ref(c1)
    c.add_ports(ref.ports)
    c.show()
