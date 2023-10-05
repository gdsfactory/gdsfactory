"""Component is a canvas for geometry."""
from __future__ import annotations

from typing import TYPE_CHECKING

import kfactory as kf
import numpy as np
from kfactory import kdb

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
        l1, l2 = layer
        x, y = position
        trans = kdb.DTrans(0, False, x, y)
        return self.shapes(self.kcl.layer(l1, l2)).insert(kf.kdb.DText(text, trans))

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


if __name__ == "__main__":
    from gdsfactory.generic_tech import LAYER

    c = Component()
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=(1, 0))
    # c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer="SLAB150")
    c.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)], layer=LAYER.WG)
    # c.create_port(name="o1", position=(10, 10), angle=1, layer=LAYER.WG, width=2000)
    c.add_port(name="o1", center=(0, 0), orientation=270, layer=LAYER.WG, width=2.0)
    c.add_label(text="hello", position=(2, 2), layer=LAYER.TEXT)
    c.show()
