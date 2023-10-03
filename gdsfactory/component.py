"""Component is a canvas for geometry.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import kfactory as kf
import numpy as np
from kfactory import kdb

if TYPE_CHECKING:
    from gdsfactory.typings import (
        LayerSpec,
    )


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

    def add_polygon(self, points: np.ndarray | kdb.Polygon, layer: LayerSpec):
        """Adds a Polygon to the Component.

        Args:
            points: Coordinates of the vertices of the Polygon.
            layer: layer spec to add polygon on.
        """
        # from gdsfactory.pdk import get_layer

        # layer = get_layer(layer)

        if not isinstance(points, kdb.DPolygon):
            points = kdb.DPolygon([kdb.DPoint(point[0], point[1]) for point in points])

        self.shapes(self.kcl.layer(layer[0], layer[1])).insert(points)


if __name__ == "__main__":
    c = Component()

    p = c.add_polygon(
        [(-8, 6, 7, 9), (-6, 8, 17, 5)], layer=(1, 0)
    )  # GDS layers are tuples of ints (but if we use only one number it assumes the other number is 0)
    # c.write_gds("hi.gds")
    c.show()
    # print(CONF.last_saved_files)
