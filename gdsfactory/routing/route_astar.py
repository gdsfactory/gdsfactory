from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import klayout.dbcore as kdb
import networkx as nx
import numpy as np
import numpy.typing as npt
from klayout.dbcore import DPoint
from shapely.geometry import LineString

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import (
    ComponentSpec,
    Coordinate,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    Port,
    Route,
)


class Node:
    def __init__(
        self, parent: Node | None = None, position: tuple[int, int] = (0, 0)
    ) -> None:
        """Initializes a node. A node is a point on the grid."""
        self.parent = parent  # parent node of current node
        self.position = position  # position of current node

        self.g = 0  # distance between current node and start node
        self.h = 0  # distance between current node and end node
        self.f = self.g + self.h  # cost of the node (sum of g and h)


def _extract_all_bbox(
    c: Component, avoid_layers: Sequence[LayerSpec] | None = None
) -> list[dict[tuple[int, int] | str | int, list[kdb.Polygon]]]:
    """Extract all polygons whose layer is in `avoid_layers`.

    Args:
        c: Component to extract polygons from.
        avoid_layers: List of layers to avoid.

    """
    return [c.get_polygons(by="name", layers=avoid_layers)]


def _parse_bbox_to_array(bbox: kdb.DBox | kdb.Polygon) -> npt.NDArray[np.floating[Any]]:
    """Parses bbox in the form of (a,b;c,d) to [[a, b], [c, d]].

    Args:
        bbox: Parses bbox in the form of (a,b;c,d).

    """
    bbox_str = str(bbox).strip("()")
    rows = bbox_str.split(";")
    bbox_values = [list(map(float, row.split(","))) for row in rows]
    return np.array(bbox_values, dtype=np.float64)


def _generate_grid(
    c: Component,
    resolution: float = 0.5,
    avoid_layers: Sequence[LayerSpec] | None = None,
    distance: float = 1,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.integer[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Generate discretization grid that the algorithm will step through.

    Args:
        c: Component to route through.
        resolution: Discretization resolution in um.
        avoid_layers: List of layers to avoid.
        distance: Distance from obstacles in um.
    """
    bbox_int = _parse_bbox_to_array(c.dbbox())
    bbox = bbox_int

    _a1 = float(bbox[0][0]) - resolution
    _a2 = float(bbox[1][0]) + resolution
    _b1 = float(bbox[0][1]) - resolution
    _b2 = float(bbox[1][1]) + resolution

    _a = np.linspace(_a1, _a2, int((_a2 - _a1) / resolution), endpoint=True)

    _b = np.linspace(_b1, _b2, int((_b2 - _b1) / resolution), endpoint=True)

    x, y = np.meshgrid(_a, _b)  # discretize component space
    x, y = x[0], y[:, 0]  # weed out copies
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    grid = np.zeros(
        (len(x), len(y))
    )  # mapping from gdsfactory's x-, y- coordinate to grid vertex

    # assign 1 for obstacles
    if avoid_layers is None:
        for inst in c.insts:
            bbox_array = _parse_bbox_to_array(inst.bbox())
            xmin = np.abs(x - bbox_array[0][0] + distance).argmin()
            xmax = np.abs(x - bbox_array[1][0] - distance).argmin()
            ymin = np.abs(y - bbox_array[0][1] + distance).argmin()
            ymax = np.abs(y - bbox_array[1][1] - distance).argmin()
            grid[xmin:xmax, ymin:ymax] = 1
    else:
        all_refs = _extract_all_bbox(c, avoid_layers)
        for layer in all_refs:
            for polygons in layer.values():
                for polygon in polygons:
                    bbox_array = _parse_bbox_to_array(polygon)
                    bbox_array_float = bbox_array / 1000
                    xmin = np.abs(x - bbox_array_float[0][0] + distance).argmin()
                    xmax = np.abs(x - bbox_array_float[2][0] - distance).argmin()
                    ymin = np.abs(y - bbox_array_float[0][1] + distance).argmin()
                    ymax = np.abs(y - bbox_array_float[2][1] - distance).argmin()
                    grid[xmin:xmax, ymin:ymax] = 1

    return np.round(grid, 3), np.round(x, 3), np.round(y, 3)


def simplify_path(waypoints: Coordinates, tolerance: float) -> list[Coordinate]:
    """Simplifies a list of waypoints using the Douglas-Peucker algorithm.

    Args:
        waypoints: List of waypoints as coordinate pairs (x, y).
        tolerance: Simplification tolerance.

    Returns:
        List of simplified waypoints.
    """
    line = LineString(waypoints)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return list(simplified_line.coords)


def route_astar(
    component: Component,
    port1: Port,
    port2: Port,
    resolution: float = 1,
    avoid_layers: Sequence[LayerSpec] | None = None,
    distance: float = 8,
    cross_section: CrossSectionSpec = "strip",
    bend: ComponentSpec = "wire_corner",
    **kwargs: Any,
) -> Route:
    """Bidirectional routing function using NetworkX. Finds a route between two ports avoiding obstacles.

    Args:
        component: Component the route and ports belong to.
        port1: Input port.
        port2: Output port.
        resolution: Discretization resolution in um.
        avoid_layers: List of layers to avoid.
        distance: Distance from obstacles in um.
        cross_section: Cross-section specification.
        bend: Component to use for bends. Use wire_corner for Manhattan routing or bend_euler for Euler routing.
        kwargs: cross-section settings.
    """
    cross_section = gf.get_cross_section(cross_section, **kwargs)
    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)
    G_ = nx.grid_2d_graph(len(x), len(y))
    G = cast(nx.Graph, G_)

    # Remove nodes representing obstacles
    for i in range(len(x)):
        for j in range(len(y)):
            if grid[i, j] == 1:
                G.remove_node((i, j))

    # Unit conversion
    port1x = port1.x
    port1y = port1.y
    port2x = port2.x
    port2y = port2.y

    # Define start and end nodes
    start_node = (
        round((port1x - x.min()) / resolution),
        (round((port1y - y.min()) / resolution)),
    )
    end_node = (
        (round((port2x - x.min()) / resolution)),
        (round((port2y - y.min()) / resolution)),
    )

    # Find the closest valid nodes
    start_node = min(
        G.nodes,
        key=lambda node: np.linalg.norm(np.array(node) - np.array(start_node)),  # type: ignore[arg-type, return-value]
    )
    end_node = min(
        G.nodes,
        key=lambda node: np.linalg.norm(np.array(node) - np.array(end_node)),  # type: ignore[arg-type, return-value]
    )

    path = nx.astar_path(G, start_node, end_node)  # Find shortest path

    # Convert path to waypoints
    waypoints = [(x[i] + resolution / 2, y[j] + resolution / 2) for i, j in path]

    # Simplify the route
    simplified_path = simplify_path([(port1x, port1y)] + waypoints, tolerance=0.05)

    # Prepare waypoints
    my_waypoints = [list(np.round(pt, 1)) for pt in simplified_path]
    if port2.orientation in [0, 180]:
        my_waypoints += [[my_waypoints[-1][0], port2y]]
    else:
        my_waypoints += [[port2x, my_waypoints[-1][1]]]

    # Align second waypoint y with first waypoint y
    # my_waypoints[1][1] = my_waypoints[0][1]

    if port1.orientation in [0, 180]:
        my_waypoints[0:1] = [my_waypoints[0], [my_waypoints[0][0], my_waypoints[1][1]]]
    else:
        my_waypoints[0:1] = [my_waypoints[0], [my_waypoints[1][0], my_waypoints[0][1]]]
    if port2.orientation in [0, 180]:
        my_waypoints.append([port2x, my_waypoints[-1][1]])
    else:
        my_waypoints.append([my_waypoints[-1][0], port2y])
    my_waypoints += [[port2x, port2y]]

    # Convert to native floats or Point instances
    waypoints_ = [DPoint(x, y) for x, y in my_waypoints]
    return gf.routing.route_single(
        component=component,
        port1=port1,
        port2=port2,
        waypoints=gf.kf.routing.manhattan.clean_points(waypoints_),
        cross_section=cross_section,
        bend=bend,
    )


if __name__ == "__main__":
    # cross_section = "xs_metal_routing"
    # port_prefix = "e"
    # bend = gf.components.wire_corner

    c = gf.Component()
    cross_section_name = "strip"
    port_prefix = "o"
    bend = gf.components.bend_euler

    cross_section = gf.get_cross_section(cross_section_name, radius=5)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90)
    right.move((168, 63))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M2")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle4.rotate(90)
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    route = route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=15,
        distance=12,
        avoid_layers=("M2",),
        bend=bend,
    )
    c.show()
