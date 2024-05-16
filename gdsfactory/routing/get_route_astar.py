from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, LayerSpec, Route


class Node:
    def __init__(self, parent=None, position: tuple = ()) -> None:
        """Initializes a node. A node is a point on the grid."""
        self.parent = parent  # parent node of current node
        self.position = position  # position of current node

        self.g = 0  # distance between current node and start node
        self.h = 0  # distance between current node and end node
        self.f = self.g + self.h  # cost of the node (sum of g and h)


def _extract_all_bbox(c: Component, avoid_layers: list[LayerSpec] | None = None):
    """Extract all polygons whose layer is in `avoid_layers`.

    Args:
        c: Component to extract polygons from.
        avoid_layers: List of layers to avoid.

    """
    return [c.get_polygons(layer) for layer in avoid_layers]


def _generate_grid(
    c: Component,
    resolution: float = 0.5,
    avoid_layers: list[LayerSpec] | None = None,
    distance: float = 1,
) -> np.ndarray:
    """Generate discretization grid that the algorithm will step through.

    Args:
        c: Component to route through.
        resolution: Discretization resolution in um.
        avoid_layers: List of layers to avoid.
        distance: Distance from obstacles in um.
    """
    bbox = c.bbox
    x, y = np.meshgrid(
        np.linspace(
            bbox[0][0] - resolution,
            bbox[1][0] + resolution,
            int((bbox[1][0] - bbox[0][0] + 2 * resolution) / resolution),
            endpoint=True,
        ),
        np.linspace(
            bbox[0][1] - resolution,
            bbox[1][1] + resolution,
            int((bbox[1][1] - bbox[0][1] + 2 * resolution) / resolution),
            endpoint=True,
        ),
    )  # discretize component space
    x, y = x[0], y[:, 0]  # weed out copies
    grid = np.zeros(
        (len(x), len(y))
    )  # mapping from gdsfactory's x-, y- coordinate to grid vertex

    # assign 1 for obstacles
    if avoid_layers is None:
        for ref in c.references:
            bbox = ref.bbox
            xmin = np.abs(x - bbox[0][0] + distance).argmin()
            xmax = np.abs(x - bbox[1][0] - distance).argmin()
            ymin = np.abs(y - bbox[0][1] + distance).argmin()
            ymax = np.abs(y - bbox[1][1] - distance).argmin()

            grid[xmin:xmax, ymin:ymax] = 1
    else:
        all_refs = _extract_all_bbox(c, avoid_layers)
        for layer in all_refs:
            for bbox in layer:
                xmin = np.abs(x - bbox[0][0] + distance).argmin()
                xmax = np.abs(x - bbox[2][0] - distance).argmin()
                ymin = np.abs(y - bbox[0][1] + distance).argmin()
                ymax = np.abs(y - bbox[2][1] - distance).argmin()
                grid[xmin:xmax, ymin:ymax] = 1

    return np.ndarray.round(grid, 3), np.ndarray.round(x, 3), np.ndarray.round(y, 3)


def simplify_path(waypoints, tolerance):
    """
    Simplifies a list of waypoints using the Douglas-Peucker algorithm.

    Args:
        waypoints: List of waypoints as coordinate pairs (x, y).
        tolerance: Simplification tolerance.

    Returns:
        List of simplified waypoints.
    """
    line = LineString(waypoints)  # Create a line from waypoints
    simplified_line = line.simplify(
        tolerance, preserve_topology=False
    )  # Simplify the line
    return list(
        simplified_line.coords
    )  # Convert simplified line back to a list of waypoints


def get_route_astar(
    component: Component,
    port1: Port,
    port2: Port,
    resolution: float = 1,
    avoid_layers: list[LayerSpec] | None = None,
    distance: float = 8,
    cross_section: CrossSectionSpec = "xs_sc",
    bend: gf.Component = gf.components.wire_corner,
    **kwargs,
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
        kwargs: Cross-section settings.
    """
    cross_section = gf.get_cross_section(cross_section, **kwargs)
    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)

    G = nx.grid_2d_graph(len(x), len(y))  # Create graph

    # Remove nodes representing obstacles
    for i in range(len(x)):
        for j in range(len(y)):
            if grid[i, j] == 1:
                G.remove_node((i, j))

    # Define start and end nodes
    start_node = (
        int(round((port1.x - x.min()) / resolution)),
        int(round((port1.y - y.min()) / resolution)),
    )
    end_node = (
        int(round((port2.x - x.min()) / resolution)),
        int(round((port2.y - y.min()) / resolution)),
    )

    # Find the closest valid nodes
    start_node = min(
        G.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(start_node))
    )
    end_node = min(
        G.nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(end_node))
    )

    path = nx.astar_path(G, start_node, end_node)  # Find shortest path

    # Convert path to waypoints
    waypoints = [(x[i] + resolution / 2, y[j] + resolution / 2) for i, j in path]

    # Simplify the route
    simplified_path = simplify_path(waypoints, tolerance=5)

    # Prepare waypoints
    my_waypoints = (
        [[port1.x, port1.y]]
        + [list(np.round(pt, 1)) for pt in simplified_path]
        + [[port2.x, port2.y]]
    )
    my_waypoints[1] = [my_waypoints[1][0], my_waypoints[0][1]]
    my_waypoints[-2] = [my_waypoints[-2][0], my_waypoints[-1][1]]

    return gf.routing.get_route_from_waypoints(
        waypoints=my_waypoints, cross_section=cross_section, bend=bend
    )


if __name__ == "__main__":
    c = gf.Component("get_route_astar_avoid_layers")
    # cross_section = "xs_metal_routing"
    # port_prefix = "e"
    # bend = gf.components.wire_corner

    cross_section = "xs_sc"
    port_prefix = "o"
    bend = gf.components.bend_euler

    cross_section = gf.get_cross_section(cross_section, radius=5)
    w = gf.components.straight(cross_section=cross_section)
    left = c << w
    right = c << w
    right.rotate(90.0)
    right.move((168, 63.2))

    obstacle = gf.components.rectangle(size=(250, 3), layer="M2")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle3 = c << obstacle
    obstacle4 = c << obstacle
    obstacle5 = c << obstacle
    obstacle4.rotate(90)
    obstacle5.rotate(90)
    obstacle1.ymin = 50
    obstacle1.xmin = -10
    obstacle2.xmin = 35
    obstacle3.ymin = 42
    obstacle3.xmin = 72.23
    obstacle4.xmin = 200
    obstacle4.ymin = 55
    obstacle5.xmin = 600
    obstacle5.ymin = 200
    port1 = left.ports[f"{port_prefix}1"]
    port2 = right.ports[f"{port_prefix}2"]

    route = get_route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=15,
        distance=12,
        avoid_layers=("M2",),
        bend=bend,
    )

    c.add(route.references)
    c.show()
