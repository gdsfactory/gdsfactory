from __future__ import annotations

import warnings

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


def get_route_networkx(
    component: Component,
    port1: Port,
    port2: Port,
    resolution: float = 1,
    avoid_layers: list[LayerSpec] | None = None,
    distance: float = 8,
    cross_section: CrossSectionSpec = "xs_sc",
    radius: float = 10.0,  # Radius of curvature
    bend: gf.Component = gf.components.wire_corner,
    **kwargs,
) -> Route:
    """Bidirectional routing function using NetworkX. Finds a route between two ports avoiding obstacles.

    Args:
        component: Component the route, and ports belong to.
        port1: input.
        port2: output.
        resolution: discretization resolution in um.
        avoid_layers: list of layers to avoid.
        distance: distance from obstacles in um.
        cross_section: spec.
        radius: radius of curvature for smoothing (default: 10.0)
        bend: component to use for bends (default: gf.components.wire_corner)
        kwargs: cross_section settings.
    """
    cross_section = gf.get_cross_section(cross_section, **kwargs)

    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)

    # Create graph
    G = nx.grid_2d_graph(len(x), len(y))

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
    # find the closet one
    for i in G:
        node = i
        distance1_lim = 15 * resolution
        distance2_lim = 15 * resolution
        distance1 = np.sqrt(
            pow(start_node[0] - node[0], 2) + pow(start_node[1] - node[1], 2)
        )
        distance2 = np.sqrt(
            pow(end_node[0] - node[0], 2) + pow(end_node[1] - node[1], 2)
        )
        if distance1 < distance1_lim:
            start_node_auxi = node
            distance1_lim = distance1
        if distance2 < distance2_lim:
            end_node_auxi = node
            distance2_lim = distance2
    # Find shortest path
    start_node = start_node_auxi
    end_node = end_node_auxi
    try:
        path = nx.astar_path(G, start_node, end_node)
    except nx.NetworkXNoPath:
        warnings.warn("No path found.")
        return None

    # Convert path to waypoints
    waypoints = [(x[i] + resolution / 2, y[j] + resolution / 2) for i, j in path]

    # # Smooth the route
    simplified_path = simplify_path(waypoints, tolerance=5)
    # Convert path to steps
    steps = []
    paths = []
    my_waypoints = []
    # print(waypoints)
    my_waypoints.append([port1.x, port1.y])
    paths.append([port1.x, port1.y])
    for i in range(len(simplified_path)):
        paths.append(np.round(simplified_path[i], 1))
        x1, y1 = simplified_path[i]
        steps.append({"x": x1, "y": y1})
    for i in range(len(waypoints)):
        my_waypoints.append(waypoints[i])
    # print(steps)
    paths.append([port2.x, port2.y])
    my_waypoints.append([port2.x, port2.y])
    my_waypoints[1] = [my_waypoints[1][0], my_waypoints[0][1]]
    my_waypoints[len(my_waypoints) - 2] = [
        my_waypoints[len(my_waypoints) - 2][0],
        my_waypoints[len(my_waypoints) - 1][1],
    ]
    routes = []
    for i in range(len(steps)):
        position1 = []
        position2 = []
        if i == 0:
            position1.append(port1.x)
            position1.append(port1.y)
            position2.append(steps[0]["x"])
            position2.append(steps[0]["y"])
        elif i == len(steps):
            position1.append(steps[len(steps)]["x"])
            position1.append(steps[len(steps)]["y"])
            position2.append(port2.x)
            position2.append(port2.y)
        else:
            position1.append(steps[i - 1]["x"])
            position1.append(steps[i - 1]["y"])
            position2.append(steps[i]["x"])
            position2.append(steps[i]["y"])

        port1 = gf.Port(
            name="aux1",
            center=(position1[0] - 5, position1[1]),
            orientation=port1.orientation,
            cross_section=cross_section,
            width=port1.width,
            layer=port1.layer,
        )
        port2 = gf.Port(
            name="aux2",
            center=(position2[0] + 10, position2[1]),
            orientation=port1.orientation,
            cross_section=cross_section,
            width=port1.width,
            layer=port1.layer,
        )
        routes.append(
            gf.routing.get_route(
                input_port=port1,
                output_port=port2,
                cross_section=cross_section,
                with_sbend=True,
                end_straight_length=0.5,
                radius=radius,
            )
        )

    return gf.routing.get_route_from_waypoints(
        waypoints=my_waypoints, cross_section=cross_section, bend=bend
    )


if __name__ == "__main__":
    c = gf.Component("get_route_astar_avoid_layers")
    cross_section = "xs_sc"
    port_prefix = "e"
    port_prefix = "o"
    cross_section = gf.get_cross_section(cross_section)
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

    routes = get_route_networkx(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=1,
        distance=12,
        avoid_layers=("M2",),
        bend=gf.components.bend_euler,
    )

    # for route in routes:
    #     c.add(route.references)
    c.add(routes.references)
    c.show()
