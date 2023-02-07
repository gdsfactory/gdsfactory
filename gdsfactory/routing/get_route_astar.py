from __future__ import annotations

from typing import List
from warnings import warn

import numpy as np

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.components.wire import wire_corner
from gdsfactory.routing import get_route_from_waypoints
from gdsfactory.routing.manhattan import route_manhattan
from gdsfactory.typings import CrossSectionSpec, LayerSpec, Route


class Node:
    def __init__(self, parent=None, position: tuple = ()):
        """Initializes a node. A node is a point on the grid."""
        self.parent = parent  # parent node of current node
        self.position = position  # position of current node

        self.g = 0  # distance between current node and start node
        self.h = 0  # distance between current node and end node
        self.f = self.g + self.h  # cost of the node (sum of g and h)


def get_route_astar(
    component: Component,
    port1: Port,
    port2: Port,
    resolution: float = 1,
    avoid_layers: List[LayerSpec] = None,
    distance: float = 1,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Route:
    """A* routing function. Finds a route between two ports avoiding obstacles.

    Args:
        component: Component the route, and ports belong to.
        port1: input.
        port2: output.
        resolution: discretization resolution in um.
            Lower resolution can help avoid accidental overlapping between route
            and components but adds more bends.
            The resolution decides how many "leaps/hops" the algorithm has to do.
        avoid_layers: list of layers to avoid.
        distance: distance from obstacles in um.
        cross_section: spec.
        kwargs: cross_section settings.
    """
    cross_section = gf.get_cross_section(cross_section, **kwargs)

    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)

    # Tell the algorithm which start and end directions to follow based on port orientation
    input_orientation = {
        0.0: (resolution, 0),
        90.0: (0, resolution),
        180.0: (-resolution, 0),
        270.0: (0, -resolution),
        None: (0, 0),
    }[port1.orientation]

    output_orientation = {
        0.0: (resolution, 0),
        90.0: (0, resolution),
        180.0: (-resolution, 0),
        270.0: (0, -resolution),
        None: (0, 0),
    }[port2.orientation]

    # Instantiate nodes
    start_node = Node(
        None,
        (
            round(port1.x + input_orientation[0]),
            round(port1.y + input_orientation[1]),
        ),
    )
    start_node.g = start_node.h = start_node.f = 0

    end_node = Node(
        None,
        (
            round(port2.x + output_orientation[0]),
            round(port2.y + output_orientation[1]),
        ),
    )
    end_node.g = end_node.h = end_node.f = 0

    # Add the start node
    open_list = [start_node]
    closed = []

    while open_list:
        # Current node
        current_index = 0
        for index in range(len(open_list)):
            if open_list[index].f < open_list[current_index].f:
                current_index = index

        # Pop current off open_list list, add to closed list
        current_node = open_list[current_index]
        closed.append(open_list.pop(current_index))

        # Reached end port
        if (
            current_node.position[0] == end_node.position[0]
            and current_node.position[1] == end_node.position[1]
        ):
            points = []
            current = current_node

            # trace back path from end node to start node
            while current is not None:
                points.append(current.position)
                current = current.parent
            # reverse to get true path
            points = points[::-1]

            # add the start and end ports
            points.insert(0, port1.center)
            points.append(port2.center)

            # return route from points
            if cross_section.radius:
                return get_route_from_waypoints(points, cross_section=cross_section)
            else:
                return get_route_from_waypoints(
                    points, cross_section=cross_section, bend=wire_corner
                )

        # Generate neighbours
        neighbours = _generate_neighbours(
            grid=grid,
            x=x,
            y=y,
            current_node=current_node,
            resolution=resolution,
        )

        # Loop through neighbours
        for neighbour in neighbours:
            for closed_neighbour in closed:
                if neighbour == closed_neighbour:
                    continue

            # Compute f, g, h
            neighbour.g = current_node.g + resolution
            # print(neighbour.g)
            neighbour.h = np.sqrt(
                (neighbour.position[0] - end_node.position[0]) ** 2
            ) + ((neighbour.position[1] - end_node.position[1]) ** 2)
            neighbour.f = neighbour.g + neighbour.h

            if current_node.parent is not None and (
                neighbour.position[0] - current_node.parent.position[0],
                neighbour.position[1] - current_node.parent.position[1],
            ) in [
                (resolution, -resolution),
                (-resolution, resolution),
                (resolution, resolution),
                (-resolution, -resolution),
            ]:
                neighbour.f *= 1.1  # penalize for turns

            # neighbour is already in the open_list
            for open_list_node in open_list:
                if neighbour == open_list_node and neighbour.g > open_list_node.g:
                    continue

            # Add the neighbour to open_list
            open_list.append(neighbour)

    warn("A* algorithm failed, resorting to Manhattan routing. Watch for overlaps.")
    return route_manhattan(port1, port2, cross_section=cross_section)


def _extract_all_bbox(c: Component, avoid_layers: List[LayerSpec] = None):
    """Extract all polygons whose layer is in `avoid_layers`."""
    return [c.get_polygons(layer) for layer in avoid_layers]


def _generate_grid(
    c: Component,
    resolution: float = 0.5,
    avoid_layers: List[LayerSpec] = None,
    distance: float = 1,
) -> np.ndarray:
    """Generate discretization grid that the algorithm will step through."""
    bbox = c.bbox
    x, y = np.meshgrid(
        np.linspace(
            bbox[0][0],
            bbox[1][0],
            int((bbox[1][0] - bbox[0][0]) / resolution),
            endpoint=True,
        ),
        np.linspace(
            bbox[0][1],
            bbox[1][1],
            int((bbox[1][1] - bbox[0][1]) / resolution),
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


def _generate_neighbours(
    current_node: Node,
    grid,
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
) -> List[Node]:
    """Generate neighbours of a node."""
    neighbours = []

    for new_position in [
        (0, -resolution),
        (0, resolution),
        (-resolution, 0),
        (resolution, 0),
    ]:  # Adjacent nodes along Manhattan path
        # Get node position
        node_position = (
            current_node.position[0] + new_position[0],
            current_node.position[1] + new_position[1],
        )

        # Make sure within range and not in obstacle
        if (
            node_position[0] > x.max()
            or node_position[0] < x.min()
            or node_position[1] > y.max()
            or node_position[1] < y.min()
        ):
            continue

        if (
            grid[
                next(
                    i
                    for i, _ in enumerate(x)
                    if np.isclose(_, node_position[0], atol=resolution)
                )
            ][
                next(
                    i
                    for i, _ in enumerate(y)
                    if np.isclose(_, node_position[1], atol=resolution)
                )
            ]
            == 1.0
        ):
            continue

        # Create new node
        new_node = Node(current_node, node_position)

        # Append
        neighbours.append(new_node)

    return neighbours


if __name__ == "__main__":
    # cross_section = gf.get_cross_section("metal1", width=3)

    # c = gf.Component("get_route_astar")
    # w = gf.components.straight(cross_section=cross_section)

    # left = c << w
    # right = c << w
    # right.move((100, 80))

    # obstacle = gf.components.rectangle(size=(100, 3), layer="M1")
    # obstacle1 = c << obstacle
    # obstacle2 = c << obstacle
    # obstacle1.ymin = 40
    # obstacle2.xmin = 25

    # port1 = left.ports["e2"]
    # port2 = right.ports["e2"]

    # routes = get_route_astar(
    #     component=c,
    #     port1=port1,
    #     port2=port2,
    #     cross_section=cross_section,
    #     resolution=5,
    #     distance=6.5,
    #     avoid_layers=("M1",),
    # )
    # c.add(routes.references)

    c = gf.Component("get_route_astar_avoid_layers")
    cross_section = gf.get_cross_section("metal1", width=3)
    w = gf.components.straight(cross_section=cross_section)

    left = c << w
    right = c << w
    right.move((100, 80))

    obstacle = gf.components.rectangle(size=(100, 3), layer="WG")
    obstacle1 = c << obstacle
    obstacle2 = c << obstacle
    obstacle1.ymin = 40
    obstacle2.xmin = 25

    port1 = left.ports["e2"]
    port2 = right.ports["e2"]

    routes = gf.routing.get_route_astar(
        component=c,
        port1=port1,
        port2=port2,
        cross_section=cross_section,
        resolution=10,
        distance=6.5,
        avoid_layers=("M1",),
    )

    c.add(routes.references)
    c.show()
