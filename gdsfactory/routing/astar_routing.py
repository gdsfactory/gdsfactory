from typing import List
from warnings import warn

import numpy as np

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.component import Component
from gdsfactory.routing import get_route_from_waypoints
from gdsfactory.routing.manhattan import route_manhattan
from gdsfactory.types import CrossSectionSpec


class Node:
    def __init__(self, parent=None, position: tuple = ()):
        """A node class for A* Pathfinding."""
        self.parent = parent  # parent node of current node
        self.position = position  # position of current node

        self.g = 0  # distance between current node and start node
        self.h = 0  # distance between current node and end node
        self.f = self.g + self.h  # cost of the node (sum of g and h)


def astar_routing(
    c: Component,
    input_port: Port,
    output_port: Port,
    resolution: float = 1,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
):
    cross_section = gf.get_cross_section(cross_section, **kwargs)
    grid, x, y = _generate_grid(c, resolution)

    input_orientation = {
        0.0: (resolution, 0),
        90.0: (0, resolution),
        180.0: (-resolution, 0),
        270.0: (0, -resolution),
    }[input_port.orientation]
    output_orientation = {
        0.0: (resolution, 0),
        90.0: (0, resolution),
        180.0: (-resolution, 0),
        270.0: (0, -resolution),
    }[output_port.orientation]

    # Instantiate nodes
    start_node = Node(
        None,
        (
            round(input_port.x + input_orientation[0]),
            round(input_port.y + input_orientation[1]),
        ),
    )
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(
        None,
        (
            round(output_port.x + output_orientation[0]),
            round(output_port.y + output_orientation[1]),
        ),
    )
    end_node.g = end_node.h = end_node.f = 0

    # Add the start node
    open_list = [start_node]
    closed_list = []

    # Loop until you find the end
    while open_list:
        # Get the current node
        current_index = 0
        for index in range(len(open_list)):
            if open_list[index].f < open_list[current_index].f:
                current_index = index

        # Pop current off open list, add to closed list
        current_node = open_list[current_index]
        closed_list.append(open_list.pop(current_index))

        # Found the goal
        if (
            current_node.position[0] == end_node.position[0]
            and current_node.position[1] == end_node.position[1]
        ):
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path = path[::-1]
            path.insert(0, input_port.center)
            path.append(output_port.center)

            return get_route_from_waypoints(path, cross_section=cross_section)

        # Generate children
        children = _generate_children(
            grid=grid,
            x=x,
            y=y,
            current_node=current_node,
            resolution=resolution,
        )

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + resolution
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                (child.position[1] - end_node.position[1]) ** 2
            )
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

    warn("A* algorithm failed, resorting to Manhattan routing. Watch for overlaps.")
    x = gf.get_cross_section(cross_section, **kwargs)
    return route_manhattan(input_port, output_port, cross_section=cross_section)


def _generate_grid(c: Component, resolution: float = 0.5) -> np.ndarray:
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
    for ref in c.references:
        bbox = ref.bbox
        xmin = np.abs(x - bbox[0][0]).argmin()
        xmax = np.abs(x - bbox[1][0]).argmin()
        ymin = np.abs(y - bbox[0][1]).argmin()
        ymax = np.abs(y - bbox[1][1]).argmin()

        grid[xmin:xmax, ymin:ymax] = 1

    return np.ndarray.round(grid, 3), np.ndarray.round(x, 3), np.ndarray.round(y, 3)


def _generate_children(
    current_node: Node,
    grid,
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
) -> List[Node]:
    children = []

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
                    if np.isclose(_, node_position[0], atol=1)
                )
            ][
                next(
                    i
                    for i, _ in enumerate(y)
                    if np.isclose(_, node_position[1], atol=1)
                )
            ]
            == 1.0
        ):
            continue

        # Create new node
        new_node = Node(current_node, node_position)

        # Append
        children.append(new_node)

    return children


if __name__ == "__main__":
    c = gf.Component()

    # mzi_ = c << gf.components.mzi()
    # mzi_2 = c << gf.components.mzi()

    # mzi_2.move(destination=(100, -10))
    rect1 = c << gf.components.rectangle()
    rect2 = c << gf.components.rectangle()
    rect3 = c << gf.components.rectangle((2, 2))
    rect2.move(destination=(8, 4))
    rect3.move(destination=(5.5, 1.5))

    port1 = Port(
        "o1", 0, rect1.center + (0, 3), cross_section=gf.get_cross_section("strip")
    )
    port2 = port1.copy("o2")
    port2.orientation = 180
    port2.center = rect2.center + (0, -3)
    c.add_ports([port1, port2])
    c.show(show_ports=True)

    route = astar_routing(c, port1, port2, radius=0.5, width=0.5)
    # route = route_manhattan(port1, port2, radius=0.5, width=0.5)
    # route = astar_routing(c, mzi_.ports["o2"], mzi_2.ports["o1"], radius=0.5)
    # route = route_manhattan(mzi_.ports["o2"], mzi_2.ports["o1"], radius=0.5)
    c.add(route.references)

    c.show(show_ports=True)
