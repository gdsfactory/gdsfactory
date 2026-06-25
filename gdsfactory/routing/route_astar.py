from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, cast

import klayout.dbcore as kdb
import networkx as nx
import numpy as np
import numpy.typing as npt
from kfactory.routing.aa.optical import OpticalAllAngleRoute
from kfactory.routing.generic import ManhattanRoute
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

ROUTE_BUNDLE_KWARGS = {"raise_on_error"}


def get_route_bend_count(route: Route) -> int:
    """Returns the number of 90° bends for any Route type.

    ManhattanRoute → uses n_bend90 attribute
    OpticalAllAngleRoute → treated as having 100 Manhattan bends, so that it is
    deprioritized when selecting the route with the fewest bends.
    """
    if isinstance(route, ManhattanRoute):
        return route.n_bend90
    if isinstance(route, OpticalAllAngleRoute):
        return 100
    # mypy exhaustiveness
    raise TypeError(f"Unsupported Route type: {type(route)}")


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


def _parse_bbox_to_array(bbox: kdb.DBox | kdb.Box) -> npt.NDArray[np.floating[Any]]:
    """Parses bbox in the form of (a,b;c,d) to [[a, b], [c, d]].

    Args:
        bbox: Parses bbox in the form of (a,b;c,d).

    """
    bbox_values = ((bbox.p1.x, bbox.p1.y), (bbox.p2.x, bbox.p2.y))
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

    _a1 = float(bbox[0][0]) - max(distance, resolution)
    _a2 = float(bbox[1][0]) + max(distance, resolution)
    _b1 = float(bbox[0][1]) - max(distance, resolution)
    _b2 = float(bbox[1][1]) + max(distance, resolution)

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
            bbox_array = _parse_bbox_to_array(inst.dbbox())
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
                    bbox_array = _parse_bbox_to_array(polygon.bbox())
                    bbox_array_float = bbox_array / 1000
                    xmin = np.abs(x - bbox_array_float[0][0] + distance).argmin()
                    xmax = np.abs(x - bbox_array_float[1][0] - distance).argmin()
                    ymin = np.abs(y - bbox_array_float[0][1] + distance).argmin()
                    ymax = np.abs(y - bbox_array_float[1][1] - distance).argmin()
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


def nearest_open_node(
    node: tuple[int, int],
    blocked: npt.NDArray[np.bool_],
) -> tuple[int, int]:
    """Return the closest unblocked grid node using local breadth-first search."""
    shape_x, shape_y = blocked.shape
    i = min(max(node[0], 0), shape_x - 1)
    j = min(max(node[1], 0), shape_y - 1)
    if not blocked[i, j]:
        return i, j

    queue: deque[tuple[int, int]] = deque([(i, j)])
    seen = {(i, j)}
    while queue:
        ci, cj = queue.popleft()
        for ni, nj in (
            (ci - 1, cj),
            (ci + 1, cj),
            (ci, cj - 1),
            (ci, cj + 1),
        ):
            if ni < 0 or nj < 0 or ni >= shape_x or nj >= shape_y or (ni, nj) in seen:
                continue
            if not blocked[ni, nj]:
                return ni, nj
            seen.add((ni, nj))
            queue.append((ni, nj))

    raise RuntimeError("No open grid nodes available for A* routing.")


def astar_grid(
    blocked: npt.NDArray[np.bool_],
    start_node: tuple[int, int],
    end_node: tuple[int, int],
) -> list[tuple[int, int]]:
    """Find a shortest Manhattan grid path without constructing a NetworkX graph."""
    start = nearest_open_node(start_node, blocked)
    end = nearest_open_node(end_node, blocked)
    if start == end:
        return [start]

    nx_, ny_ = blocked.shape
    g_score = np.full(blocked.shape, np.inf)
    closed = np.zeros(blocked.shape, dtype=bool)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    def heuristic(node: tuple[int, int]) -> int:
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    g_score[start] = 0
    heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start), 0, start)]

    while heap:
        _, current_g, current = heapq.heappop(heap)
        if closed[current]:
            continue
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed[current] = True
        ci, cj = current
        for neighbor in (
            (ci - 1, cj),
            (ci + 1, cj),
            (ci, cj - 1),
            (ci, cj + 1),
        ):
            ni, nj = neighbor
            if (
                ni < 0
                or nj < 0
                or ni >= nx_
                or nj >= ny_
                or blocked[neighbor]
                or closed[neighbor]
            ):
                continue
            tentative_g = current_g + 1
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    heap, (tentative_g + heuristic(neighbor), tentative_g, neighbor)
                )

    raise nx.NetworkXNoPath(f"No path between {start_node} and {end_node}.")


def path_from_nodes(
    *,
    G: nx.Graph | None,
    blocked_grid: npt.NDArray[np.bool_] | None,
    start_node: tuple[int, int],
    end_node: tuple[int, int],
) -> list[tuple[int, int]]:
    if blocked_grid is not None:
        return astar_grid(blocked_grid, start_node, end_node)

    assert G is not None, "route_astar_with_nodes: G must not be None"
    start_node = min(
        G.nodes,
        key=lambda node: float(np.linalg.norm(np.array(node) - np.array(start_node))),
    )
    end_node = min(
        G.nodes,
        key=lambda node: float(np.linalg.norm(np.array(node) - np.array(end_node))),
    )
    return cast("list[tuple[int, int]]", nx.astar_path(G, start_node, end_node))


def count_bends(waypoints: Sequence[DPoint]) -> int:
    if len(waypoints) < 3:
        return 0
    bends = 0
    previous_direction: tuple[float, float] | None = None
    for p1, p2 in pairwise(waypoints):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        length = np.hypot(dx, dy)
        direction = (round(dx / length, 4), round(dy / length, 4))
        if previous_direction is not None and direction != previous_direction:
            bends += 1
        previous_direction = direction
    return bends


def route_astar_waypoints(
    *,
    port1: Port,
    port2: Port,
    resolution: float,
    G: nx.Graph | None,
    blocked_grid: npt.NDArray[np.bool_] | None,
    x: npt.NDArray[np.number[Any]],
    y: npt.NDArray[np.number[Any]],
    start_node: tuple[int, int],
    end_node: tuple[int, int],
) -> list[DPoint]:
    path = path_from_nodes(
        G=G,
        blocked_grid=blocked_grid,
        start_node=start_node,
        end_node=end_node,
    )

    # Convert path to waypoints, move to center of grid cell
    waypoints = [(x[i] + resolution / 2, y[j] + resolution / 2) for i, j in path]

    # Simplify the route
    simplified_path = simplify_path(waypoints, tolerance=0.05)

    # Turning simplified_path, which is list of tuples, to my_waypoints, which is list of lists
    my_waypoints = [list(np.round(pt, 1)) for pt in simplified_path]

    # List to iterate through both ports and the indices of the two waypoints closest to them
    port_data = [
        (port1, 0, 1),
        (port2, -1, -2),
    ]

    for port, closest_index, second_closest_index in port_data:
        if len(my_waypoints) < 2:
            continue
        # If the orientation of the port is vertical and the path leading to it is vertical
        if (
            port.orientation in [90, 270]
            and abs(my_waypoints[closest_index][0] - port.x) <= resolution
            and my_waypoints[second_closest_index][0] == my_waypoints[closest_index][0]
        ):
            # In order to not have a bend, the last two waypoints must be aligned with the port x
            my_waypoints[second_closest_index][0] = port.x
            my_waypoints[closest_index][0] = port.x
        # If the orientation of the port is horizontal and the path leading to it is horizontal
        elif (
            port.orientation in [0, 180]
            and abs(my_waypoints[closest_index][1] - port.y) <= resolution
            and my_waypoints[second_closest_index][1] == my_waypoints[closest_index][1]
        ):
            # In order to not have a bend, the last two waypoints must be aligned with the port y
            my_waypoints[second_closest_index][1] = port.y
            my_waypoints[closest_index][1] = port.y

    waypoints_ = [DPoint(x, y) for x, y in my_waypoints]
    return gf.kf.routing.manhattan.clean_points(waypoints_)


def route_astar_single(
    component: Component,
    port1: Port,
    port2: Port,
    resolution: float = 1,
    cross_section: CrossSectionSpec = "strip",
    bend: ComponentSpec = "wire_corner",
    G: nx.Graph | None = None,
    x: npt.NDArray[np.number[Any]] | None = None,
    y: npt.NDArray[np.number[Any]] | None = None,
    start_node: tuple[int, int] | None = None,
    end_node: tuple[int, int] | None = None,
    blocked_grid: npt.NDArray[np.bool_] | None = None,
    **kwargs: Any,
) -> Route:
    """Runs a single A* routing attempt between two ports.

    Uses explicitly provided start and end grid-node indices.

    Args:
        component: Component in which the final route geometry will be inserted.
        port1: Start port of the route.
        port2: End port of the route.
        resolution: Grid discretization step in microns.
        cross_section: Cross-section specification for the routed waveguide.
        bend: Component used for bends (e.g. wire_corner or bend_euler).
        G: Precomputed NetworkX grid graph with obstacle nodes removed.
        x: 1D array of x-coordinates for grid columns.
        y: 1D array of y-coordinates for grid rows.
        start_node: Approximate (i, j) index of the start grid cell.
        end_node: Approximate (i, j) index of the end grid cell.
        blocked_grid: Precomputed grid with blocked obstacle cells.
        **kwargs: Additional arguments passed into the cross-section or route_bundle.

    Returns:
        A single `Route` object created from the computed A* path.

    Raises:
        ValueError: If any required grid input is None.
        nx.NetworkXNoPath: If no valid A* route exists between the nodes.
    """
    if x is None:
        raise ValueError("x array must not be None")
    if y is None:
        raise ValueError("y array must not be None")
    if start_node is None:
        raise ValueError("start_node must not be None")
    if end_node is None:
        raise ValueError("end_node must not be None")

    waypoints_ = route_astar_waypoints(
        port1=port1,
        port2=port2,
        resolution=resolution,
        G=G,
        blocked_grid=blocked_grid,
        x=x,
        y=y,
        start_node=start_node,
        end_node=end_node,
    )
    route_bundle_kwargs = {
        key: value for key, value in kwargs.items() if key in ROUTE_BUNDLE_KWARGS
    }
    cross_section_kwargs = {
        key: value for key, value in kwargs.items() if key not in ROUTE_BUNDLE_KWARGS
    }
    cross_section = (
        gf.get_cross_section(cross_section, **cross_section_kwargs)
        if cross_section_kwargs
        else cross_section
    )
    return gf.routing.route_bundle(
        component=component,
        ports1=[port1],
        ports2=[port2],
        waypoints=waypoints_,
        cross_section=cross_section,
        bend=bend,
        **route_bundle_kwargs,
    )[0]


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
    """A* router that evaluates several start/end node options and returns the best route.

    Candidate paths are computed on a grid, validated on a scratch component,
    and only the optimized route is built on the real component.

    Args:
        component: Component on which the final (optimized) route will be built.
        port1: Starting port for the route.
        port2: Ending port for the route.
        resolution: Grid discretization step in microns.
        avoid_layers: Layers that should be treated as obstacles.
        distance: Clearance distance from obstacles in microns.
        cross_section: Cross-section specification for the routed waveguide.
        bend: Component to use for bends (e.g. ``wire_corner`` or ``bend_euler``).
        **kwargs: Additional keyword arguments forwarded to the cross-section or route_bundle.

    Returns:
        Route: The route generated using the start/end node pairing
        that yields the fewest bends.

    Raises:
        RuntimeError: If all A* attempts fail for all start/end node combinations.
        ValueError: If a port has an unsupported orientation.
    """
    route_bundle_kwargs = {
        key: value for key, value in kwargs.items() if key in ROUTE_BUNDLE_KWARGS
    }
    cross_section_kwargs = {
        key: value for key, value in kwargs.items() if key not in ROUTE_BUNDLE_KWARGS
    }
    cross_section = gf.get_cross_section(cross_section, **cross_section_kwargs)
    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)
    blocked_grid = grid == 1

    distance_from_node_to_port = 3 * (cross_section.radius or 3)  # in um

    if port1.orientation in [0, 180]:
        start_node_coordinates = [
            (
                port1.x
                + np.cos(port1.orientation * np.pi / 180) * distance_from_node_to_port,
                port1.y,
            ),
            (
                port1.x
                + np.cos(port1.orientation * np.pi / 180) * distance_from_node_to_port,
                port1.y + distance_from_node_to_port,
            ),
            (
                port1.x
                + np.cos(port1.orientation * np.pi / 180) * distance_from_node_to_port,
                port1.y - distance_from_node_to_port,
            ),
        ]
    elif port1.orientation in [90, 270]:
        start_node_coordinates = [
            (
                port1.x,
                port1.y
                + np.sin(port1.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
            (
                port1.x + distance_from_node_to_port,
                port1.y
                + np.sin(port1.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
            (
                port1.x - distance_from_node_to_port,
                port1.y
                + np.sin(port1.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
        ]
    else:
        raise ValueError("port1 orientation must be in [0, 90, 180, 270]")
    if port2.orientation in [0, 180]:
        end_node_coordinates = [
            (
                port2.x
                + np.cos(port2.orientation * np.pi / 180) * distance_from_node_to_port,
                port2.y,
            ),
            (
                port2.x
                + np.cos(port2.orientation * np.pi / 180) * distance_from_node_to_port,
                port2.y + distance_from_node_to_port,
            ),
            (
                port2.x
                + np.cos(port2.orientation * np.pi / 180) * distance_from_node_to_port,
                port2.y - distance_from_node_to_port,
            ),
        ]
    elif port2.orientation in [90, 270]:
        end_node_coordinates = [
            (
                port2.x,
                port2.y
                + np.sin(port2.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
            (
                port2.x + distance_from_node_to_port,
                port2.y
                + np.sin(port2.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
            (
                port2.x - distance_from_node_to_port,
                port2.y
                + np.sin(port2.orientation * np.pi / 180) * distance_from_node_to_port,
            ),
        ]
    else:
        raise ValueError("port2 orientation must be in [0, 90, 180, 270]")

    # Score candidate paths without placing temporary geometry.
    candidates: list[tuple[int, int, list[DPoint]]] = []

    for start_coords in start_node_coordinates:
        for end_coords in end_node_coordinates:
            try:
                waypoints = route_astar_waypoints(
                    port1=port1,
                    port2=port2,
                    start_node=(
                        round((start_coords[0] - x.min()) / resolution),
                        round((start_coords[1] - y.min()) / resolution),
                    ),
                    end_node=(
                        round((end_coords[0] - x.min()) / resolution),
                        round((end_coords[1] - y.min()) / resolution),
                    ),
                    resolution=resolution,
                    G=None,
                    blocked_grid=blocked_grid,
                    x=x,
                    y=y,
                )
                candidates.append(
                    (
                        count_bends(waypoints),
                        len(waypoints),
                        waypoints,
                    )
                )

            except Exception:
                continue

    if not candidates:
        raise RuntimeError("All A* routing attempts failed.")

    # Validate generated waypoints against route_bundle before selecting a route.
    # A low-bend A* path can still be unbuildable once bend-radius and collision
    # constraints are applied downstream.
    valid_candidates: list[tuple[int, int, list[DPoint]]] = []
    for _, waypoint_count, waypoints in sorted(
        candidates, key=lambda item: (item[0], item[1])
    ):
        try:
            route = gf.routing.route_bundle(
                component=gf.Component(),
                ports1=[port1],
                ports2=[port2],
                waypoints=waypoints,
                cross_section=cross_section,
                bend=bend,
                raise_on_error=True,
            )
        except Exception:
            continue
        valid_candidates.append(
            (get_route_bend_count(route[0]), waypoint_count, waypoints)
        )

    if not valid_candidates:
        raise RuntimeError("All A* routing attempts failed.")

    _, _, optimized_waypoints = min(
        valid_candidates, key=lambda item: (item[0], item[1])
    )

    # Build optimized route on real component
    return gf.routing.route_bundle(
        component=component,
        ports1=[port1],
        ports2=[port2],
        waypoints=optimized_waypoints,
        cross_section=cross_section,
        bend=bend,
        **route_bundle_kwargs,
    )[0]
