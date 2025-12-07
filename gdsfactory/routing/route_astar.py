from __future__ import annotations

from collections.abc import Sequence
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


def get_route_bend_count(route: Route) -> int:
    """
    Returns the number of 90° bends for any Route type.

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
    **kwargs: Any,
) -> Route:
    """
    Runs a single A* routing attempt between two ports using explicitly
    provided start and end grid-node indices.

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
        **kwargs: Additional arguments passed into the cross-section.

    Returns:
        A single `Route` object created from the computed A* path.

    Raises:
        ValueError: If start_node or end_node is None.
        nx.NetworkXNoPath: If no valid A* route exists between the nodes.
    """
    assert G is not None, "route_astar_with_nodes: G must not be None"
    assert x is not None, "x array must not be None"
    assert y is not None, "y array must not be None"
    assert start_node is not None, "start_node must not be None"
    assert end_node is not None, "end_node must not be None"

    # Find the indices of the closest valid nodes
    start_node = min(
        G.nodes,
        key=lambda node: float(np.linalg.norm(np.array(node) - np.array(start_node))),
    )
    end_node = min(
        G.nodes,
        key=lambda node: float(np.linalg.norm(np.array(node) - np.array(end_node))),
    )

    path = nx.astar_path(G, start_node, end_node)  # Find shortest path

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

    # Convert to native floats or Point instances
    waypoints_ = [DPoint(x, y) for x, y in my_waypoints]
    return gf.routing.route_bundle(
        component=component,
        ports1=[port1],
        ports2=[port2],
        waypoints=gf.kf.routing.manhattan.clean_points(waypoints_),
        cross_section=cross_section,
        bend=bend,
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
    """
    A* router that evaluates several start/end node options and returns the
    route with the fewest bends. All candidates are computed on a temporary
    copy of the component; only the optimized route is rebuilt on the real one.

    Args:
        component: Component on which the final (optimized) route will be built.
        port1: Starting port for the route.
        port2: Ending port for the route.
        resolution: Grid discretization step in microns.
        avoid_layers: Layers that should be treated as obstacles.
        distance: Clearance distance from obstacles in microns.
        cross_section: Cross-section specification for the routed waveguide.
        bend: Component to use for bends (e.g. ``wire_corner`` or ``bend_euler``).
        **kwargs: Additional keyword arguments forwarded to the cross-section.

    Returns:
        Route: The route generated using the start/end node pairing
        that yields the fewest bends.

    Raises:
        RuntimeError: If all A* attempts fail for all start/end node combinations.
        ValueError: If a port has an unsupported orientation.
    """
    # Create a copy of the component to run preliminary A* attempts
    copy_of_component = component.copy()
    cross_section = gf.get_cross_section(cross_section, **kwargs)
    grid, x, y = _generate_grid(component, resolution, avoid_layers, distance)
    G_ = nx.grid_2d_graph(len(x), len(y))
    G = cast("nx.Graph", G_)

    # Remove nodes representing obstacles
    for i in range(len(x)):
        for j in range(len(y)):
            if grid[i, j] == 1:
                G.remove_node((i, j))

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

    # List of all attempted routes with their start/end coordinates, in order to select the one with fewest bends
    candidates: list[tuple[Route, tuple[float, float], tuple[float, float]]] = []

    for start_coords in start_node_coordinates:
        for end_coords in end_node_coordinates:
            try:
                route = route_astar_single(
                    component=copy_of_component,
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
                    cross_section=cross_section,
                    bend=bend,
                    G=G,
                    x=x,
                    y=y,
                    **kwargs,
                )
                candidates.append((route, start_coords, end_coords))

            except Exception as e:
                print(f"Attempt failed: {e}")
                continue

    if not candidates:
        raise RuntimeError("All A* routing attempts failed.")

    # Choose the route with the fewest bends
    _, optimized_start_coords, optimized_end_coords = min(
        candidates, key=lambda item: get_route_bend_count(item[0])
    )

    # Build optimized route on real component
    return route_astar_single(
        component=component,
        port1=port1,
        port2=port2,
        start_node=(
            round((optimized_start_coords[0] - x.min()) / resolution),
            round((optimized_start_coords[1] - y.min()) / resolution),
        ),
        end_node=(
            round((optimized_end_coords[0] - x.min()) / resolution),
            round((optimized_end_coords[1] - y.min()) / resolution),
        ),
        resolution=resolution,
        avoid_layers=avoid_layers,
        distance=distance,
        cross_section=cross_section,
        bend=bend,
        G=G,
        x=x,
        y=y,
        **kwargs,
    )
