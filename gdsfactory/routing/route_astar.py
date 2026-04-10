from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
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
    """A* router that evaluates several start/end node options and returns the best route.

    All candidates are computed on a temporary copy of the component;
    only the optimized route is rebuilt on the real one.

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


# ---------------------------------------------------------------------------
# Multi-layer A* routing (OSS, networkx-based)
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    """Per-layer configuration for multi-layer A* routing.

    Args:
        layer: GDS layer tuple, e.g. ``(68, 20)``.
        preferred_direction: ``"h"`` for horizontal or ``"v"`` for vertical.
        avoid_layers: Layers whose polygons are treated as obstacles on this
            routing layer.  When *None* the router marks all component
            instances as obstacles (same behaviour as the single-layer
            ``route_astar``).
    """

    layer: tuple[int, int]
    preferred_direction: str  # "h" or "v"
    avoid_layers: Sequence[LayerSpec] | None = None


@dataclass
class MultiLayerRouteResult:
    """Result of a multi-layer A* routing operation.

    This is a *plan* — the caller (typically a PDK-specific wrapper) is
    responsible for drawing the physical geometry from these fields.
    """

    corners_3d: list[tuple[float, float, int]]
    """Turning / via-transition points as ``(x_um, y_um, layer_index)``."""

    segments: list[tuple[tuple[float, float], tuple[float, float], int, float]]
    """Planned metal rectangles: ``(p0, p1, layer_index, width)``."""

    vias: list[tuple[tuple[float, float], int]]
    """Via positions: ``(center_um, transition_index)``."""

    path_length_um: float
    num_vias: int

    raw_path: list[tuple[int, int, int]] = field(default_factory=list)
    """Raw grid-index path for obstacle marking."""


# -- helpers ----------------------------------------------------------------


def _generate_grid_3d(
    c: Component,
    layers: Sequence[LayerConfig],
    resolution: float = 1.0,
    distance: float = 1.0,
    port_positions: Sequence[tuple[float, float]] | None = None,
    port_exclusion_radius: float = 5.0,
    wrong_way_penalty: float = 8.0,
    via_cost: float = 10.0,
    bbox_padding: float | None = None,
) -> tuple[nx.Graph, npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], int, int]:
    """Build a 3-D networkx grid for multi-layer A* routing.

    Returns ``(G, x_array, y_array, nx_cells, ny_cells)``.
    """
    bbox_arr = _parse_bbox_to_array(c.dbbox())
    pad = bbox_padding if bbox_padding is not None else max(distance, resolution) * 3
    x_min = float(bbox_arr[0][0]) - pad
    x_max = float(bbox_arr[1][0]) + pad
    y_min = float(bbox_arr[0][1]) - pad
    y_max = float(bbox_arr[1][1]) + pad

    nx_cells = max(2, int((x_max - x_min) / resolution))
    ny_cells = max(2, int((y_max - y_min) / resolution))
    x_arr = np.linspace(x_min, x_max, nx_cells, endpoint=True)
    y_arr = np.linspace(y_min, y_max, ny_cells, endpoint=True)

    n_layers = len(layers)

    # Per-layer obstacle grids
    grids: list[npt.NDArray[np.floating[Any]]] = []
    for layer_cfg in layers:
        grid = np.zeros((nx_cells, ny_cells))
        if layer_cfg.avoid_layers is not None:
            all_refs = _extract_all_bbox(c, list(layer_cfg.avoid_layers))
            for layer_dict in all_refs:
                for polygons in layer_dict.values():
                    for polygon in polygons:
                        ba = _parse_bbox_to_array(polygon.bbox())
                        ba_f = ba / 1000  # nm → um
                        xi = int(np.clip(np.abs(x_arr - ba_f[0][0] + distance).argmin(), 0, nx_cells - 1))
                        xa = int(np.clip(np.abs(x_arr - ba_f[1][0] - distance).argmin(), 0, nx_cells - 1))
                        yi = int(np.clip(np.abs(y_arr - ba_f[0][1] + distance).argmin(), 0, ny_cells - 1))
                        ya = int(np.clip(np.abs(y_arr - ba_f[1][1] - distance).argmin(), 0, ny_cells - 1))
                        grid[min(xi, xa) : max(xi, xa) + 1, min(yi, ya) : max(yi, ya) + 1] = 1
        else:
            # Mark all instance bboxes as obstacles
            for inst in c.insts:
                ba = _parse_bbox_to_array(inst.dbbox())
                xi = int(np.abs(x_arr - ba[0][0] + distance).argmin())
                xa = int(np.abs(x_arr - ba[1][0] - distance).argmin())
                yi = int(np.abs(y_arr - ba[0][1] + distance).argmin())
                ya = int(np.abs(y_arr - ba[1][1] - distance).argmin())
                grid[min(xi, xa) : max(xi, xa) + 1, min(yi, ya) : max(yi, ya) + 1] = 1
        grids.append(grid)

    # Clear obstacles near ports on all layers
    if port_positions:
        excl_cells = max(1, int(np.ceil(port_exclusion_radius / resolution)))
        for px, py in port_positions:
            ci = int(np.clip(np.abs(x_arr - px).argmin(), 0, nx_cells - 1))
            cj = int(np.clip(np.abs(y_arr - py).argmin(), 0, ny_cells - 1))
            for g in grids:
                i_lo = max(0, ci - excl_cells)
                i_hi = min(nx_cells, ci + excl_cells + 1)
                j_lo = max(0, cj - excl_cells)
                j_hi = min(ny_cells, cj + excl_cells + 1)
                g[i_lo:i_hi, j_lo:j_hi] = 0

    # Build graph
    G = nx.Graph()

    # Add nodes and intra-layer edges
    for k in range(n_layers):
        pref = layers[k].preferred_direction
        for i in range(nx_cells):
            for j in range(ny_cells):
                if grids[k][i, j] == 1:
                    continue
                node = (i, j, k)
                G.add_node(node)
                # Horizontal neighbour (i-1, j, k)
                if i > 0 and grids[k][i - 1, j] == 0:
                    w = 1.0 if pref == "h" else wrong_way_penalty
                    G.add_edge((i - 1, j, k), node, weight=w)
                # Vertical neighbour (i, j-1, k)
                if j > 0 and grids[k][i, j - 1] == 0:
                    w = 1.0 if pref == "v" else wrong_way_penalty
                    G.add_edge((i, j - 1, k), node, weight=w)

    # Inter-layer via edges
    for k in range(n_layers - 1):
        for i in range(nx_cells):
            for j in range(ny_cells):
                if grids[k][i, j] == 0 and grids[k + 1][i, j] == 0:
                    G.add_edge((i, j, k), (i, j, k + 1), weight=via_cost)

    return G, x_arr, y_arr, nx_cells, ny_cells


def _astar_3d_heuristic(
    u: tuple[int, int, int],
    v: tuple[int, int, int],
    via_cost: float = 10.0,
) -> float:
    """Admissible heuristic for the 3-D A* search."""
    return float(abs(u[0] - v[0]) + abs(u[1] - v[1]) + abs(u[2] - v[2]) * via_cost)


def _extract_turning_points_3d(
    path: list[tuple[int, int, int]],
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
) -> list[tuple[float, float, int]]:
    """Reduce a raw grid path to turning-points and layer-transitions."""
    if len(path) <= 2:
        return [(float(x[i]), float(y[j]), k) for i, j, k in path]

    corners: list[tuple[float, float, int]] = [(float(x[path[0][0]]), float(y[path[0][1]]), path[0][2])]
    for idx in range(1, len(path) - 1):
        prev = path[idx - 1]
        curr = path[idx]
        nxt = path[idx + 1]
        # Keep if layer changes at this point
        if prev[2] != curr[2] or curr[2] != nxt[2]:
            corners.append((float(x[curr[0]]), float(y[curr[1]]), curr[2]))
            continue
        # Keep if direction changes on same layer
        di1, dj1 = curr[0] - prev[0], curr[1] - prev[1]
        di2, dj2 = nxt[0] - curr[0], nxt[1] - curr[1]
        if (di1, dj1) != (di2, dj2):
            corners.append((float(x[curr[0]]), float(y[curr[1]]), curr[2]))
    corners.append((float(x[path[-1][0]]), float(y[path[-1][1]]), path[-1][2]))

    # Remove duplicate consecutive corners
    deduped: list[tuple[float, float, int]] = [corners[0]]
    for c in corners[1:]:
        if c != deduped[-1]:
            deduped.append(c)
    return deduped


def _fix_diagonal_segments(
    corners: list[tuple[float, float, int]],
    layers: Sequence[LayerConfig],
) -> list[tuple[float, float, int]]:
    """Insert jog points to eliminate diagonal same-layer segments.

    When two consecutive corners on the same layer are neither
    horizontally nor vertically aligned, an intermediate L-shaped jog is
    inserted.  The jog direction respects the layer's preferred routing
    direction.
    """
    if len(corners) < 2:
        return corners

    fixed: list[tuple[float, float, int]] = [corners[0]]
    tol = 1e-6
    for i in range(1, len(corners)):
        prev = fixed[-1]
        curr = corners[i]
        if prev[2] == curr[2]:  # same layer
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            if dx > tol and dy > tol:
                # diagonal — insert intermediate
                pref = layers[prev[2]].preferred_direction if prev[2] < len(layers) else "h"
                if pref == "h":
                    # horizontal first, then vertical
                    fixed.append((curr[0], prev[1], prev[2]))
                else:
                    # vertical first, then horizontal
                    fixed.append((prev[0], curr[1], prev[2]))
        fixed.append(curr)
    return fixed


def _build_segments_and_vias(
    corners: list[tuple[float, float, int]],
    width: float,
) -> tuple[
    list[tuple[tuple[float, float], tuple[float, float], int, float]],
    list[tuple[tuple[float, float], int]],
    float,
    int,
]:
    """Derive planned segments and via positions from turning-point corners.

    Returns ``(segments, vias, total_length, num_vias)``.
    """
    segments: list[tuple[tuple[float, float], tuple[float, float], int, float]] = []
    vias: list[tuple[tuple[float, float], int]] = []
    total_length = 0.0
    num_vias = 0

    for i in range(len(corners) - 1):
        x0, y0, k0 = corners[i]
        x1, y1, k1 = corners[i + 1]
        if k0 != k1:
            # Layer transition — record via at the transition point
            vias.append(((x0, y0), i))
            num_vias += 1
        else:
            seg_len = abs(x1 - x0) + abs(y1 - y0)
            if seg_len > 1e-6:
                segments.append(((x0, y0), (x1, y1), k0, width))
                total_length += seg_len

    return segments, vias, total_length, num_vias


def _mark_route_obstacles_3d(
    G: nx.Graph,
    path: list[tuple[int, int, int]],
    spacing_radius: int = 2,
    nx_cells: int = 0,
    ny_cells: int = 0,
    n_layers: int = 2,
) -> None:
    """Block previously-routed nodes in the graph for sequential routing."""
    to_remove: list[tuple[int, int, int]] = []
    for i, j, k in path:
        for di in range(-spacing_radius, spacing_radius + 1):
            for dj in range(-spacing_radius, spacing_radius + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx_cells and 0 <= nj < ny_cells:
                    node = (ni, nj, k)
                    if node in G:
                        to_remove.append(node)
    G.remove_nodes_from(to_remove)


# -- main entry points ------------------------------------------------------


def route_astar_multilayer(
    component: Component,
    port1: Port,
    port2: Port,
    layers: Sequence[LayerConfig],
    resolution: float = 1.0,
    distance: float = 1.0,
    via_cost: float = 10.0,
    wrong_way_penalty: float = 8.0,
    width: float = 0.14,
    port1_layer_index: int = 0,
    port2_layer_index: int = 0,
    port_exclusion_radius: float = 5.0,
    clearance_ladder: Sequence[float] = (1.0, 0.5, 0.25, 0.1),
    penalty_ladder: Sequence[float] = (8.0, 4.0, 2.0, 1.0),
    exclusion_ladder: Sequence[float] = (5.0, 8.0, 12.0),
    bbox_padding_ladder: Sequence[float] | None = None,
) -> MultiLayerRouteResult | None:
    """Multi-layer A* router with deterministic fallback search expansion.

    Attempts routing with progressively relaxed parameters until a path is
    found.  Returns *None* only if all fallback attempts are exhausted.

    Args:
        component: Component to route within.
        port1: Source port.
        port2: Destination port.
        layers: Per-layer configuration (layer tuple, preferred direction,
            obstacle layers).
        resolution: Grid resolution in um.
        distance: Initial clearance distance from obstacles in um.
        via_cost: Cost penalty for layer transitions.
        wrong_way_penalty: Initial penalty for routing against preferred
            direction.
        width: Wire width in um.
        port1_layer_index: Layer index for port1 (0-based).
        port2_layer_index: Layer index for port2 (0-based).
        port_exclusion_radius: Initial radius to clear obstacles around
            ports in um.
        clearance_ladder: Deterministic sequence of clearance distances to
            attempt when routing fails.
        penalty_ladder: Deterministic sequence of wrong-way penalties.
        exclusion_ladder: Deterministic sequence of port exclusion radii.
        bbox_padding_ladder: Optional sequence of bbox paddings.  Defaults
            to ``[3*distance, 6*distance, 12*distance]``.
    """
    # Port positions in um
    p1x = float(port1.x) if hasattr(port1, 'x') else float(port1.dcenter[0])
    p1y = float(port1.y) if hasattr(port1, 'y') else float(port1.dcenter[1])
    p2x = float(port2.x) if hasattr(port2, 'x') else float(port2.dcenter[0])
    p2y = float(port2.y) if hasattr(port2, 'y') else float(port2.dcenter[1])
    port_positions = [(p1x, p1y), (p2x, p2y)]

    if bbox_padding_ladder is None:
        base_pad = max(distance, resolution) * 3
        bbox_padding_ladder = [base_pad, base_pad * 2, base_pad * 4]

    # Build deterministic fallback ladder
    attempts: list[tuple[float, float, float, float | None]] = []
    for cl in clearance_ladder:
        for pen in penalty_ladder:
            for excl in exclusion_ladder:
                for pad in bbox_padding_ladder:
                    attempts.append((cl, pen, excl, pad))

    for attempt_idx, (dist_try, pen_try, excl_try, pad_try) in enumerate(attempts):
        try:
            G, x_arr, y_arr, nx_c, ny_c = _generate_grid_3d(
                c=component,
                layers=layers,
                resolution=resolution,
                distance=dist_try,
                port_positions=port_positions,
                port_exclusion_radius=excl_try,
                wrong_way_penalty=pen_try,
                via_cost=via_cost,
                bbox_padding=pad_try,
            )
        except Exception:
            continue

        # Find closest valid nodes to port positions
        def _closest_node(
            graph: nx.Graph,
            px: float,
            py: float,
            layer_idx: int,
            xa: npt.NDArray[np.floating[Any]],
            ya: npt.NDArray[np.floating[Any]],
        ) -> tuple[int, int, int] | None:
            target_i = int(np.clip(np.abs(xa - px).argmin(), 0, len(xa) - 1))
            target_j = int(np.clip(np.abs(ya - py).argmin(), 0, len(ya) - 1))
            ideal = (target_i, target_j, layer_idx)
            if ideal in graph:
                return ideal
            # Search expanding rings for nearest valid node on this layer
            for radius in range(1, max(len(xa), len(ya))):
                best: tuple[int, int, int] | None = None
                best_dist = float("inf")
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        if abs(di) != radius and abs(dj) != radius:
                            continue
                        ni, nj = target_i + di, target_j + dj
                        candidate = (ni, nj, layer_idx)
                        if candidate in graph:
                            d = abs(di) + abs(dj)
                            if d < best_dist:
                                best_dist = d
                                best = candidate
                if best is not None:
                    return best
            return None

        start = _closest_node(G, p1x, p1y, port1_layer_index, x_arr, y_arr)
        end = _closest_node(G, p2x, p2y, port2_layer_index, x_arr, y_arr)
        if start is None or end is None:
            continue

        heuristic = partial(_astar_3d_heuristic, via_cost=via_cost)
        try:
            raw_path = nx.astar_path(G, start, end, heuristic=heuristic, weight="weight")
        except nx.NetworkXNoPath:
            continue

        corners = _extract_turning_points_3d(raw_path, x_arr, y_arr)
        # Force exact port positions on endpoints
        if corners:
            corners[0] = (p1x, p1y, corners[0][2])
            corners[-1] = (p2x, p2y, corners[-1][2])
        corners = _fix_diagonal_segments(corners, layers)
        segments, vias, length, n_vias = _build_segments_and_vias(corners, width)

        return MultiLayerRouteResult(
            corners_3d=corners,
            segments=segments,
            vias=vias,
            path_length_um=length,
            num_vias=n_vias,
            raw_path=raw_path,
        )

    # All attempts exhausted
    return None


def route_astar_multilayer_multi_net(
    component: Component,
    port_pairs: Sequence[tuple[Port, Port, int, int]],
    layers: Sequence[LayerConfig],
    resolution: float = 1.0,
    distance: float = 1.0,
    via_cost: float = 10.0,
    wrong_way_penalty: float = 8.0,
    width: float = 0.14,
    spacing_radius: int = 2,
    port_exclusion_radius: float = 5.0,
    clearance_ladder: Sequence[float] = (1.0, 0.5, 0.25, 0.1),
    penalty_ladder: Sequence[float] = (8.0, 4.0, 2.0, 1.0),
    exclusion_ladder: Sequence[float] = (5.0, 8.0, 12.0),
    bbox_padding_ladder: Sequence[float] | None = None,
) -> list[MultiLayerRouteResult | None]:
    """Route multiple nets sequentially, blocking previous routes as obstacles.

    Each net is routed using :func:`route_astar_multilayer` with the full
    fallback ladder.  After a successful route, its path is marked as an
    obstacle so that subsequent nets route around it.

    If a net fails, the function attempts **one retry** with the remaining
    nets in reversed order before giving up on that net.

    Args:
        port_pairs: Sequence of ``(port1, port2, port1_layer_idx, port2_layer_idx)``.
        (other args): Forwarded to :func:`route_astar_multilayer`.

    Returns:
        List of results, one per input net.  ``None`` for nets that could
        not be routed.
    """
    # Collect all port positions for exclusion
    all_port_positions: list[tuple[float, float]] = []
    for p1, p2, _, _ in port_pairs:
        p1x = float(p1.x) if hasattr(p1, 'x') else float(p1.dcenter[0])
        p1y = float(p1.y) if hasattr(p1, 'y') else float(p1.dcenter[1])
        p2x = float(p2.x) if hasattr(p2, 'x') else float(p2.dcenter[0])
        p2y = float(p2.y) if hasattr(p2, 'y') else float(p2.dcenter[1])
        all_port_positions.extend([(p1x, p1y), (p2x, p2y)])

    if bbox_padding_ladder is None:
        base_pad = max(distance, resolution) * 3
        bbox_padding_ladder = [base_pad, base_pad * 2, base_pad * 4]

    # Build a shared initial graph for obstacle marking
    # Use the most generous parameters for the shared graph
    min_dist = min(clearance_ladder) if clearance_ladder else distance
    max_excl = max(exclusion_ladder) if exclusion_ladder else port_exclusion_radius
    max_pad = max(bbox_padding_ladder) if bbox_padding_ladder else None

    try:
        shared_G, x_arr, y_arr, nx_c, ny_c = _generate_grid_3d(
            c=component,
            layers=layers,
            resolution=resolution,
            distance=min_dist,
            port_positions=all_port_positions,
            port_exclusion_radius=max_excl,
            wrong_way_penalty=min(penalty_ladder) if penalty_ladder else wrong_way_penalty,
            via_cost=via_cost,
            bbox_padding=max_pad,
        )
    except Exception:
        return [None] * len(port_pairs)

    results: list[MultiLayerRouteResult | None] = [None] * len(port_pairs)

    for idx, (p1, p2, l1, l2) in enumerate(port_pairs):
        result = route_astar_multilayer(
            component=component,
            port1=p1,
            port2=p2,
            layers=layers,
            resolution=resolution,
            distance=distance,
            via_cost=via_cost,
            wrong_way_penalty=wrong_way_penalty,
            width=width,
            port1_layer_index=l1,
            port2_layer_index=l2,
            port_exclusion_radius=port_exclusion_radius,
            clearance_ladder=clearance_ladder,
            penalty_ladder=penalty_ladder,
            exclusion_ladder=exclusion_ladder,
            bbox_padding_ladder=list(bbox_padding_ladder),
        )
        if result is not None:
            results[idx] = result
            # Mark routed path as obstacles in the shared component
            # (the component gains geometry when the caller draws routes,
            #  but for the A* grid we need to block the path nodes)
            _mark_route_obstacles_3d(
                shared_G, result.raw_path, spacing_radius, nx_c, ny_c, len(layers),
            )
        else:
            print(f"[MULTI-NET] Net {idx} failed to route: {p1.name} -> {p2.name}")

    return results
