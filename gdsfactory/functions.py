from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import kfactory as kf
import numpy as np
from numpy import cos, float64, ndarray, sin

import gdsfactory as gf

if TYPE_CHECKING:
    from gdsfactory.component import Component, Instance
    from gdsfactory.typings import LayerSpecs

RAD2DEG = 180.0 / np.pi
DEG2RAD = 1 / RAD2DEG


def move_port_to_zero(
    component: Component, port_name: str = "o1", mirror: bool = False
) -> gf.Component:
    """Return a container that contains a reference to the original component.

    The new component has port_name in (0, 0).

    Args:
        component: to move the port to (0, 0).
        port_name: to move to (0, 0).
        mirror: if True, mirrors the component.
    """
    port_names = [p.name for p in component.ports]
    if port_name not in port_names:
        raise ValueError(f"port_name = {port_name!r} not in {port_names}")

    c = gf.Component()
    ref = c << component
    if mirror:
        ref.dmirror()

    movement = np.array(ref.ports[port_name].dcenter)
    ref.dmove(-movement)
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


def get_layers(component: Component) -> list[tuple[int, int]]:
    """Returns the layers of a component.

    Args:
        component: to get the layers from.
    """
    return [
        (info.layer, info.datatype)
        for info in component.kcl.layer_infos()
        if not component.bbox(component.kcl.layer(info)).empty()
    ]


def extract(
    component,
    layers: LayerSpecs,
    recursive: bool = True,
) -> Component:
    """Extracts a list of layers and adds them to a new Component.

    Args:
        component: to extract the layers from.
        layers: list of layers to extract.
        recursive: if True, extracts layers recursively and returns a flattened Component.
    """
    from gdsfactory.pdk import get_layer_tuple

    c = component.dup()
    if recursive:
        c.flatten()

    layer_tuples = [get_layer_tuple(layer) for layer in layers]
    component_layers = get_layers(c)

    for layer_tuple in layer_tuples:
        if layer_tuple not in component_layers:
            warnings.warn(
                f"Layer {layer_tuple} not found in component {component.name!r} layers."
            )

    for layer_tuple in component_layers:
        if layer_tuple not in layer_tuples:
            layer_index = c.kcl.layer(*layer_tuple)
            c.shapes(layer_index).clear()

    return c


def move_to_center(component: Component, dx: float = 0, dy: float = 0) -> gf.Component:
    """Moves the component to the center of the bounding box."""
    c = component
    c.transform(
        gf.kdb.DTrans(-c.dbbox().center().x + dx, -c.dbbox().center().y + dy),
        no_warn=True,
    )
    return c


def get_polygons(
    component_or_instance: Component | Instance,
    merge: bool = False,
    by: Literal["index"] | Literal["name"] | Literal["tuple"] = "index",
    layers: LayerSpecs | None = None,
) -> dict[tuple[int, int] | str | int, list[kf.kdb.Polygon]]:
    """Returns a dict of Polygons per layer.

    Args:
        component_or_instance: to extract the polygons.
        merge: if True, merges the polygons.
        by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple').
        layers: list of layer specs to extract the polygons from. If None, extracts all layers.
    """
    from gdsfactory.pdk import get_layer, get_layer_name, get_layer_tuple

    if by == "index":
        get_key = get_layer
    elif by == "name":
        get_key = get_layer_name
    elif by == "tuple":
        get_key = get_layer_tuple

    else:
        raise ValueError("argument 'by' should be 'index' | 'name' | 'tuple'")

    polygons = {}

    c = component_or_instance
    if layers is None:
        layers = [
            (info.layer, info.datatype)
            for info in c.kcl.layer_infos()
            if not c.bbox(c.kcl.layer(info)).empty()
        ]

    layer_indexes = [gf.get_layer(layer) for layer in layers]

    for layer_index in layer_indexes:
        layer_key = get_key(layer_index)
        if isinstance(component_or_instance, gf.Component):
            r = gf.kdb.Region(c.begin_shapes_rec(layer_index))
        else:
            r = kf.kdb.Region(c.cell.begin_shapes_rec(layer_index)).transformed(
                c.cplx_trans
            )
        if layer_key not in polygons:
            polygons[layer_key] = []
        if merge:
            r.merge()
        for p in r.each():
            polygons[layer_key].append(p)
    return polygons


def get_polygons_points(
    component_or_instance: Component | Instance,
    merge: bool = False,
    scale: float | None = None,
    by: Literal["index"] | Literal["name"] | Literal["tuple"] = "index",
    layers: LayerSpecs | None = None,
) -> dict[int | str | tuple[int, int], list[tuple[float, float]]]:
    """Returns a dict with list of points per layer.

    Args:
        component_or_instance: to extract the polygons.
        merge: if True, merges the polygons.
        scale: if True, scales the points.
        by: the format of the resulting keys in the dictionary ('index', 'name', 'tuple').
        layers: list of layer specs to extract the polygons from. If None, extracts all layers.
    """
    polygons_dict = get_polygons(
        component_or_instance=component_or_instance, merge=merge, by=by, layers=layers
    )
    polygons_points = {}
    for layer, polygons in polygons_dict.items():
        all_points = []
        for polygon in polygons:
            if scale:
                points = np.array(
                    [
                        (point.x * scale, point.y * scale)
                        for point in polygon.to_simple_polygon()
                        .to_dtype(component_or_instance.kcl.dbu)
                        .each_point()
                    ]
                )
            else:
                points = np.array(
                    [
                        (point.x, point.y)
                        for point in polygon.to_simple_polygon()
                        .to_dtype(component_or_instance.kcl.dbu)
                        .each_point()
                    ]
                )
            all_points.append(points)
        polygons_points[layer] = all_points
    return polygons_points


def get_point_inside(component_or_instance: Component | Instance, layer) -> np.ndarray:
    """Returns a point inside the component or instance.

    Args:
        component_or_instance: to find a point inside.
        layer: to find a point inside.
    """
    layer = gf.get_layer(layer)
    return get_polygons_points(component_or_instance, layers=[layer])[layer][0][0]


def sign_shape(pts: ndarray) -> float64:
    pts2 = np.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return np.sign((dx * y).sum())


def area(pts: ndarray) -> float64:
    """Returns the area."""
    pts2 = np.roll(pts, 1, axis=0)
    dx = pts2[:, 0] - pts[:, 0]
    y = pts2[:, 1] + pts[:, 1]
    return (dx * y).sum() / 2


def centered_diff(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / 2
    return d[1:-1]


def centered_diff2(a: ndarray) -> ndarray:
    d = (np.roll(a, -1, axis=0) - a) - (a - np.roll(a, 1, axis=0))
    return d[1:-1]


def curvature(points: ndarray, t: ndarray) -> ndarray:
    """Args are the points and the tangents at each point.

        points : numpy.array shape (n, 2)
        t: numpy.array of size n

    Return:
        The curvature at each point.

    Computes the curvature at every point excluding the first and last point.

    For a planar curve parametrized as P(t) = (x(t), y(t)), the curvature is given
    by (x' y'' - x'' y' ) / (x' **2 + y' **2)**(3/2)

    """
    # Use centered difference for derivative
    dt = centered_diff(t)
    dp = centered_diff(points)
    dp2 = centered_diff2(points)

    dx = dp[:, 0] / dt
    dy = dp[:, 1] / dt

    dx2 = dp2[:, 0] / dt**2
    dy2 = dp2[:, 1] / dt**2

    return (dx * dy2 - dx2 * dy) / (dx**2 + dy**2) ** (3 / 2)


def radius_of_curvature(points, t):
    return 1 / curvature(points, t)


def path_length(points: ndarray) -> float64:
    """Returns: The path length.

    Args:
        points: With shape (N, 2) representing N points with coordinates x, y.
    """
    dpts = points[1:, :] - points[:-1, :]
    _d = dpts**2
    return np.sum(np.sqrt(_d[:, 0] + _d[:, 1]))


def snap_angle(a: float64) -> int:
    """Returns angle snapped along manhattan angle (0, 90, 180, 270).

    a: angle in deg
    Return angle snapped along manhattan angle
    """
    a = a % 360
    if -45 < a < 45:
        return 0
    elif 45 < a < 135:
        return 90
    elif 135 < a < 225:
        return 180
    elif 225 < a < 315:
        return 270
    else:
        return 0


def angles_rad(pts: ndarray) -> ndarray:
    """Returns the angles (radians) of the connection between each point and the next."""
    _pts = np.roll(pts, -1, 0)
    return np.arctan2(_pts[:, 1] - pts[:, 1], _pts[:, 0] - pts[:, 0])


def angles_deg(pts: ndarray) -> ndarray:
    """Returns the angles (degrees) of the connection between each point and the next."""
    return angles_rad(pts) * RAD2DEG


def extrude_path(
    points: ndarray,
    width: float,
    with_manhattan_facing_angles: bool = True,
    spike_length: float64 | int | float = 0,
    start_angle: int | None = None,
    end_angle: int | None = None,
    grid: float | None = None,
) -> ndarray:
    """Deprecated. Use gdsfactory.path.Path.extrude() instead.

    Extrude a path of `width` along a curve defined by `points`.

    Args:
        points: numpy 2D array of shape (N, 2).
        width: of the path to extrude.
        with_manhattan_facing_angles: snaps to manhattan angles.
        spike_length: in um.
        start_angle: in degrees.
        end_angle: in degrees.
        grid: in um.

    Returns:
        numpy 2D array of shape (2*N, 2).
    """
    grid = grid or gf.kcl.dbu

    if isinstance(points, list):
        points = np.stack([(p[0], p[1]) for p in points], axis=0)

    a = angles_deg(points)
    if with_manhattan_facing_angles:
        _start_angle = snap_angle(a[0] + 180)
        _end_angle = snap_angle(a[-2])
    else:
        _start_angle = a[0] + 180
        _end_angle = a[-2]

    start_angle = start_angle if start_angle is not None else _start_angle
    end_angle = end_angle if end_angle is not None else _end_angle

    a2 = angles_rad(points) * 0.5
    a1 = np.roll(a2, 1)

    a2[-1] = end_angle * DEG2RAD - a2[-2]
    a1[0] = start_angle * DEG2RAD - a1[1]

    a_plus = a2 + a1
    cos_a_min = np.cos(a2 - a1)
    offsets = np.column_stack((-sin(a_plus) / cos_a_min, cos(a_plus) / cos_a_min)) * (
        0.5 * width
    )

    points_back = np.flipud(points - offsets)
    if spike_length != 0:
        d = spike_length
        a_start = start_angle * DEG2RAD
        a_end = end_angle * DEG2RAD
        p_start_spike = points[0] + d * np.array([[cos(a_start), sin(a_start)]])
        p_end_spike = points[-1] + d * np.array([[cos(a_end), sin(a_end)]])

        pts = np.vstack((p_start_spike, points + offsets, p_end_spike, points_back))
    else:
        pts = np.vstack((points + offsets, points_back))

    return np.round(pts / grid) * grid


def trim(
    component: Component,
    domain: list[tuple[float, float]],
    flatten: bool = False,
) -> gf.Component:
    """Trim a component by another geometry, preserving the component's layers and ports.

    Useful to get a smaller component from a larger one for simulation.

    Args:
        component: Component(/Reference).
        domain: list of array-like[N][2] representing the boundary of the component to keep.
        flatten: if True, flattens the component.

    Returns: New component with layers (and possibly ports) of the component restricted to the domain.

    .. plot::
      :include-source:

      import gdsfactory as gf
      c = gf.components.straight_pin(length=10)
      trimmed_c = gf.functions.trim(component=c, domain=[[0, -5], [0, 5], [5, 5], [5, -5]])
      trimmed_c.plot()
    """
    dummy = gf.Component()
    dummy.add_polygon(domain, layer=(1, 0))
    dbbox = dummy.dbbox()
    left, bottom, right, top = dbbox.left, dbbox.bottom, dbbox.right, dbbox.top
    component.trim(left=left, right=right, bottom=bottom, top=top, flatten=flatten)
    return component


if __name__ == "__main__":
    c = gf.components.ring_single()
    c = gf.components.straight_pin(length=11, taper=None)
    # c.trim(left=0, right=10, bottom=0, top=10)
    c = trim(c, domain=[[0, 0], [0, 10], [10, 10], [10, 0]])
    c.show()

    # c = gf.c.rectangle(size=(10, 10), centered=True)
    # p = c.get_polygons(layers=("WG",), by="tuple")
    # print(list(p.keys())[0])

    # c = gf.Component()
    # ref = c << gf.components.mzi_lattice()
    # ref.dmovey(15)
    # p = get_polygons_points(ref)
    # p = get_point_inside(ref, layer=(1, 0))
    # c.add_label(text="hello", position=p)
    # c.show()
    # c = move_port_to_zero(c, port_name="e4")
    # c.show()
