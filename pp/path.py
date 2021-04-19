"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component

The CrossSection defines the layer numbers, widths and offsetts

Based on phidl.path
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import phidl.path as path
from phidl.device_layout import CrossSection, Path, _simplify
from phidl.path import smooth as smooth_phidl

from pp.component import Component
from pp.hash_points import hash_points
from pp.layers import LAYER
from pp.port import auto_rename_ports
from pp.transition import transition
from pp.types import Coordinates, Number, PathFactory


def component(
    p: Path,
    cross_section: CrossSection,
    simplify: Optional[float] = None,
    snap_to_grid_nm: Optional[int] = None,
    rename_ports: bool = True,
) -> Component:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component

    The CrossSection defines the layer numbers, widths and offsetts

    Args:
        p: a path is a list of points (arc, straight, euler)
        cross_section: extrudes a cross_section over a path
        simplify: Tolerance value for the simplification algorithm.
            All points that can be removed without changing the resulting
            polygon by more than the value listed here will be removed.
        snap_to_grid_nm: optionally snap to any design grid (nm)
        rename_ports: rename ports
    """
    cross_section = cross_section() if callable(cross_section) else cross_section
    xsection_points = []

    c = Component()

    for section in cross_section.sections:
        width = section["width"]
        offset = section["offset"]
        layer = section["layer"]
        ports = section["ports"]

        if isinstance(width, (int, float)) and isinstance(offset, (int, float)):
            xsection_points.append([width, offset])
        if isinstance(layer, int):
            layer = (layer, 0)
        if (
            isinstance(layer, Iterable)
            and len(layer) == 2
            and isinstance(layer[0], int)
            and isinstance(layer[1], int)
        ):
            xsection_points.append([layer[0], layer[1]])

        if callable(offset):
            P_offset = p.copy()
            P_offset.offset(offset)
            points = P_offset.points
            start_angle = P_offset.start_angle
            end_angle = P_offset.end_angle
            offset = 0
        else:
            points = p.points
            start_angle = p.start_angle
            end_angle = p.end_angle

        if callable(width):
            # Compute lengths
            dx = np.diff(p.points[:, 0])
            dy = np.diff(p.points[:, 1])
            lengths = np.cumsum(np.sqrt((dx) ** 2 + (dy) ** 2))
            lengths = np.concatenate([[0], lengths])
            width = width(lengths / lengths[-1])
        else:
            pass

        points1 = p._parametric_offset_curve(
            points,
            offset_distance=offset + width / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        points2 = p._parametric_offset_curve(
            points,
            offset_distance=offset - width / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        # Simplify lines using the Ramer–Douglas–Peucker algorithm
        if isinstance(simplify, bool):
            raise ValueError(
                "[PHIDL] the simplify argument must be a number (e.g. 1e-3) or None"
            )
        if simplify is not None:
            points1 = _simplify(points1, tolerance=simplify)
            points2 = _simplify(points2, tolerance=simplify)

        if snap_to_grid_nm:
            points1 = (
                snap_to_grid_nm
                * np.round(np.array(points1) * 1e3 / snap_to_grid_nm)
                / 1e3
            )
            points2 = (
                snap_to_grid_nm
                * np.round(np.array(points2) * 1e3 / snap_to_grid_nm)
                / 1e3
            )

        # Join points together
        points = np.concatenate([points1, points2[::-1, :]])

        # Combine the offset-lines into a polygon and union if join_after == True
        # if join_after == True: # Use clipper to perform a union operation
        #     points = np.array(clipper.offset([points], 0, 'miter', 2, int(1/simplify), 0)[0])
        # print(points)

        c.add_polygon(points, layer=layer)

        # Add ports if they were specified
        if ports[0] is not None:
            new_port = c.add_port(name=ports[0], layer=layer)
            new_port.endpoints = (points1[0], points2[0])
        if ports[1] is not None:
            new_port = c.add_port(name=ports[1], layer=layer)
            new_port.endpoints = (points2[-1], points1[-1])

    points = np.concatenate((p.points, np.array(xsection_points)))
    c.name = f"path_{hash_points(points)}"
    # c.path = path
    # c.cross_section = cross_section
    if rename_ports:
        auto_rename_ports(c)
    return c


def arc(radius: Number = 10, angle: Number = 90, npoints: int = 720) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature
        angle: total angle of the curve
        npoints: Number of points used per 360 degrees

    """
    return path.arc(radius=radius, angle=angle, num_pts=npoints)


def euler(
    radius: Number = 10,
    angle: Number = 90,
    p: float = 1,
    use_eff: bool = False,
    npoints: int = 720,
) -> Path:
    """Returns an euler bend that adiabatically transitions from straight to curved.
    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    Args:
        radius: minimum radius of curvature
        angle: total angle of the curve
        p: Proportion of the curve that is an Euler curve
        use_eff: If False: `radius` is the minimum radius of curvature of the bend
            If True: The curve will be scaled such that the endpoints match an arc
            with parameters `radius` and `angle`
        npoints: Number of points used per 360 degrees

    """
    return path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, num_pts=npoints)


def straight(length: Number = 10, npoints: int = 2) -> Path:
    """Returns a straight path

    For transitions you should increase have at least 100 points

    Args:
        length: of straight
        npoints: number of points
    """
    return path.straight(length=length, num_pts=npoints)


def smooth(
    points: Coordinates,
    radius: float = 4.0,
    bend_path_function: PathFactory = euler,
    **kwargs,
) -> Path:
    """Returns a smooth Path from a series of waypoints. Corners will be rounded
    using `bend_path_function` and any additional key word arguments (for example,
    `use_eff = True` when `bend_path_function = pp..path.euler`)

    Args:
        points: array-like[N][2] List of waypoints for the path to follow
        radius: radius of curvature, passed to `bend_path_function`
        bend_path_function: function that controls how the corners are rounded.
        **kwargs: Extra keyword arguments that will be passed to `bend_path_function`
    """
    return smooth_phidl(
        points=points, radius=radius, corner_fun=bend_path_function, **kwargs
    )


__all__ = ["straight", "euler", "arc", "component", "path", "transition", "smooth"]

if __name__ == "__main__":

    P = euler(radius=10, use_eff=True)
    # P = euler()
    # P = Path()
    # P.append(straight(length=5))
    # P.append(path.arc(radius=10, angle=90))
    # P.append(path.spiral())

    # Create a blank CrossSection
    X = CrossSection()
    X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])

    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # Combine the Path and the CrossSection

    c = component(P, X, simplify=5e-3, snap_to_grid_nm=5)
    # c = pp.add_pins(c)
    # c << pp.components.bend_euler(radius=10)
    # c << pp.components.bend_circular(radius=10)
    print(c.ports["W0"].layer)
    c.show()
