"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component

The CrossSection defines the layer numbers, widths and offsetts

Based on phidl.path
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import phidl.path as path
from phidl.device_layout import Path, _simplify
from phidl.path import smooth as smooth_phidl

from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection
from gdsfactory.hash_points import hash_points
from gdsfactory.tech import LAYER
from gdsfactory.types import (
    Coordinates,
    CrossSectionOrFactory,
    Float2,
    Layer,
    Number,
    PathFactory,
)


def _sinusoidal_transition(y1, y2):
    dx = y2 - y1
    return lambda t: y1 + (1 - np.cos(np.pi * t)) / 2 * dx


def _linear_transition(y1, y2):
    dx = y2 - y1
    return lambda t: y1 + t * dx


def transition(
    cross_section1: CrossSection, cross_section2: CrossSection, width_type: str = "sine"
) -> CrossSection:
    """Creates a CrossSection that smoothly transitions between two input
    CrossSections. Only cross-sectional elements that have the `name` (as in
    X.add(..., name = 'wg') ) parameter specified in both input CrosSections
    will be created. Port names will be cloned from the input CrossSections in
    reverse.
    adapted from phidl.path

    Args:
        cross_section1: First input CrossSection
        cross_section2: Second input CrossSection
        width_type: sine or linear
          Sets the type of width transition used if any widths are different
          between the two input CrossSections.

    Returns A smoothly-transitioning CrossSection
    """

    X1 = cross_section1
    X2 = cross_section2
    Xtrans = CrossSection()

    if not X1.aliases or not X2.aliases:
        raise ValueError(
            """transition() found no named sections in one
        or both inputs (cross_section1/cross_section2)."""
        )

    layers1 = {section["layer"] for section in X1.sections}
    layers2 = {section["layer"] for section in X2.sections}

    has_common_layers = True if layers1.intersection(layers2) else False
    if not has_common_layers:
        raise ValueError(
            f"transition() found no common layers X1 {layers1} and X2 {layers2}"
        )

    for alias in X1.aliases.keys():
        if alias in X2.aliases:

            offset1 = X1[alias]["offset"]
            offset2 = X2[alias]["offset"]
            width1 = X1[alias]["width"]
            width2 = X2[alias]["width"]

            if callable(offset1):
                offset1 = offset1(1)
            if callable(offset2):
                offset2 = offset2(0)
            if callable(width1):
                width1 = width1(1)
            if callable(width2):
                width2 = width2(0)

            offset_fun = _sinusoidal_transition(offset1, offset2)

            if width_type == "sine":
                width_fun = _sinusoidal_transition(width1, width2)
            elif width_type == "linear":
                width_fun = _linear_transition(width1, width2)
            else:
                raise ValueError(
                    "transition() width_type "
                    + "argument must be one of {'sine','linear'}"
                )

            if X1[alias]["layer"] != X2[alias]["layer"]:
                hidden = True
                layer = (X1[alias]["layer"], X2[alias]["layer"])
            else:
                hidden = False
                layer = X1[alias]["layer"]
            Xtrans.add(
                width=width_fun,
                offset=offset_fun,
                layer=layer,
                ports=(X2[alias]["ports"][0], X1[alias]["ports"][1]),
                port_types=(X2[alias]["port_types"][0], X1[alias]["port_types"][1]),
                name=alias,
                hidden=hidden,
            )

    Xtrans.cross_sections = (X1, X2)
    return Xtrans


def extrude(
    p: Path,
    cross_section: Optional[CrossSectionOrFactory] = None,
    layer: Optional[Layer] = None,
    width: Optional[float] = None,
    widths: Optional[Float2] = None,
    simplify: Optional[float] = None,
) -> Component:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component

    The CrossSection defines the layer numbers, widths and offsetts

    adapted from phidl.path

    Args:
        p: a path is a list of points (arc, straight, euler)
        cross_section: to extrude
        layer:
        width:
        widths: tuple of starting and end width
        simplify: Tolerance value for the simplification algorithm.
          All points that can be removed without changing the resulting
          polygon by more than the value listed here will be removed.
    """
    if cross_section is None and layer is None:
        raise ValueError("CrossSection or layer needed")

    if cross_section is not None and layer is not None:
        raise ValueError("Define only CrossSection or layer")

    if layer is not None and width is None and widths is None:
        raise ValueError("Need to define layer width or widths")
    elif width:
        cross_section = CrossSection()
        cross_section.add(width=width, layer=layer)

    elif widths:
        cross_section = CrossSection()
        cross_section.add(width=_linear_transition(widths[0], widths[1]), layer=layer)

    xsection_points = []
    c = Component()

    cross_section = cross_section() if callable(cross_section) else cross_section
    snap_to_grid = cross_section.info.get("snap_to_grid", None)

    for section in cross_section.sections:
        width = section["width"]
        offset = section["offset"]
        layer = section["layer"]
        ports = section["ports"]
        port_types = section["port_types"]
        hidden = section["hidden"]

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

        points1 = p._centerpoint_offset_curve(
            points,
            offset_distance=offset + width / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        points2 = p._centerpoint_offset_curve(
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

        if snap_to_grid:
            snap_to_grid_nm = snap_to_grid * 1e3
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

        layers = layer if hidden else [layer, layer]
        if not hidden:
            c.add_polygon(points, layer=layer)
        # Add ports if they were specified
        if ports[0] is not None:
            orientation = (p.start_angle + 180) % 360
            _width = width if np.isscalar(width) else width[0]
            new_port = c.add_port(
                name=ports[0],
                layer=layers[0],
                port_type=port_types[0],
                width=_width,
                orientation=orientation,
                cross_section=cross_section.cross_sections[0]
                if hasattr(cross_section, "cross_sections")
                else cross_section,
            )
            new_port.endpoints = (points1[0], points2[0])
        if ports[1] is not None:
            orientation = (p.end_angle + 180) % 360
            _width = width if np.isscalar(width) else width[-1]
            new_port = c.add_port(
                name=ports[1],
                layer=layers[1],
                port_type=port_types[1],
                width=_width,
                orientation=orientation,
                cross_section=cross_section.cross_sections[1]
                if hasattr(cross_section, "cross_sections")
                else cross_section,
            )
            new_port.endpoints = (points2[-1], points1[-1])

    points = np.concatenate((p.points, np.array(xsection_points)))
    c.name = f"path_{hash_points(points)[:26]}"
    # c.path = path
    # c.cross_section = cross_section
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
    if length < 0:
        raise ValueError(f"length = {length} needs to be > 0")
    return path.straight(length=length, num_pts=npoints)


def smooth(
    points: Coordinates,
    radius: float = 4.0,
    bend: PathFactory = euler,
    **kwargs,
) -> Path:
    """Returns a smooth Path from a series of waypoints. Corners will be rounded
    using `bend` and any additional key word arguments (for example,
    `use_eff = True` for `bend = gf.path.euler`)

    Args:
        points: array-like[N][2] List of waypoints for the path to follow
        radius: radius of curvature, passed to `bend`
        bend: bend function to round corners
        **kwargs: Extra keyword arguments that will be passed to `bend`
    """
    return smooth_phidl(points=points, radius=radius, corner_fun=bend, **kwargs)


__all__ = ["straight", "euler", "arc", "extrude", "path", "transition", "smooth"]

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

    c = extrude(P, X, simplify=5e-3)
    # c = gf.add_pins(c)
    # c << gf.components.bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    print(c.ports["in"].layer)
    c.show()
