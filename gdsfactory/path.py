"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component

The CrossSection defines the layer numbers, widths and offsetts

Based on phidl.path

"""

import warnings
from collections.abc import Iterable
from typing import Optional

import numpy as np
import shapely.ops
from phidl import path
from phidl.device_layout import Path as PathPhidl
from phidl.path import smooth as smooth_phidl

from gdsfactory import snap
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection, Section, Transition
from gdsfactory.port import Port
from gdsfactory.types import (
    Coordinates,
    CrossSectionSpec,
    Float2,
    LayerSpec,
    PathFactory,
    WidthTypes,
)


def _simplify(points, tolerance):
    import shapely.geometry as sg

    ls = sg.LineString(points)
    ls_simple = ls.simplify(tolerance=tolerance)
    return np.asarray(ls_simple.coords)


class Path(PathPhidl):
    """Path object for smooth Paths.

    You can extrude a Path with a CrossSection to create a Component.

    Parameters:
        path: array-like[N][2], Path, or list of Paths.
            Points or Paths to append() initially.

    """

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        """Pydantic path validator."""
        assert isinstance(v, PathPhidl), f"TypeError, Got {type(v)}, expecting Path"
        return v

    def to_dict(self):
        return self.hash_geometry()

    def plot(self):
        """Plot path in matplotlib."""
        from gdsfactory.quickplotter import quickplot

        return quickplot(self)

    def extrude(
        self,
        cross_section: Optional[CrossSectionSpec] = None,
        layer: Optional[LayerSpec] = None,
        width: Optional[float] = None,
        widths: Optional[Float2] = None,
        simplify: Optional[float] = None,
        shear_angle_start: Optional[float] = None,
        shear_angle_end: Optional[float] = None,
    ) -> Component:
        """Returns Component by extruding a Path with a CrossSection.

        A path can be extruded using any CrossSection returning a Component
        The CrossSection defines the layer numbers, widths and offsetts.

        Args:
            p: a path is a list of points (arc, straight, euler).
            cross_section: to extrude.
            layer: optional layer.
            width: optional width in um.
            widths: tuple of starting and end width for a linear taper.
            simplify: Tolerance value for the simplification algorithm.
              All points that can be removed without changing the resulting
              polygon by more than the value listed here will be removed.
            shear_angle_start: an optional angle to shear the starting face by (in degrees).
            shear_angle_end: an optional angle to shear the ending face by (in degrees).
        """
        return extrude(
            p=self,
            cross_section=cross_section,
            layer=layer,
            width=width,
            widths=widths,
            simplify=simplify,
            shear_angle_start=shear_angle_start,
            shear_angle_end=shear_angle_end,
        )

    def copy(self):
        """Returns a copy of the Path."""
        p = Path()
        p.info = self.info.copy()
        p.points = np.array(self.points)
        p.start_angle = self.start_angle
        p.end_angle = self.end_angle
        return p

    def from_phidl(self, path_phidl: PathPhidl):
        """Returns a path from a phidl path."""
        p = Path()
        p.info = path_phidl.info.copy()
        p.points = np.array(path_phidl.points)
        p.start_angle = path_phidl.start_angle
        p.end_angle = path_phidl.end_angle
        return p


def _sinusoidal_transition(y1, y2):
    dy = y2 - y1

    def sine(t):
        return y1 + (1 - np.cos(np.pi * t)) / 2 * dy

    return sine


def _linear_transition(y1, y2):
    dy = y2 - y1

    def linear(t):
        return y1 + t * dy

    return linear


def transition_exponential(y1, y2, exp=0.5):
    """Returns the function for an exponential transition.

    Args:
        y1: start width in um.
        y2: end width in um.
        exp: exponent.

    """
    return lambda t: y1 + (y2 - y1) * t**exp


def transition(
    cross_section1: CrossSection,
    cross_section2: CrossSection,
    width_type: WidthTypes = "sine",
) -> Transition:
    """Returns a smoothly-transitioning between two CrossSections.

    Only cross-sectional elements that have the `name` (as in X.add(..., name = 'wg') )
    parameter specified in both input CrosSections will be created.
    Port names will be cloned from the input CrossSections in reverse.

    Args:
        cross_section1: First CrossSection.
        cross_section2: Second CrossSection.
        width_type: sine or linear.
          Sets the type of width transition used if any widths are different
          between the two input CrossSections.

    """
    from gdsfactory.pdk import get_layer

    X1 = cross_section1
    X2 = cross_section2

    if not X1.aliases or not X2.aliases:
        raise ValueError(
            """transition() found no named sections in one
        or both inputs (cross_section1/cross_section2)."""
        )

    layers1 = {get_layer(section.layer) for section in X1.sections}
    layers2 = {get_layer(section.layer) for section in X2.sections}
    layers1.add(get_layer(X1.layer))
    layers2.add(get_layer(X2.layer))

    has_common_layers = bool(layers1.intersection(layers2))
    if not has_common_layers:
        raise ValueError(
            f"transition() found no common layers X1 {layers1} and X2 {layers2}"
        )

    sections = []
    for alias in X1.aliases.keys():
        if alias in X2.aliases:
            section1 = X1.aliases[alias]
            section2 = X2.aliases[alias]

            offset1 = section1.offset
            offset2 = section2.offset
            width1 = section1.width
            width2 = section2.width

            if callable(offset1):
                offset1 = offset1(1)
            if callable(offset2):
                offset2 = offset2(0)
            if callable(width1):
                width1 = width1(1)
            if callable(width2):
                width2 = width2(0)

            offset_fun = _sinusoidal_transition(offset1, offset2)

            if width_type == "linear":
                width_fun = _linear_transition(width1, width2)
            elif width_type == "sine":
                width_fun = _sinusoidal_transition(width1, width2)
            else:
                raise ValueError(f"width_type={width_type!r} must be {'sine','linear'}")

            if section1.layer != section2.layer:
                hidden = True
                layer1 = get_layer(section1.layer)
                layer2 = get_layer(section2.layer)
                layer = (layer1, layer2)
            else:
                hidden = False
                layer = get_layer(section1.layer)

            s = Section(
                width=width_fun,
                offset=offset_fun,
                layer=layer,
                port_names=(section2.port_names[0], section1.port_names[1]),
                port_types=(section2.port_types[0], section1.port_types[1]),
                name=alias,
                hidden=hidden,
            )
            sections.append(s)

    return Transition(
        cross_section1=X1,
        cross_section2=X2,
        width_type=width_type,
        sections=sections,
        width=max([X1.width, X2.width]),
    )


@cell
def extrude(
    p: Path,
    cross_section: Optional[CrossSectionSpec] = None,
    layer: Optional[LayerSpec] = None,
    width: Optional[float] = None,
    widths: Optional[Float2] = None,
    simplify: Optional[float] = None,
    shear_angle_start: Optional[float] = None,
    shear_angle_end: Optional[float] = None,
) -> Component:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component
    The CrossSection defines the layer numbers, widths and offsetts

    Args:
        p: a path is a list of points (arc, straight, euler).
        cross_section: to extrude.
        layer: optional layer to extrude.
        width: optional width to extrude.
        widths: tuple of starting and end width.
        simplify: Tolerance value for the simplification algorithm.
          All points that can be removed without changing the resulting.
          polygon by more than the value listed here will be removed.
        shear_angle_start: an optional angle to shear the starting face by (in degrees).
        shear_angle_end: an optional angle to shear the ending face by (in degrees).

    """
    from gdsfactory.pdk import (
        get_active_pdk,
        get_cross_section,
        get_grid_size,
        get_layer,
    )

    if cross_section is None and layer is None:
        raise ValueError("CrossSection or layer needed")

    if cross_section is not None and layer is not None:
        raise ValueError("Define only CrossSection or layer")

    if layer is not None and width is None and widths is None:
        raise ValueError("Need to define layer width or widths")
    elif width:
        cross_section = CrossSection(width=width, layer=layer)

    elif widths:
        cross_section = CrossSection(
            width=_linear_transition(widths[0], widths[1]), layer=layer
        )

    xsection_points = []
    c = Component()

    x = get_cross_section(cross_section)
    snap_to_grid_nm = int(1e3 * (x.snap_to_grid or get_grid_size()))
    sections = x.sections or []
    sections = list(sections)

    if x.layer and x.width and x.add_center_section:
        sections += [
            Section(
                width=x.width,
                offset=x.offset,
                layer=get_layer(x.layer),
                port_names=x.port_names,
                port_types=x.port_types,
            )
        ]

    if x.cladding_layers and x.cladding_offsets:
        for layer, cladding_offset in zip(x.cladding_layers, x.cladding_offsets):
            width = x.width(1) if callable(x.width) else x.width
            sections += [
                Section(
                    width=width + 2 * cladding_offset,
                    offset=x.offset,
                    layer=get_layer(layer),
                )
            ]

    for section in sections:
        width = section.width
        offset = section.offset
        layer = get_layer(section.layer)
        port_names = section.port_names
        port_types = section.port_types
        hidden = section.hidden

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

        # print(offset, type(offset))
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
            lengths = np.cumsum(np.sqrt(dx**2 + dy**2))
            lengths = np.concatenate([[0], lengths])
            width = width(lengths / lengths[-1])
        dy = offset + width / 2
        # _points = _shear_face(points, dy, shear_angle_start, shear_angle_end)

        points1 = p._centerpoint_offset_curve(
            points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        dy = offset - width / 2
        # _points = _shear_face(points, dy, shear_angle_start, shear_angle_end)

        points2 = p._centerpoint_offset_curve(
            points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        if shear_angle_start or shear_angle_end:
            _face_angle_start = (
                start_angle + shear_angle_start - 90 if shear_angle_start else None
            )
            _face_angle_end = (
                end_angle + shear_angle_end + 90 if shear_angle_end else None
            )
            points1 = _cut_path_with_ray(
                start_point=points[0],
                start_angle=_face_angle_start,
                end_point=points[-1],
                end_angle=_face_angle_end,
                path=points1,
            )
            points2 = _cut_path_with_ray(
                start_point=points[0],
                start_angle=_face_angle_start,
                end_point=points[-1],
                end_angle=_face_angle_end,
                path=points2,
            )

        # angle = start_angle + shear_angle_start + 90
        # points2 = _cut_path_with_ray(points[0], angle, points2, start=True)
        # Simplify lines using the Ramer–Douglas–Peucker algorithm
        if isinstance(simplify, bool):
            raise ValueError("simplify argument must be a number (e.g. 1e-3) or None")

        if simplify is not None:
            points1 = _simplify(points1, tolerance=simplify)
            points2 = _simplify(points2, tolerance=simplify)

        if x.snap_to_grid:
            points = snap.snap_to_grid(points, snap_to_grid_nm)
            points1 = snap.snap_to_grid(points1, snap_to_grid_nm)
            points2 = snap.snap_to_grid(points2, snap_to_grid_nm)

        # Join points together
        points_poly = np.concatenate([points1, points2[::-1, :]])

        layers = layer if hidden else [layer, layer]
        if not hidden and p.length() > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        pdk = get_active_pdk()
        warn_off_grid_ports = pdk.warn_off_grid_ports

        # Add port_names if they were specified
        if port_names[0] is not None:
            port_width = width if np.isscalar(width) else width[0]
            port_orientation = (p.start_angle + 180) % 360
            center = points[0]
            face = [points1[0], points2[0]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            if warn_off_grid_ports:
                center_snap = snap.snap_to_grid(center, snap_to_grid_nm)
                if center[0] != center_snap[0] or center[1] != center_snap[1]:
                    warnings.warn(f"Port center {center} has off-grid ports")

            port1 = c.add_port(
                port=Port(
                    name=port_names[0],
                    layer=get_layer(layers[0]),
                    port_type=port_types[0],
                    width=port_width,
                    orientation=port_orientation,
                    center=center,
                    cross_section=x.cross_sections[0]
                    if hasattr(x, "cross_sections")
                    else x,
                    shear_angle=shear_angle_start,
                )
            )
            port1.info["face"] = face
        if port_names[1] is not None:
            port_width = width if np.isscalar(width) else width[-1]
            port_orientation = (p.end_angle) % 360
            center = points[-1]
            face = [points1[-1], points2[-1]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            if warn_off_grid_ports:

                center_snap = snap.snap_to_grid(center, snap_to_grid_nm)

                if center[0] != center_snap[0] or center[1] != center_snap[1]:
                    warnings.warn(f"Port center {center} has off-grid ports")
            port2 = c.add_port(
                port=Port(
                    name=port_names[1],
                    layer=get_layer(layers[1]),
                    port_type=port_types[1],
                    width=port_width,
                    center=center,
                    orientation=port_orientation,
                    cross_section=x.cross_sections[1]
                    if hasattr(x, "cross_sections")
                    else x,
                    shear_angle=shear_angle_end,
                )
            )
            port2.info["face"] = face

    c.info["length"] = float(np.round(p.length(), 3))

    if x.decorator:
        c = x.decorator(c) or c

    if x.add_pins:
        c = x.add_pins(c) or c

    return c


def _rotated_delta(
    point: np.ndarray, center: np.ndarray, orientation: float
) -> np.ndarray:
    """Gets the rotated distance of a point from a center.

    Args:
        point: the initial point.
        center: a center point to use as a reference.
        orientation: the rotation, in degrees.

    Returns: the normalized delta between the point and center, accounting for rotation
    """
    ca = np.cos(orientation * np.pi / 180)
    sa = np.sin(orientation * np.pi / 180)
    rot_mat = np.array([[ca, -sa], [sa, ca]])
    delta = point - center
    return np.dot(delta, rot_mat)


def _cut_path_with_ray(
    start_point: np.ndarray,
    start_angle: Optional[float],
    end_point: np.ndarray,
    end_angle: Optional[float],
    path: np.ndarray,
) -> np.ndarray:
    """Cuts or extends a path given a point and angle to project."""
    import shapely.geometry as sg

    # a distance to approximate infinity to find ray-segment intersections
    far_distance = 10000

    path_cmp = np.copy(path)
    # pad start
    dp = path[0] - path[1]
    d_ext = far_distance / np.sqrt(np.sum(dp**2)) * np.array([dp[0], dp[1]])
    path_cmp[0] += d_ext
    # pad end
    dp = path[-1] - path[-2]
    d_ext = far_distance / np.sqrt(np.sum(dp**2)) * np.array([dp[0], dp[1]])
    path_cmp[-1] += d_ext

    intersections = [sg.Point(path[0]), sg.Point(path[-1])]
    distances = []
    ls = sg.LineString(path_cmp)
    for i, angle, point in [(0, start_angle, start_point), (1, end_angle, end_point)]:
        if angle:
            # get intersection
            angle_rad = np.deg2rad(angle)
            dx_far = np.cos(angle_rad) * far_distance
            dy_far = np.sin(angle_rad) * far_distance
            d_far = np.array([dx_far, dy_far])
            ls_ray = sg.LineString([point - d_far, point + d_far])
            intersection = ls.intersection(ls_ray)

            if not isinstance(intersection, sg.Point):
                if not isinstance(intersection, sg.MultiPoint):
                    raise ValueError(
                        f"Expected intersection to be a point, but got {intersection}"
                    )
                _, nearest = shapely.ops.nearest_points(sg.Point(point), intersection)
                intersection = nearest
            intersections[i] = intersection
        else:
            intersection = intersections[i]
        distance = ls.project(intersection)
        distances.append(distance)
    # when trimming the start, start counting at the intersection point, then add all subsequent points
    points = [np.array(intersections[0].coords[0])]
    for point in path[1:-1]:
        if distances[0] < ls.project(sg.Point(point)) < distances[1]:
            points.append(point)
    points.append(np.array(intersections[1].coords[0]))
    return np.array(points)


def arc(radius: float = 10.0, angle: float = 90, npoints: int = 720) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        npoints: Number of points used per 360 degrees.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.arc(radius=10, angle=45)
        p.plot()

    """
    return Path().from_phidl(path.arc(radius=radius, angle=angle, num_pts=npoints))


def euler(
    radius: float = 10,
    angle: float = 90,
    p: float = 0.5,
    use_eff: bool = False,
    npoints: int = 720,
) -> Path:
    """Returns an euler bend that adiabatically transitions from straight to curved.

    `radius` is the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc).
    If p < 1.0, will create a "partial euler" curve as described in Vogelbacher et. al.
    https://dx.doi.org/10.1364/oe.27.031394

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        use_eff: If False: `radius` is the minimum radius of curvature of the bend.
            If True: The curve will be scaled such that the endpoints match an arc.
            with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.euler(radius=10, angle=45, p=1, use_eff=True, npoints=720)
        p.plot()

    """
    return Path().from_phidl(
        path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, num_pts=npoints)
    )


def straight(length: float = 10.0, npoints: int = 2) -> Path:
    """Returns a straight path.

    For transitions you should increase have at least 100 points

    Args:
        length: of straight.
        npoints: number of points.

    """
    if length < 0:
        raise ValueError(f"length = {length} needs to be > 0")
    x = np.linspace(0, length, npoints)
    y = x * 0
    points = np.array((x, y)).T

    p = Path()
    p.append(points)
    return p


def spiral_archimedean(
    min_bend_radius: float, separation: float, number_of_loops: float, npoints: int
) -> Path:
    """Returns an Archimedean spiral.

    Args:
        radius: Inner radius of the spiral.
        separation: Separation between the loops in um.
        number_of_loops: number of loops.
        npoints: number of Points.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.spiral_archimedean(min_bend_radius=5, separation=2, number_of_loops=3, npoints=200)
        p.plot()

    """
    return Path(
        [
            (separation / np.pi * theta + min_bend_radius)
            * np.array((np.sin(theta), np.cos(theta)))
            for theta in np.linspace(0, number_of_loops * 2 * np.pi, npoints)
        ]
    )


def smooth(
    points: Coordinates,
    radius: float = 4.0,
    bend: PathFactory = euler,
    **kwargs,
) -> Path:
    """Returns a smooth Path from a series of waypoints.

    Args:
        points: array-like[N][2] List of waypoints for the path to follow.
        radius: radius of curvature, passed to `bend`.
        bend: bend function to round corners.
        kwargs: Extra keyword arguments that will be passed to `bend`.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.smooth(([0, 0], [0, 10], [10, 10]))
        p.plot()

    """
    return Path().from_phidl(
        smooth_phidl(points=points, radius=radius, corner_fun=bend, **kwargs)
    )


__all__ = ["straight", "euler", "arc", "extrude", "path", "transition", "smooth"]


def _demo() -> None:
    import gdsfactory as gf

    c = gf.Component()
    X1 = gf.CrossSection()
    X1.add(width=1.2, offset=0, layer=2, name="wg", ports=("in1", "out1"))
    X1.add(width=2.2, offset=0, layer=3, name="etch")
    X1.add(width=1.1, offset=3, layer=1, name="wg2")

    # Create the second CrossSection that we want to transition to
    X2 = gf.CrossSection()
    X2.add(width=1, offset=0, layer=2, name="wg", ports=("in2", "out2"))
    X2.add(width=3.5, offset=0, layer=3, name="etch")
    X2.add(width=3, offset=5, layer=1, name="wg2")

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")

    P1 = gf.path.straight(length=5)
    P2 = gf.path.straight(length=5)

    wg1 = gf.path.extrude(P1, X1)
    wg2 = gf.path.extrude(P2, X2)

    P4 = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    wg_trans = gf.path.extrude(P4, Xtrans)
    wg1_ref = c << wg1
    wg2_ref = c << wg2
    wgt_ref = c << wg_trans
    wgt_ref.connect("in2", wg1_ref.ports["out1"])
    wg2_ref.connect("in2", wgt_ref.ports["out1"])

    print(wg1)
    print(wg2)
    print(wg_trans)
    c.show(show_ports=True)


def _my_custom_width_fun(t):
    # Note: Custom width/offset functions MUST be vectorizable--you must be able
    # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
    num_periods = 5
    return 3 + np.cos(2 * np.pi * t * num_periods)


def _demo_variable_width() -> None:
    # Create the Path
    P = straight(length=40, npoints=40)

    # Create two cross-sections: one fixed width, one modulated by my_custom_offset_fun
    s = Section(width=_my_custom_width_fun, offset=0, layer=(1, 0))
    X = CrossSection(width=3, offset=-6, layer=(2, 0), sections=[s])

    # Extrude the Path to create the Component
    c = extrude(P, cross_section=X)
    c.show(show_ports=True)


def _my_custom_offset_fun(t):
    # Note: Custom width/offset functions MUST be vectorizable--you must be able
    # to call them with an array input like my_custom_offset_fun([0, 0.1, 0.2, 0.3, 0.4])
    num_periods = 3
    return 3 + np.cos(2 * np.pi * t * num_periods)


def _demo_variable_offset() -> None:
    # Create the Path
    P = straight(length=40, npoints=30)

    # Create two cross-sections: one fixed offset, one modulated by my_custom_offset_fun
    s1 = Section(width=1, offset=_my_custom_offset_fun, layer=(2, 0))
    X = CrossSection(layer=(1, 0), width=1, sections=[s1])

    # Extrude the Path to create the Component
    c = extrude(P, cross_section=X)
    c.show(show_ports=True)


if __name__ == "__main__":
    import gdsfactory as gf

    # p = gf.path.euler(radius=25, angle=45, p=0.5, use_eff=False)
    # s1 = gf.Section(width=2, offset=2, layer=(2, 0))
    # s2 = gf.Section(width=2, offset=-2, layer=(2, 0))
    # x = gf.CrossSection(
    #     width=1, offset=0, layer=(1, 0), ports=("in", "out"), sections=[s1, s2]
    # )
    # c1 = gf.path.extrude(p, cross_section=x)
    # p = gf.path.straight()
    # c2 = gf.path.extrude(p, cross_section=x)
    # c2.show()
    # Create our first CrossSection
    # s1 = gf.Section(width=2.2, offset=0, layer=(3, 0), name="etch")
    # s2 = gf.Section(width=1.1, offset=3, layer=(1, 0), name="wg2")
    # X1 = gf.CrossSection(
    #     width=1.2,
    #     offset=0,
    #     layer=(2, 0),
    #     name="wg",
    #     ports=("o1", "o2"),
    #     sections=[s1, s2],
    # )
    # # Create the second CrossSection that we want to transition to
    # s1 = gf.Section(width=3.5, offset=0, layer=(3, 0), name="etch")
    # s2 = gf.Section(width=3, offset=5, layer=(1, 0), name="wg2")
    # X2 = gf.CrossSection(
    #     width=1,
    #     offset=0,
    #     layer=(2, 0),
    #     name="wg",
    #     ports=("o1", "o2"),
    #     sections=[s1, s2],
    # )
    # # To show the cross-sections, let's create two Paths and
    # # create Devices by extruding them
    # P1 = gf.path.straight(length=5)
    # P2 = gf.path.straight(length=5)
    # wg1 = gf.path.extrude(P1, X1)
    # wg2 = gf.path.extrude(P2, X2)
    # # Place both cross-section Devices and quickplot them
    # c = gf.Component()
    # wg1ref = c << wg1
    # wg2ref = c << wg2
    # wg2ref.movex(7.5)
    # # Create the transitional CrossSection
    # Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type="sine")
    # # Create a Path for the transitional CrossSection to follow
    # P3 = gf.path.straight(length=15, npoints=100)
    # # Use the transitional CrossSection to create a Component
    # straight_transition = gf.path.extrude(P3, Xtrans)
    # straight_transition.show()

    P = gf.path.straight(length=10, npoints=101)
    # c = gf.path.extrude(P, layer=(1, 0), widths=(1, 3))
    # c.show(show_ports=True)

    # s = gf.Section(width=3, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    # X1 = gf.CrossSection(
    #     width=1,
    #     offset=0,
    #     layer=gf.LAYER.WG,
    #     name="core",
    #     port_names=("o1", "o2"),
    #     sections=[s],
    # )
    # c = gf.path.extrude(P, X1)

    # s = gf.Section(width=0.1, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    # X2 = gf.CrossSection(
    #     width=3,
    #     offset=0,
    #     layer=gf.LAYER.WG,
    #     name="core",
    #     port_names=("o1", "o2"),
    #     sections=[s],
    # )
    # c2 = gf.path.extrude(P, X2)

    # T = gf.path.transition(X1, X2)
    # c3 = gf.path.extrude(P, T)
    # c3.show()

    # c = gf.Component("bend")
    # b = c << gf.components.bend_circular(angle=40)
    # s = c << gf.components.straight(length=5)
    # s.connect("o1", b.ports["o2"])
    # c = c.flatten()
    # c.show(show_ports=True, precision=1e-9)

    P3 = arc()
    P3.plot()
