"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component

The CrossSection defines the layer numbers, widths and offsetts

Based on phidl.path
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
from phidl import path
from phidl.device_layout import Path as PathPhidl
from phidl.device_layout import _simplify
from phidl.path import smooth as smooth_phidl

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection, Section, Transition
from gdsfactory.types import Coordinates, CrossSectionSpec, Float2, Layer, PathFactory


class Path(PathPhidl):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """check Path has the correct type."""
        assert isinstance(v, PathPhidl), f"TypeError, Got {type(v)}, expecting Path"
        return v

    def to_dict(self):
        return self.hash_geometry()

    def extrude(
        self,
        cross_section: Optional[CrossSectionSpec] = None,
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
        return extrude(
            p=self,
            cross_section=cross_section,
            layer=layer,
            width=width,
            widths=widths,
            simplify=simplify,
        )


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
    """Returns the function for an exponential transition

    Args:
        y1: start width
        y2: end width
        exp: exponent
    """
    return lambda t: y1 + (y2 - y1) * t**exp


def transition(
    cross_section1: CrossSection,
    cross_section2: CrossSection,
    width_type: str = "sine",
) -> Transition:
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

    if not X1.aliases or not X2.aliases:
        raise ValueError(
            """transition() found no named sections in one
        or both inputs (cross_section1/cross_section2)."""
        )

    layers1 = {section.layer for section in X1.sections}
    layers2 = {section.layer for section in X2.sections}
    layers1.add(X1.layer)
    layers2.add(X2.layer)

    has_common_layers = True if layers1.intersection(layers2) else False
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

            if width_type == "sine":
                width_fun = _sinusoidal_transition(width1, width2)
            elif width_type == "linear":
                width_fun = _linear_transition(width1, width2)
            else:
                raise ValueError(f"width_type={width_type!r} must be {'sine','linear'}")

            if section1.layer != section2.layer:
                hidden = True
                layer = (section1.layer, section2.layer)
            else:
                hidden = False
                layer = section1.layer

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

    Xtrans = Transition(
        cross_section1=X1, cross_section2=X2, width_type=width_type, sections=sections
    )
    return Xtrans


@cell
def extrude(
    p: Path,
    cross_section: Optional[CrossSectionSpec] = None,
    layer: Optional[Layer] = None,
    width: Optional[float] = None,
    widths: Optional[Float2] = None,
    simplify: Optional[float] = None,
    shear_angle_start: Optional[float] = None,
    shear_angle_end: Optional[float] = None,
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
          All points that can be removed without changing the resulting.
          polygon by more than the value listed here will be removed.
        shear_angle_start: an optional angle to shear the starting face by (in degrees).
        shear_angle_end: an optional angle to shear the ending face by (in degrees).
    """
    from gdsfactory.pdk import get_cross_section

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
    snap_to_grid = x.snap_to_grid
    sections = x.sections or []
    sections = list(sections)

    if x.layer and x.width:
        sections += [
            Section(
                width=x.width,
                offset=x.offset,
                layer=x.layer,
                port_names=x.port_names,
                port_types=x.port_types,
            )
        ]

    for section in sections:
        width = section.width
        offset = section.offset
        layer = section.layer
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
        else:
            pass
        dy = offset + width / 2
        _points = _shear_face(points, dy, shear_angle_start, shear_angle_end)

        points1 = p._centerpoint_offset_curve(
            _points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        dy = offset - width / 2
        _points = _shear_face(points, dy, shear_angle_start, shear_angle_end)

        points2 = p._centerpoint_offset_curve(
            _points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        # Simplify lines using the Ramer–Douglas–Peucker algorithm
        if isinstance(simplify, bool):
            raise ValueError("simplify argument must be a number (e.g. 1e-3) or None")
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
        if not hidden and p.length() > 1e-3:
            c.add_polygon(points, layer=layer)

        # Add port_names if they were specified
        if port_names[0] is not None:
            orientation = (p.start_angle + 180) % 360
            _width = width if np.isscalar(width) else width[0]
            new_port = c.add_port(
                name=port_names[0],
                layer=layers[0],
                port_type=port_types[0],
                width=_width,
                orientation=orientation,
                cross_section=x.cross_sections[0]
                if hasattr(x, "cross_sections")
                else x,
            )
            new_port.endpoints = (points1[0], points2[0])
        if port_names[1] is not None:
            orientation = (p.end_angle + 180) % 360
            _width = width if np.isscalar(width) else width[-1]
            new_port = c.add_port(
                name=port_names[1],
                layer=layers[1],
                port_type=port_types[1],
                width=_width,
                orientation=orientation,
                cross_section=x.cross_sections[1]
                if hasattr(x, "cross_sections")
                else x,
            )
            new_port.endpoints = (points2[-1], points1[-1])

    # points = np.concatenate((p.points, np.array(xsection_points)))
    # points_hash = hash_points(points)[:26]
    # name = f"path_{points_hash}"
    # c.info.points_hash = points_hash
    # clean_dict(p.info)
    # clean_dict(cross_section.info)
    # c.info.path = p.info
    # c.info.cross_section = cross_section.info
    c.info["length"] = float(np.round(p.length(), 3))

    if x.decorator:
        c = x.decorator(c) or c

    return c


def _shear_face(
    points: np.ndarray,
    dy: float,
    shear_angle_start: Optional[float],
    shear_angle_end: Optional[float],
) -> np.ndarray:
    """
    Displaces a point sequence offset a distance dy away by a shear angle,
    given in degrees, on either side.

    Args:
        points: points sequence.
        dy: y offset.
        shear_angle_start: an optional angle to shear the starting face by (in degrees).
        shear_angle_end: an optional angle to shear the ending face by (in degrees).
    """
    if shear_angle_start or shear_angle_end:
        shear_angle_start = shear_angle_start or 0
        shear_angle_end = shear_angle_end or 0

        starts_monotonically_increasing = np.all(points[1:, 0] > points[:-1, 0])
        if not starts_monotonically_increasing:
            raise ValueError(
                "Cannot shear face! Current algorithm assumes that all points are monotonically increasing in x."
            )
        dx_start = np.tan(np.deg2rad(shear_angle_start)) * dy
        dx_end = np.tan(np.deg2rad(shear_angle_end)) * dy
        points = points.copy()
        points[0][0] += dx_start
        points[-1][0] += dx_end
        ends_monotonically_increasing = np.all(points[1:, 0] > points[:-1, 0])

        if not ends_monotonically_increasing:
            raise ValueError(
                "Cannot shear face! Sheared segment is not monotonically increasing! Maybe the segment is too short or has curvature?"
            )
    return points


def arc(radius: float = 10.0, angle: float = 90, npoints: int = 720) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature
        angle: total angle of the curve
        npoints: Number of points used per 360 degrees

    """
    p = path.arc(radius=radius, angle=angle, num_pts=npoints)
    p.extrude = extrude
    return p


def euler(
    radius: float = 10,
    angle: float = 90,
    p: float = 0.5,
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
    p = path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, num_pts=npoints)
    p.extrude = extrude
    return p


def straight(length: float = 10.0, npoints: int = 2) -> Path:
    """Returns a straight path

    For transitions you should increase have at least 100 points

    Args:
        length: of straight
        npoints: number of points
    """
    if length < 0:
        raise ValueError(f"length = {length} needs to be > 0")
    x = np.linspace(0, length, npoints)
    y = x * 0
    points = np.array((x, y)).T

    p = Path()
    p.append(points)
    return p


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


def _demo():
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
    c.show()


def _my_custom_width_fun(t):
    # Note: Custom width/offset functions MUST be vectorizable--you must be able
    # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
    num_periods = 5
    w = 3 + np.cos(2 * np.pi * t * num_periods)
    return w


def _demo_variable_width():
    # Create the Path
    P = straight(length=40, npoints=40)

    # Create two cross-sections: one fixed width, one modulated by my_custom_offset_fun
    s = Section(width=_my_custom_width_fun, offset=0, layer=(1, 0))
    X = CrossSection(width=3, offset=-6, layer=(2, 0), sections=[s])

    # Extrude the Path to create the Component
    c = extrude(P, cross_section=X)
    c.show()


def _my_custom_offset_fun(t):
    # Note: Custom width/offset functions MUST be vectorizable--you must be able
    # to call them with an array input like my_custom_offset_fun([0, 0.1, 0.2, 0.3, 0.4])
    num_periods = 3
    w = 3 + np.cos(2 * np.pi * t * num_periods)
    return w


def _demo_variable_offset():
    # Create the Path
    P = straight(length=40, npoints=30)

    # Create two cross-sections: one fixed offset, one modulated by my_custom_offset_fun
    s1 = Section(width=1, offset=_my_custom_offset_fun, layer=(2, 0))
    X = CrossSection(layer=(1, 0), width=1, sections=[s1])

    # Extrude the Path to create the Component
    c = extrude(P, cross_section=X)
    c.show()


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

    s = gf.Section(width=3, offset=0, layer=gf.LAYER.SLAB90)
    X1 = gf.CrossSection(
        width=1,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        ports=("o1", "o2"),
        sections=[s],
    )
    c = gf.path.extrude(P, X1)

    X2 = gf.CrossSection(
        width=3, offset=0, layer=gf.LAYER.WG, name="core", ports=("o1", "o2")
    )
    c2 = gf.path.extrude(P, X2)

    T = gf.path.transition(X1, X2)
    c3 = gf.path.extrude(P, T)
    c3.show()
