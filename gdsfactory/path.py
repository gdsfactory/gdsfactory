"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component
The CrossSection defines the layer numbers, widths and offsets

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from numpy import mod, pi
from typing_extensions import Self

from gdsfactory._deprecation import deprecate
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.component_layout import (
    GeometryHelper,
    parse_move,
    reflect_points,
    rotate_points,
)
from gdsfactory.cross_section import (  # type: ignore[attr-defined]
    CrossSection,
    Section,
    Transition,
)
from gdsfactory.typings import (
    AngleInDegrees,
    AnyComponent,
    Axis,
    ComponentSpec,
    Coordinate,
    CrossSectionSpec,
    LayerSpec,
    WidthTypes,
)


def _simplify(
    points: npt.NDArray[np.floating[Any]], tolerance: float
) -> npt.NDArray[np.floating[Any]]:
    import shapely.geometry as sg  # type: ignore[import-untyped]

    ls = sg.LineString(points)
    ls_simple = ls.simplify(tolerance=tolerance)
    return np.asarray(ls_simple.coords)


class Path(GeometryHelper):
    """You can extrude a Path with a CrossSection to create a Component.

    Parameters:
        path: array-like[N][2], Path, or list of Paths.

    """

    def __init__(
        self, path: npt.NDArray[np.floating[Any]] | Path | None = None
    ) -> None:
        """Creates an empty path."""
        self.points: npt.NDArray[np.floating[Any]] = np.array(
            [[0, 0]], dtype=np.float64
        )
        self.start_angle: float = 0
        self.end_angle: float = 0
        self.info: dict[str, Any] = {}
        if path is not None:
            # If array[N][2]
            if isinstance(path, Path):
                self.points = np.array(path.points, dtype=np.float64)
                self.start_angle = path.start_angle
                self.end_angle = path.end_angle
                self.info = {}
            elif (
                (np.asarray(path, dtype=object).ndim == 2)
                and np.issubdtype(np.array(path).dtype, np.number)
                and (np.shape(path)[1] == 2)
            ):
                self.points = np.array(path, dtype=np.float64)
                nx1, ny1 = self.points[1] - self.points[0]
                self.start_angle = np.arctan2(ny1, nx1) / np.pi * 180
                nx2, ny2 = self.points[-1] - self.points[-2]
                self.end_angle = np.arctan2(ny2, nx2) / np.pi * 180
            elif np.asarray(path, dtype=object).size > 1:
                self.append(path)
            else:
                raise ValueError(
                    "Path() the `path` argument must be either blank, a path Object, "
                    "an array-like[N][2] list of points, or a list of these"
                )

    def __getattribute__(self, __k: str) -> Any:
        """Deprecate dbu prefixed attributes."""
        if __k in {
            "dcenter",
            "dmirror",
            "dmove",
            "dmovex",
            "dmovey",
            "drotate",
            "dx",
            "dxmin",
            "dxmax",
            "dxsize",
            "dy",
            "dymin",
            "dymax",
            "dysize",
        }:
            deprecate(__k, f"{__k[1:]}")
            return getattr(self, f"{__k[1:]}")
        return super().__getattribute__(__k)

    def __repr__(self) -> str:
        """Returns path points."""
        return (
            f"Path(start_angle={self.start_angle}, "
            f"end_angle={self.end_angle}, "
            f"points={self.points})"
        )

    def __len__(self) -> int:
        """Returns path points."""
        return len(self.points)

    def __iadd__(self, path_or_points: npt.NDArray[np.floating[Any]] | Path) -> Path:
        """Adds points to current path."""
        return self.append(path_or_points)

    def __add__(self, path: npt.NDArray[np.floating[Any]] | Path) -> Path:
        """Returns new path concatenating current and new path."""
        new = self.copy()
        return new.append(path)

    @property
    def bbox(self) -> npt.NDArray[np.floating[Any]]:
        """Returns the bounding box of the Path."""
        bbox = [
            (np.min(self.points[:, 0]), np.min(self.points[:, 1])),
            (np.max(self.points[:, 0]), np.max(self.points[:, 1])),
        ]
        return np.array(bbox)

    def append(self, path: npt.NDArray[np.floating[Any]] | Path | list[Path]) -> Path:
        """Attach Path to the end of this Path.

        The input path automatically rotates and translates such that it continues
        smoothly from the previous segment.

        Args:
            path: Path, array-like[N][2], or list of Paths. The input path that will be appended.
        """
        # If appending another Path, load relevant variables
        if isinstance(path, Path):
            start_angle = path.start_angle
            end_angle = path.end_angle
            points = path.points
        # If array[N][2]
        elif (
            (np.asarray(path, dtype=object).ndim == 2)
            and not isinstance(path[0], Path)
            and np.issubdtype(np.array(path).dtype, np.number)
            and (np.shape(path)[1] == 2)
        ):
            points = np.asarray(path, dtype=np.float64)
            nx1, ny1 = points[1] - points[0]
            start_angle = np.arctan2(ny1, nx1) / np.pi * 180
            nx2, ny2 = points[-1] - points[-2]
            end_angle = np.arctan2(ny2, nx2) / np.pi * 180
        elif isinstance(path, list):
            for p in path:
                self.append(p)
            return self
        else:
            raise ValueError(
                "Path.append() the `path` argument must be either "
                "a Path object, an array-like[N][2] list of points, or a list of these"
            )

        # Connect beginning of new points with old points
        points = rotate_points(points, angle=self.end_angle - start_angle)
        points += self.points[-1, :] - points[0, :]

        # Update end angle
        self.end_angle = mod(end_angle + self.end_angle - start_angle, 360)

        # Concatenate old points + new points
        self.points = np.vstack([self.points, points[1:]])

        return self

    def offset(
        self,
        offset: float | Callable[[float], float] = 0,
    ) -> Path:
        """Offsets Path so that it follows the Path centerline plus an offset.

        The offset can either be a fixed value, or a function
        of the form my_offset(t) where t goes from 0->1

        Args:
            offset: int or float, callable. Magnitude of the offset
        """
        if offset == 0:
            points = self.points
            start_angle: float = self.start_angle
            end_angle: float = self.end_angle
        elif callable(offset):
            # Compute lengths
            dx = np.diff(self.points[:, 0])
            dy = np.diff(self.points[:, 1])
            lengths = np.cumsum(np.sqrt((dx) ** 2 + (dy) ** 2))
            lengths = np.concatenate([[0], lengths])
            # Create list of offset points and perform offset
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=offset(lengths / lengths[-1]),  # type: ignore[unused-ignore]
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            # Numerically compute start and end angles
            tol = 1e-6
            ds = tol / lengths[-1]
            ny1 = offset(ds) - offset(0)
            start_angle = np.arctan2(-ny1, tol) / np.pi * 180 + self.start_angle
            # start_angle = np.round(start_angle, decimals=6)
            ny2 = offset(1) - offset(1 - ds)
            end_angle = np.arctan2(-ny2, tol) / np.pi * 180 + self.end_angle
            # end_angle = np.round(end_angle, decimals=6)
        else:  # Offset is just a number
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=offset,  # type: ignore[unused-ignore]
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            start_angle = self.start_angle
            end_angle = self.end_angle

        self.points = points
        self.start_angle = start_angle
        self.end_angle = end_angle
        return self

    def move(
        self,
        origin: Coordinate | npt.NDArray[np.floating[Any]],
        destination: Coordinate | None = None,
        axis: Axis | None = None,
    ) -> Self:
        """Moves the Path from the origin point to the destination.

        Both origin and destination can be 1x2 array-like or a Port.

        Args:
            origin : array-like[2], Port Origin point of the move.
            destination : array-like[2], Port Destination point of the move.
            axis : {'x', 'y'} Direction of move.

        """
        dx, dy = parse_move(origin, destination, axis)
        self.points += np.array([dx, dy])
        return self

    def rotate(self, angle: float = 45, center: Coordinate = (0, 0)) -> Self:
        """Rotates all Polygons in the Component around the specified center point.

        If no center point specified will rotate around (0,0).

        Args:
            angle: Angle to rotate the Component in degrees.
            center: array-like[2] or None. component of the Component.
        """
        if angle == 0:
            return self
        self.points = rotate_points(self.points, angle, center)
        if self.start_angle is not None:
            self.start_angle = mod(self.start_angle + angle, 360)
        if self.end_angle is not None:
            self.end_angle = mod(self.end_angle + angle, 360)
        return self

    def mirror(self, p1: Coordinate = (0, 1), p2: Coordinate = (0, 0)) -> Path:
        """Mirrors the Path across the line formed between the two specified points.

        ``points`` may be input as either single points [1,2]
        or array-like[N][2], and will return in kind.

        Args:
            p1: First point of the line.
            p2: Second point of the line.
        """
        self.points = reflect_points(self.points, p1, p2)
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        if self.start_angle is not None:
            self.start_angle = mod(2 * angle - self.start_angle, 360)
        if self.end_angle is not None:
            self.end_angle = mod(2 * angle - self.end_angle, 360)
        return self

    def centerpoint_offset_curve(
        self,
        points: npt.NDArray[np.floating[Any]],
        offset_distance: float | npt.NDArray[np.floating[Any]],
        start_angle: float | None,
        end_angle: float | None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Creates a offset curve computing the centerpoint offset of x and y points.

        Args:
            points: array-like[N][2] The points to be offset.
            offset_distance: array-like[N] The distance to offset the points.
            start_angle: float or None The angle at the start of the path.
            end_angle: float or None The angle at the end of the path.

        """
        new_points = np.array(points, dtype=np.float64)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        theta = np.arctan2(dy, dx)
        theta = np.concatenate([theta[:1], theta, theta[-1:]])
        theta_mid = (np.pi + theta[1:] + theta[:-1]) / 2  # Mean angle between segments
        dtheta_int = np.pi + theta[:-1] - theta[1:]  # Internal angle between segments
        offset_distance_array = np.array(offset_distance) / np.sin(dtheta_int / 2)

        # Ensure offset_distance has the correct shape
        if offset_distance_array.ndim == 0:
            offset_distance_array = np.full(points.shape[0], offset_distance_array)
        elif offset_distance_array.ndim == 1 and offset_distance_array.size == 1:
            offset_distance_array = np.full(points.shape[0], offset_distance_array[0])

        new_points[:, 0] -= offset_distance_array * np.cos(theta_mid)
        new_points[:, 1] -= offset_distance_array * np.sin(theta_mid)

        if start_angle is not None:
            start_angle_rad = start_angle * np.pi / 180
            new_points[0, :] = points[0, :] + (
                np.sin(start_angle_rad) * offset_distance_array[0],
                -np.cos(start_angle_rad) * offset_distance_array[0],
            )
        if end_angle is not None:
            end_angle_rad = end_angle * np.pi / 180
            new_points[-1, :] = points[-1, :] + (
                np.sin(end_angle_rad) * offset_distance_array[-1],
                -np.cos(end_angle_rad) * offset_distance_array[-1],
            )
        return new_points

    def _parametric_offset_curve(
        self,
        points: npt.NDArray[np.floating[Any]],
        offset_distance: npt.NDArray[np.floating[Any]],
        start_angle: float | None,
        end_angle: float | None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Creates a parametric offset by using gradient of the supplied x and y points.

        Args:
            points: array-like[N][2] The points to be offset.
            offset_distance: array-like[N] The distance to offset the points.
            start_angle: float or None The angle at the start of the path.
            end_angle: float or None The angle at the end of the path.


        """
        x = points[:, 0]
        y = points[:, 1]
        dxdt = np.gradient(x)
        dydt = np.gradient(y)
        if start_angle is not None:
            dxdt[0] = np.cos(start_angle * np.pi / 180)
            dydt[0] = np.sin(start_angle * np.pi / 180)
        if end_angle is not None:
            dxdt[-1] = np.cos(end_angle * np.pi / 180)
            dydt[-1] = np.sin(end_angle * np.pi / 180)
        x_offset = x + offset_distance * dydt / np.sqrt(dxdt**2 + dydt**2)
        y_offset = y - offset_distance * dxdt / np.sqrt(dydt**2 + dxdt**2)
        return np.array([x_offset, y_offset]).T

    def length(self) -> float:
        """Return cumulative length."""
        x = self.points[:, 0]
        y = self.points[:, 1]
        dx: npt.NDArray[np.floating[Any]] = np.diff(x)
        dy: npt.NDArray[np.floating[Any]] = np.diff(y)
        return float(np.round(np.sum(np.sqrt((dx) ** 2 + (dy) ** 2)), 3))  # type: ignore[unused-ignore]

    def curvature(
        self,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Calculates Path curvature.

        The curvature is numerically computed so areas where the curvature
        jumps instantaneously (such as between an arc and a straight segment)
        will be slightly interpolated, and sudden changes in point density
        along the curve can cause discontinuities.

        Returns:
            s: array-like[N] The arc-length of the Path
            K: array-like[N] The curvature of the Path
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt((dx) ** 2 + (dy) ** 2)
        s = np.cumsum(ds)
        theta = np.arctan2(dy, dx)

        # Fix discontinuities arising from np.arctan2
        dtheta = np.diff(theta)
        dtheta[np.where(dtheta > np.pi)] += -2 * np.pi
        dtheta[np.where(dtheta < -np.pi)] += 2 * np.pi
        theta = np.concatenate([[0], np.cumsum(dtheta)]) + theta[0]

        match len(ds):
            case 0 | 1:
                k = np.array([np.inf])
            case 2:
                k = np.nan_to_num(np.gradient(theta, s, edge_order=1), nan=np.inf)
            case _:
                k = np.gradient(theta, s, edge_order=2)

        return s, k

    def __hash__(self) -> int:
        """Computes a hash of the Path."""
        return self.hash_geometry()

    def hash_geometry(self, precision: float = 1e-4) -> int:
        """Computes an SHA1 hash of the points in the Path and the start_angle and end_angle.

        Args:
            precision: Rounding precision for the the objects in the Component. For instance, \
                    a precision of 1e-2 will round a point at (0.124, 1.748) to (0.12, 1.75)

        Returns:
            str Hash result in the form of an SHA1 hex digest string.

        .. code::

            hash(
                hash(First layer information: [layer1, datatype1]),
                hash(Polygon 1 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Polygon 2 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 3 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Second layer information: [layer2, datatype2]),
                hash(Polygon 1 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 2 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3)] ),
            )
        """
        magic_offset = 0.17048614

        # Create a SHA1 hash object
        final_hash = hashlib.sha1()

        # Adjust points by precision and add the magic offset, then convert to bytes
        adjusted_points = (
            ((self.points / precision) + magic_offset).round().astype(np.int64)
        )
        final_hash.update(adjusted_points.tobytes())

        # Adjust angles by precision, round and convert to bytes
        adjusted_angles = np.array([self.start_angle, self.end_angle])
        adjusted_angles = (
            ((adjusted_angles / precision) + magic_offset).round().astype(np.int64)
        )
        final_hash.update(adjusted_angles.tobytes())
        hash_bytes = final_hash.digest()
        return int.from_bytes(hash_bytes, byteorder="big")

    @classmethod
    def __get_validators__(cls) -> Iterator[Callable[[Any, Any], Path]]:
        """For pydantic."""
        yield cls._validate

    @classmethod
    def _validate(cls, v: Any, validation_info: Any) -> Path:
        """Pydantic Path validator."""
        assert isinstance(v, Path), f"TypeError, Got {type(v)}, expecting Path"
        return v

    def plot(self) -> None:
        """Plot path in matplotlib.

        .. plot::
            :include-source:

            import gdsfactory as gf

            p = gf.path.euler(radius=10)
            p.plot()
        """
        import matplotlib.pyplot as plt

        plt.plot(self.points[:, 0], self.points[:, 1])  # type: ignore
        plt.axis("equal")  # type: ignore
        plt.grid(True)  # type: ignore
        plt.show()  # type: ignore

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: Literal[False] = False,
    ) -> Component: ...

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: Literal[True] = True,
    ) -> ComponentAllAngle: ...

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: bool = True,
    ) -> AnyComponent: ...

    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: bool = False,
    ) -> AnyComponent:
        """Returns Component by extruding a Path with a CrossSection.

        A path can be extruded using any CrossSection returning a Component
        The CrossSection defines the layer numbers, widths and offsets.

        Args:
            cross_section: to extrude.
            layer: optional layer.
            width: optional width in um.
            simplify: Tolerance value for the simplification algorithm. \
                    All points that can be removed without changing the resulting polygon\
                    by more than the value listed here will be removed.

            all_angle: if True, the bend is drawn with a single euler curve.

        .. plot::
            :include-source:

            import gdsfactory as gf

            p = gf.path.euler(radius=10)
            c = p.extrude(layer=(1, 0), width=0.5)
            c.plot()
        """
        return extrude(
            p=self,
            cross_section=cross_section,
            layer=layer,
            width=width,
            simplify=simplify,
            all_angle=all_angle,
        )

    def extrude_transition(self, transition: Transition) -> Component:
        return extrude_transition(p=self, transition=transition)

    def copy(self) -> Path:
        """Returns a copy of the Path."""
        p = Path()
        p.info = self.info.copy()
        p.points = np.array(self.points)
        p.start_angle = self.start_angle
        p.end_angle = self.end_angle
        return p


PathFactory = Callable[..., Path]


def _sinusoidal_transition(
    y1: float, y2: float
) -> Callable[[float], npt.NDArray[np.floating[Any]]]:
    dy = y2 - y1

    def sine(t: float) -> npt.NDArray[np.floating[Any]]:
        return np.array(y1 + (1 - np.cos(np.pi * t)) / 2 * dy)

    return sine


def _parabolic_transition(
    y1: float, y2: float
) -> Callable[[float], npt.NDArray[np.floating[Any]] | float]:
    dy = y2 - y1

    def parabolic(t: float) -> npt.NDArray[np.floating[Any]] | float:
        res = y1 + np.sqrt(t) * dy
        if np.isscalar(t):
            return float(res)
        return np.array(res)

    return parabolic


def _linear_transition(y1: float, y2: float) -> Callable[[float], float]:
    dy = y2 - y1

    def linear(t: float) -> float:
        return y1 + t * dy

    return linear


def transition_exponential(
    y1: float, y2: float, exp: float = 0.5
) -> Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]]:
    """Returns the function for an exponential transition.

    Args:
        y1: start width in um.
        y2: end width in um.
        exp: exponent.

    """
    return lambda t: y1 + (y2 - y1) * t**exp  # type: ignore


adiabatic_polyfit_TE1550SOI_220nm = np.array(
    [
        1.02478963e-09,
        -8.65556534e-08,
        3.32415694e-06,
        -7.68408985e-05,
        1.19282177e-03,
        -1.31366332e-02,
        1.05721429e-01,
        -6.31057637e-01,
        2.80689677e00,
        -9.26867694e00,
        2.24535191e01,
        -3.90664800e01,
        4.71899278e01,
        -3.74726005e01,
        1.77381560e01,
        -1.12666286e00,
    ]
)


def transition_adiabatic(
    w1: float,
    w2: float,
    neff_w: Callable[[float], float],
    wavelength: float = 1.55,
    alpha: float = 1,
    max_length: float = 200,
    num_points_ODE: int = 2000,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Returns the points for an optimal adiabatic transition for well-guided modes.

    Args:
        w1: start width in um.
        w2: end width in um.
        neff_w: a callable that returns the effective index as a function of width. \
                By default, use a compact model of neff(y) for fundamental 1550 nm TE \
                mode of 220nm-thick core with 3.45 index, fully clad with 1.44 index.\
                Many coefficients are needed to capture the behaviour.
        wavelength: wavelength, in same units as widths.
        alpha: parameter that scales the rate of width change
            - closer to 0 means longer and more adiabatic;
            - 1 is the intuitive limit beyond which higher order modes are excited;
            - [2] reports good performance up to 1.4 for fundamental TE in SOI (for multiple core thicknesses)
        max_length: maximum length in um.
        num_points_ODE: number of samplings points for the ODE solve.

    References:
        [1] Burns, W. K., et al. "Optical waveguide parabolic coupling horns."
            Appl. Phys. Lett., vol. 30, no. 1, 1 Jan. 1977, pp. 28-30, doi:10.1063/1.89199.
        [2] Fu, Yunfei, et al. "Efficient adiabatic silicon-on-insulator waveguide taper."
            Photonics Res., vol. 2, no. 3, 1 June 2014, pp. A41-A44, doi:10.1364/PRJ.2.000A41.
    """
    from scipy.integrate import odeint  # type: ignore

    # Define ODE
    def dWdx(
        w: float,
        x: float,
        neff_w: Callable[[float], float],
        wavelength: float,
        alpha: float,
    ) -> float:
        return alpha * wavelength / (neff_w(w) * w)

    # Parse input
    if w2 < w1:
        wmin = w2
        wmax = w1
        order = -1
    else:
        wmin = w1
        wmax = w2
        order = 1

    # Solve ODE
    x = np.linspace(0, max_length, num_points_ODE)

    sol = odeint(dWdx, wmin, x, args=(neff_w, wavelength, alpha))

    # Extract optimal curve
    xs = x[np.where(sol[:, 0] < wmax)]
    ws = sol[:, 0][np.where(sol[:, 0] < wmax)]

    return xs, ws[::order]


def transition(
    cross_section1: CrossSectionSpec,
    cross_section2: CrossSectionSpec,
    width_type: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type: WidthTypes | Callable[[float, float, float], float] = "sine",
) -> Transition:
    """Returns a smoothly-transitioning between two CrossSections.

    Only cross-sectional elements that have the `name` (as in X.add(..., name = 'wg') )
    parameter specified in both input CrosSections will be created.
    Port names will be cloned from the input CrossSections in reverse.

    Args:
        cross_section1: First CrossSection.
        cross_section2: Second CrossSection.
        width_type: 'sine', 'parabolic', 'linear' or Callable. type of width transition used \
                if any widths are different between the two input CrossSections.
        offset_type: 'sine', 'parabolic', 'linear' or Callable. type of width transition used \
                if any widths are different between the two input CrossSections. \

    """
    from gdsfactory.pdk import get_cross_section, get_layer

    X1 = get_cross_section(cross_section1)
    X2 = get_cross_section(cross_section2)

    layers1 = {get_layer(section.layer) for section in X1.sections}
    layers2 = {get_layer(section.layer) for section in X2.sections}
    layers1.add(get_layer(X1.layer))
    layers2.add(get_layer(X2.layer))

    has_common_layers = bool(layers1.intersection(layers2))
    if not has_common_layers:
        raise ValueError(
            f"transition() found no common layers X1 {layers1} and X2 {layers2}"
        )

    return Transition(
        cross_section1=X1,
        cross_section2=X2,
        width_type=width_type,
        offset_type=offset_type,
    )


def along_path(
    p: Path,
    component: ComponentSpec,
    spacing: float,
    padding: float,
) -> Component:
    """Returns Component containing many copies of `component` along `p`.

    Places as many copies of `component` along each segment of `p` as possible
    under the given constraints. `spacing` is always followed precisely, but
    actual `padding` may exceed the provided value to place components evenly.

    Args:
        p: Path to place components along.
        component: Component to repeat along the path. The unrotated version of \
                this object should be oriented for placement on a horizontal line.
        spacing: distance between component placements.
        padding: minimum distance from the path start to the first component.
    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    length = p.length()
    number = (length - 2 * padding) // spacing + 1

    c = Component()

    cum_dist = 0.0
    next_component = (length - (number - 1) * spacing) / 2
    stop = length - next_component

    # Prepare in advance the rotation angle for each segment
    angle_list = [
        np.rad2deg(
            np.arctan2(
                (p.points[i + 1] - p.points[i])[1], (p.points[i + 1] - p.points[i])[0]
            )
        )
        for i in range(len(p.points) - 1)
    ]

    for i, start_pt in enumerate(p.points[:-1]):
        end_pt = p.points[i + 1]
        segment_vector = end_pt - start_pt
        segment_length = float(np.linalg.norm(segment_vector))
        unit_vector = segment_vector / segment_length

        # Get the pre-calculated angle for this segment
        angle = angle_list[i]

        while next_component <= cum_dist + segment_length and next_component <= stop:
            added_dist = next_component - cum_dist
            offset = added_dist * unit_vector
            component_ref = c << component
            component_ref.drotate(angle).dmove(start_pt + offset)
            next_component += spacing
        cum_dist += segment_length

    return c


def _get_named_sections(sections: tuple[Section, ...]) -> dict[str, Section]:
    from gdsfactory.pdk import get_layer

    named_sections = {}
    for section in sections:
        name = section.name or get_layer(section.layer)
        if name in named_sections:
            raise ValueError(
                f"Duplicate name or layer '{name}' of section used for cross-section in transition. Cross-sections with multiple Sections for a single layer must have unique names for each section"
            )
        named_sections[name] = section
    return named_sections


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: Literal[False] = False,
) -> Component: ...


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: Literal[True] = True,
) -> ComponentAllAngle: ...


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: bool = ...,
) -> AnyComponent: ...


def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: bool = False,
) -> AnyComponent:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component
    The CrossSection defines the layer numbers, widths and offsets

    Args:
        p: a path is a list of points (arc, straight, euler).
        cross_section: to extrude.
        layer: optional layer to extrude.
        width: optional width to extrude.
        simplify: Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting polygon \
                by more than the value listed here will be removed.
        all_angle: if True, the bend is drawn with a single euler curve.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    if cross_section is None and layer is None:
        raise ValueError("CrossSection or layer needed")

    if cross_section is not None and layer is not None:
        raise ValueError("Define only CrossSection or layer")

    if layer is not None and width is None:
        raise ValueError("Need to define layer width")
    elif width:
        assert layer is not None
        s = Section(
            width=width,
            layer=layer,
            port_names=("o1", "o2"),
            port_types=("optical", "optical"),
        )
        cross_section = CrossSection(sections=(s,))

    xsection_points: list[list[float | npt.NDArray[np.floating[Any]]]] = []
    c = ComponentAllAngle() if all_angle else Component()

    if isinstance(cross_section, Transition):
        deprecate("extrude", "extrude_transition")
        return extrude_transition(p, transition=cross_section)

    assert cross_section is not None

    x = get_cross_section(cross_section)

    layer = layer or x.layer
    layer = get_layer(layer)

    for section in x.sections:
        p_sec = p.copy()
        port_names = section.port_names
        port_types = section.port_types
        hidden = section.hidden

        offset_value: float | npt.NDArray[np.floating[Any]] = section.offset
        width_value: float | npt.NDArray[np.floating[Any]] = section.width
        width_function = section.width_function
        offset_function = section.offset_function
        layer = section.layer

        xsection_points.append([width_value, offset_value])

        if section.insets and section.insets != (0, 0):
            p_pts = p_sec.points

            # This excludes the first point, so length of output array is smaller by 1
            p_xy_segment_lengths = np.array(
                [
                    np.diff(p_pts[:, 0]),
                    np.diff(p_pts[:, 1]),
                ]
            ).T

            # Using the axis=1 makes output equivalent to [np.linalg.norm(p_xy_segment_lengths[i, :])
            #                                              for i
            #                                              in range(len(p_pts[:, 0]))]
            p_segment_lengths = np.linalg.norm(p_xy_segment_lengths, axis=1)

            p_segment_lengths_forward_cumsum = np.cumsum(
                p_segment_lengths
            )  # To get start inset idx & path length
            p_segment_lengths_reverse_cumsum = np.cumsum(
                p_segment_lengths[::-1]
            )  # To get stop inset idx & path length

            if all(section.insets[:] > p_segment_lengths_forward_cumsum[-1]):
                warnings.warn(
                    f"Cannot apply delay to Section '{section.name}', delay results in points outside "
                    f"of original path.",
                    stacklevel=3,
                )
                continue

            """
            Find forward cumsum idx (start_diff_idx), reverse cumsum idx (reversed_stop_diff_idx), and the reverse
            cumsum idx as indexed on the forward cumsum

            For the forward cumsum, this is the same as the idx of the vector that describes the path segment the
            start inset lies within or at the boundary of (due to the process of finding p_xy_segment_lengths, if
            the forward cumsum idx is 0, this corresponds to p_pts[1, :])

            The reverse cumsum idx, is the idx from the end of p_xy_segment_lengths (when the reverse
            cumsum idx is 0, this corresponds to p_pts[-2, :])
            """
            start_diff_idx = np.argwhere(
                p_segment_lengths_forward_cumsum >= section.insets[0]
            )[0, 0]
            reversed_stop_diff_idx = np.argwhere(
                p_segment_lengths_reverse_cumsum >= section.insets[1]
            )[0, 0]
            stop_diff_idx = (len(p_xy_segment_lengths) - 1) - reversed_stop_diff_idx

            """
            Find vectors describing the segments the insets lie within or at the boundary of. Also reverse direction of
            start vector to ensure vectors point from inside to out (this ensures that a positive/negative inset
            shortens/lengthens the segment's path, respectively)

            e.g.)   For a straight path (chosen because I don't know how to draw a representation of a curved path
                    with ASCII characters) with len(p_pts) == 7,
                    this implies len(p_xy_segment_lengths) == len(p_segment_lengths) == 6

                    so if start_diff_idx == 1 and stop_diff_idx == 5, the start and stop vectors would be

                    (0)  (1)  (2)  (3)  (4)  (5)
                    ---  ---  ---  ---  ---  ---
                         <--                 -->
                          ^(v_start)  (v_stop)^
            """
            v_start = -p_xy_segment_lengths[
                start_diff_idx, :
            ]  # Reversing vector direction so points inside-out
            v_stop = p_xy_segment_lengths[
                stop_diff_idx, :
            ]  # Vector already points inside-out

            v_start_direction = v_start / np.linalg.norm(v_start)  # Unit vector
            v_stop_direction = v_stop / np.linalg.norm(v_stop)  # Unit vector

            """
            The total path length up to the inside edge of v_start/v_stop (e.g. as shown above, the total path length
            from the left-most edge of the path to the right edge of segment 1) either accounts for more than or all of
            the total inset amount, depending on whether the inset location lies within the segment defined by
            v_start/v_stop or on it's inside edge.

            i.e.)   If the inset amount places the inset location *within* the segment defined by v_start/v_stop,
                    the total path length up to the inside edge of v_start/v_stop the over-accounts for the
                    total inset amount. The difference between this path length and the inset amount given by the user
                    then gives the length of v_start/v_stop needed to correctly position the edge of the inset path.

                    If the inset location lies on the inside edge of v_start/v_stop, the total path length up to the
                    inside edge of v_start/v_stop is equal to the entire inset amount. This means the difference
                    between the path length and the user provided inset amount will be 0 and v_start/v_stop needs to be
                    set to the zero vector
            """
            start_inset_remainder = (
                p_segment_lengths_forward_cumsum[start_diff_idx] - section.insets[0]
            )
            stop_inset_remainder = (
                p_segment_lengths_reverse_cumsum[reversed_stop_diff_idx]
                - section.insets[1]
            )

            # Correcting v_start/v_stops's length
            v_start_inset = v_start_direction * start_inset_remainder
            v_stop_inset = v_stop_direction * stop_inset_remainder

            # Translate v_start_inset/v_stop_inset back to their correct positions in the path, since the
            # process of finding the vectors that define each path segment translated them all to the origin
            new_start_point = v_start_inset + p_pts[start_diff_idx + 1, :]
            new_stop_point = v_stop_inset + p_pts[stop_diff_idx, :]

            _path_points = [new_start_point]
            _path_points.extend(p_pts[start_diff_idx + 1 : stop_diff_idx])
            _path_points.append(new_stop_point)

            p_sec = Path(np.array(_path_points, dtype=np.float64))

        if callable(offset_function):
            p_sec.offset(offset_function)
            offset_value = 0
        end_angle = p_sec.end_angle
        start_angle = p_sec.start_angle
        points = p_sec.points
        if callable(width_function):
            # Compute lengths
            dx: npt.NDArray[np.floating[Any]] | float = np.diff(p_sec.points[:, 0])
            dy: npt.NDArray[np.floating[Any]] | float = np.diff(p_sec.points[:, 1])
            lengths = np.cumsum(np.sqrt(dx**2 + dy**2))
            lengths = np.concatenate([[0], lengths])
            width_value = width_function(lengths / lengths[-1])

        assert width_value is not None

        dy = offset_value + width_value / 2

        points1 = p_sec.centerpoint_offset_curve(
            points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        dy = offset_value - width_value / 2

        points2 = p_sec.centerpoint_offset_curve(
            points,
            offset_distance=dy,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        if isinstance(simplify, bool):
            raise ValueError("simplify argument must be a number (e.g. 1e-3) or None")

        with_simplify = section.simplify or simplify

        if with_simplify:
            points1 = _simplify(points1, tolerance=with_simplify)
            points2 = _simplify(points2, tolerance=with_simplify)

        # Join points together
        points_poly = np.concatenate([points1, points2[::-1, :]])

        if not hidden and p_sec.length() > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        # Add port_names if they were specified
        if port_names[0] is not None:
            port_width = (
                width_value if isinstance(width_value, float) else width_value[0]
            )
            port_orientation = (p_sec.start_angle + 180) % 360
            center = np.average([points1[0], points2[0]], axis=0)
            face = [points1[0], points2[0]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            c.add_port(
                name=port_names[0],
                layer=layer,
                port_type=port_types[0],
                width=port_width,
                orientation=port_orientation,
                center=center,
                cross_section=x,
            )
        if port_names[1] is not None:
            port_width = (
                width_value if isinstance(width_value, float) else width_value[-1]
            )
            port_orientation = (p_sec.end_angle) % 360
            center = np.average([points1[-1], points2[-1]], axis=0)
            face = [points1[-1], points2[-1]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            c.add_port(
                name=port_names[1],
                layer=layer,
                port_type=port_types[1],
                width=port_width,
                center=center,
                orientation=port_orientation,
                cross_section=x,
            )

    c.info["length"] = float(np.round(p.length(), 3))

    for via in x.components_along_path:
        if via.offset:
            points_offset = p.centerpoint_offset_curve(
                points,
                offset_distance=via.offset,
                start_angle=start_angle,
                end_angle=end_angle,
            )
            _p = Path(points_offset)
        else:
            _p = p
        _ = c << along_path(
            p=_p, component=via.component, spacing=via.spacing, padding=via.padding
        )
    return c


def extrude_transition(p: Path, transition: Transition) -> Component:
    """Extrudes a path along a transition.

    Args:
        p: path to extrude.
        transition: transition to extrude along.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    c = Component()

    x1 = get_cross_section(transition.cross_section1)
    x2 = get_cross_section(transition.cross_section2)
    width_type = transition.width_type
    offset_type = transition.offset_type

    # if named, prefer name over layer
    named_sections1 = _get_named_sections(x1.sections)
    named_sections2 = _get_named_sections(x2.sections)

    names1 = list(named_sections1.keys())
    names2 = list(named_sections2.keys())

    common_sections = set(names1).intersection(names2)
    if not common_sections:
        raise ValueError(
            f"transition() found no common section names X1 {names1} and X2 {names2}"
        )

    # Compute relative distance of points along path p
    dx = np.diff(p.points[:, 0])
    dy = np.diff(p.points[:, 1])
    lengths = np.cumsum(np.sqrt(dx**2 + dy**2))
    lengths = np.concatenate([[0], lengths]) / lengths[-1]

    for section_name in common_sections:
        section1 = named_sections1[section_name]
        section2 = named_sections2[section_name]
        port_names = section1.port_names
        port_types = section1.port_types

        offset1 = section1.offset
        offset2 = section2.offset
        width1 = section1.width
        width2 = section2.width

        if offset_type == "linear":
            offset: Callable[[float], float | npt.NDArray[np.floating[Any]]] = (
                _linear_transition(offset1, offset2)
            )
        elif offset_type == "sine":
            offset = _sinusoidal_transition(offset1, offset2)
        elif offset_type == "parabolic":
            offset = _parabolic_transition(offset1, offset2)
        elif callable(offset_type):

            def offset_func(t: float) -> float:
                return offset_type(t, offset1, offset2)  # noqa: B023

            offset = offset_func
        else:
            raise NotImplementedError()

        if width_type == "linear":
            width: Callable[[float], float | npt.NDArray[np.floating[Any]]] = (
                _linear_transition(width1, width2)
            )
        elif width_type == "sine":
            width = _sinusoidal_transition(width1, width2)
        elif width_type == "parabolic":
            width = _parabolic_transition(width1, width2)
        elif callable(width_type):

            def width_func(t: float) -> float:
                return width_type(t, width1, width2)  # noqa: B023

            width = width_func
        else:
            raise NotImplementedError()

        if section1.layer != section2.layer:
            hidden = True
            layer1 = get_layer(section1.layer)
            layer2 = get_layer(section2.layer)
            layer = (layer1, layer2)
        else:
            hidden = False
            layer = get_layer(section1.layer)

        end_angle = p.end_angle
        start_angle = p.start_angle
        points = p.points
        width_value = width(lengths)
        offset_value = offset(lengths)

        points1 = p.centerpoint_offset_curve(
            points,
            offset_distance=offset_value + width_value / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        points2 = p.centerpoint_offset_curve(
            points,
            offset_distance=offset_value - width_value / 2,
            start_angle=start_angle,
            end_angle=end_angle,
        )

        if section1.simplify is not None and section2.simplify is not None:
            tolerance = min([section1.simplify, section2.simplify])
            points1 = _simplify(points1, tolerance=tolerance)
            points2 = _simplify(points2, tolerance=tolerance)

        # Join points together
        points_poly = np.concatenate([points1, points2[::-1, :]])

        layers = layer if hidden else [layer, layer]
        if not hidden and p.length() > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        # Add port_names if they were specified
        if port_names[0] is not None:
            port_width = width1
            port_orientation = (p.start_angle + 180) % 360
            assert not isinstance(offset_value, float)
            center = p.centerpoint_offset_curve(
                points[:2],
                offset_distance=offset_value[:2],
                start_angle=start_angle,
                end_angle=None,
            )[0]

            c.add_port(
                name=port_names[0],
                layer=get_layer(layers[0]),
                port_type=port_types[0],
                width=port_width,
                orientation=port_orientation,
                center=center,
                cross_section=x1,
            )
        if port_names[1] is not None:
            port_width = width2
            port_orientation = (p.end_angle) % 360
            assert not isinstance(offset_value, float)
            center = p.centerpoint_offset_curve(
                points[-2:],
                offset_distance=offset_value[-2:],
                start_angle=None,
                end_angle=end_angle,
            )[-1]

            c.add_port(
                name=port_names[1],
                layer=get_layer(layers[1]),
                port_type=port_types[1],
                width=port_width,
                center=center,
                orientation=port_orientation,
                cross_section=x2,
            )

    c.info["length"] = float(np.round(p.length(), 3))
    return c


def _rotated_delta(
    point: npt.NDArray[np.floating[Any]],
    center: npt.NDArray[np.floating[Any]],
    orientation: AngleInDegrees,
) -> npt.NDArray[np.floating[Any]]:
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
    return np.array(np.dot(delta, rot_mat))


def _cut_path_with_ray(  # type: ignore
    start_point: npt.NDArray[np.floating[Any]],
    start_angle: float | None,
    end_point: npt.NDArray[np.floating[Any]],
    end_angle: float | None,
    path: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Cuts or extends floating[Any] path given a point and angle to project."""
    import shapely.geometry as sg
    import shapely.ops

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
    distances: list[float] = []
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
    # when trimming the start, start counting at the intersection point, then
    # add all subsequent points
    points = [np.array(intersections[0].coords[0])]
    points.extend(
        point
        for point in path[1:-1]
        if distances[0] < ls.project(sg.Point(point)) < distances[1]
    )
    points.append(np.array(intersections[1].coords[0]))
    return np.array(points)


def arc(
    radius: float | None = 10.0,
    angle: float = 90,
    npoints: int | None = None,
    start_angle: float = -90,
) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        npoints: Number of points used per 360 degrees. Defaults to pdk.bend_points_distance.
        start_angle: initial angle of the curve for drawing, default -90 degrees.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.arc(radius=10, angle=45)
        p.plot()

    """
    from gdsfactory.pdk import get_active_pdk

    PDK = get_active_pdk()

    if not radius:
        raise ValueError("arc() requires a radius argument")

    npoints = npoints or abs(int(angle / 360 * radius / PDK.bend_points_distance / 2))
    npoints = max(int(npoints), int(360 / angle) + 1)

    t = np.linspace(
        start_angle * np.pi / 180, (angle + start_angle) * np.pi / 180, npoints
    )
    x = radius * np.cos(t)
    y = radius * (np.sin(t) + 1)
    points = np.array((x, y)).T * np.sign(angle)

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    P.start_angle = start_angle + 90
    P.end_angle = start_angle + angle + 90
    return P


def _fresnel(
    R0: float, s: float, num_pts: int, n_iter: int = 8
) -> npt.NDArray[np.floating[Any]]:
    """Fresnel integral using a series expansion.

    Args:
        R0: Initial radius of curvature.
        s: Length of the curve.
        num_pts: Number of points to generate.
        n_iter: Number of iterations to use in the series expansion.
    """
    t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
    x = np.zeros(num_pts)
    y = np.zeros(num_pts)

    for n in range(n_iter):
        x += (-1) ** n * t ** (4 * n + 1) / (math.factorial(2 * n) * (4 * n + 1))
        y += (-1) ** n * t ** (4 * n + 3) / (math.factorial(2 * n + 1) * (4 * n + 3))

    return np.array([np.sqrt(2) * R0 * x, np.sqrt(2) * R0 * y])


def euler(
    radius: float = 10,
    angle: float = 90,
    p: float = 0.5,
    use_eff: bool = False,
    npoints: int | None = None,
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
        use_eff: If False: `radius` is the minimum radius of curvature of the bend. \
                If True: The curve will be scaled such that the endpoints match an \
                arc with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.euler(radius=10, angle=45, p=1, use_eff=True, npoints=720)
        p.plot()

    """
    from gdsfactory.pdk import get_active_pdk

    if not radius:
        raise ValueError("euler() requires a radius argument")

    if (p < 0) or (p > 1):
        raise ValueError("euler requires argument `p` be between 0 and 1")
    if p == 0:
        path = arc(radius=radius, angle=angle, npoints=npoints)
        path.info["Reff"] = radius
        path.info["Rmin"] = radius
        return path

    if angle < 0:
        mirror = True
        angle = np.abs(angle)
    else:
        mirror = False

    R0 = 1
    alpha = np.radians(angle)
    Rp = R0 / (np.sqrt(p * alpha))
    sp = float(R0 * np.sqrt(p * alpha))
    s0 = float(2 * sp + Rp * alpha * (1 - p))

    pdk = get_active_pdk()
    npoints = npoints or abs(int(angle / 360 * radius / pdk.bend_points_distance / 2))
    npoints = max(npoints, int(360 / angle) + 1)

    num_pts_euler = int(np.round(sp / (s0 / 2) * npoints))
    num_pts_arc = npoints - num_pts_euler

    # Ensure a minimum of 2 points for each euler/arc section
    if npoints <= 2:
        num_pts_euler = 0
        num_pts_arc = 2

    if num_pts_euler > 0:
        xbend1, ybend1 = _fresnel(R0, sp, num_pts_euler)
        xp, yp = xbend1[-1], ybend1[-1]
        dx = xp - Rp * np.sin(p * alpha / 2)
        dy = yp - Rp * (1 - np.cos(p * alpha / 2))
    else:
        xbend1 = ybend1 = np.asarray([], dtype=float)
        dx = 0
        dy = 0

    s = np.linspace(sp, s0 / 2, num_pts_arc)
    xbend2 = Rp * np.sin((s - sp) / Rp + p * alpha / 2) + dx
    ybend2 = Rp * (1 - np.cos((s - sp) / Rp + p * alpha / 2)) + dy

    x = np.concatenate([xbend1, xbend2[1:]])
    y = np.concatenate([ybend1, ybend2[1:]])
    points1 = np.array([x, y]).T
    points2 = np.flipud(np.array([x, -y]).T)

    points2 = rotate_points(points2, angle - 180)
    points2 += -points2[0, :] + points1[-1, :]

    points = np.concatenate([points1[:-1], points2])

    # Find y-axis intersection point to compute Reff
    start_angle = 180 * (angle < 0)
    end_angle = start_angle + angle
    dy = np.tan(np.radians(end_angle - 90)) * points[-1][0]
    Reff = points[-1][1] - dy
    Rmin = Rp

    # Fix degenerate condition at angle == 180
    if np.abs(180 - angle) < 1e-3:
        Reff = points[-1][1] / 2

    # Scale curve to either match Reff or Rmin
    scale = radius / Reff if use_eff else radius / Rmin
    points *= scale

    path = Path()

    # Manually add points & adjust start and end angles
    path.points = points
    path.start_angle = start_angle
    path.end_angle = end_angle
    path.info["Reff"] = Reff * scale
    path.info["Rmin"] = Rmin * scale
    if mirror:
        path.mirror((1, 0))
    return path


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
        min_bend_radius: Inner radius of the spiral.
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
        np.array(
            [
                (separation / np.pi * theta + min_bend_radius)
                * np.array((np.sin(theta), np.cos(theta)))
                for theta in np.linspace(0, number_of_loops * 2 * np.pi, npoints)
            ]
        )
    )


def _compute_segments(
    points: npt.NDArray[np.floating[Any]],
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.signedinteger[Any]],
]:
    points = np.asarray(points, dtype=float)
    normals = np.diff(points, axis=0)
    normals = (normals.T / np.linalg.norm(normals, axis=1)).T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    dtheta = dtheta - 360 * np.floor((dtheta + 180) / 360)
    return points, normals, ds, theta, dtheta


def smooth(
    points: npt.NDArray[np.floating[Any]],
    radius: float = 4.0,
    bend: PathFactory = euler,
    **kwargs: Any,
) -> Path:
    """Returns a smooth Path from a series of waypoints.

    Args:
        points: array-like[N][2] List of waypoints for the path to follow.
        radius: radius of curvature, passed to `bend`.
        bend: bend function that returns a path that round corners.
        kwargs: Extra keyword arguments that will be passed to `bend`.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.smooth(([0, 0], [0, 10], [10, 10]))
        p.plot()

    """
    if isinstance(points, Path):
        points = points.points

    points, normals, ds, theta, dtheta = _compute_segments(points)
    colinear_elements = np.concatenate([[False], np.abs(dtheta) < 1e-6, [False]])
    if np.any(colinear_elements):
        points, normals, ds, theta, dtheta = _compute_segments(
            points[~colinear_elements, :]
        )

    if np.any(np.abs(np.abs(dtheta) - 180) < 1e-6):
        raise ValueError(
            "smooth() received points which double-back on themselves"
            "--turns cannot be computed when going forwards then exactly backwards."
        )

    # FIXME add caching
    # Create arcs
    paths: list[Path] = []
    radii: list[float] = []
    for dt in dtheta:
        P = bend(radius=radius, angle=dt, **kwargs)
        chord = np.linalg.norm(P.points[-1, :] - P.points[0, :])
        r = (chord / 2) / np.sin(np.radians(dt / 2))
        r = np.abs(r)
        radii.append(r)
        paths.append(P)

    d = np.abs(np.array(radii) / np.tan(np.radians(180 - dtheta) / 2))
    encroachment = np.concatenate([[0], d]) + np.concatenate([d, [0]])
    if np.any(encroachment > ds):
        raise ValueError(
            "smooth(): Not enough distance between points to to fit curves."
            "Try reducing the radius or spacing the points out farther"
        )
    p1 = points[1:-1, :] - normals[:-1, :] * d[:, np.newaxis]

    # Move arcs into position
    new_points: list[npt.NDArray[np.floating[Any]]] = []
    new_points.append(np.array([points[0, :]]))
    for n in range(len(dtheta)):
        p = paths[n]
        p.rotate(theta[n] - 0)
        p.move(p1[n])
        new_points.append(p.points)
    new_points.append(np.array([points[-1, :]]))
    new_points_np = np.concatenate(new_points)

    path = Path()
    path.rotate(float(theta[0]))
    path.append(new_points_np)
    path.move(points[0, :])
    return path


__all__ = [
    "Path",
    "along_path",
    "arc",
    "euler",
    "extrude",
    "extrude_transition",
    "smooth",
    "spiral_archimedean",
    "straight",
    "transition",
    "transition_adiabatic",
]

if __name__ == "__main__":
    import gdsfactory as gf

    # P = gf.path.arc(angle=30)
    # P.dmovey(10)
    # s0 = gf.Section(
    #     width=1, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2")
    # )
    # s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
    # x1 = gf.CrossSection(sections=(s0, s1))
    # x1 = gf.cross_section.rib
    # layer = (1, 0)
    # s1 = gf.Section(width=5, layer=layer, port_names=("o1", "o2"), name="core")
    # s2 = gf.Section(width=50, layer=layer, port_names=("o1", "o2"), name="core")
    # xs1 = gf.CrossSection(sections=(s1,))
    # xs2 = gf.CrossSection(sections=(s2,))
    # trans12 = gf.path.transition(
    #     cross_section1=xs1, cross_section2=xs2, width_type="linear"
    # )
    # trans21 = gf.path.transition(
    #     cross_section1=xs2, cross_section2=xs1, width_type="linear"
    # )
    # WG4Path = gf.Path()
    # WG4Path.append(gf.path.straight(length=100, npoints=2))
    # c1 = gf.path.extrude_transition(WG4Path, trans12)

    p = gf.path.straight()
    p += gf.path.arc(10)
    p += gf.path.straight()
    p.movey(10)

    # Define a cross-section with a via
    via = gf.cross_section.ComponentAlongPath(
        component=gf.c.rectangle(size=(1, 1), centered=True), spacing=5, padding=2
    )
    s = gf.Section(
        width=0.5, offset=0, layer=(1, 0), port_names=("in", "out"), name="core"
    )
    x = gf.CrossSection(sections=(s,), components_along_path=(via,))

    # Combine the path with the cross-section
    # c = gf.path.extrude(p, cross_section=x)
    # assert c

    s = gf.Section(
        width=2, offset=0, layer=(1, 0), port_names=("in", "out"), name="core"
    )
    x2 = gf.CrossSection(sections=(s,), components_along_path=(via,))
    t = gf.path.transition(x, x2, width_type="linear")
    c = gf.path.extrude_transition(p, t)

    # c = gf.path.extrude(P, x1)
    # print(hash(P))
    # P.plot()

    # ref = c.ref()
    # print(ref)
    # s2 = gf.Section(
    #     width=0.5, offset=0, layer=(1, 0), name="core", port_names=("o1", "o2")
    # )
    # s3 = gf.Section(width=2.0, offset=0, layer=(3, 0), name="slab")
    # x2 = gf.CrossSection(sections=(s2, s3))
    # t = gf.path.transition(x1, x2, width_type="linear")
    # c = gf.path.extrude(P, t)
    c.show()
