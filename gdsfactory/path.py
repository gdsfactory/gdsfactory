"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using a native cross-section returning a Component.
The cross-section defines the layer numbers, widths and offsets.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar, cast, overload

import kfactory as kf
import klayout.db as kdb
import numpy as np
import numpy.typing as npt
from kfactory.conf import CheckInstances
from kfactory.geometry import UMGeometricObject
from numpy import mod
from scipy import optimize

import gdsfactory as gf
from gdsfactory._cell import cell
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.component_layout import (
    rotate_points,
)
from gdsfactory.cross_section import (
    AsymmetricExtrusionSpec,
    CrossSection,
    ExtrusionSection,
    ExtrusionSpec,
    SectionReference,
    SymmetricExtrusionSpec,
    Transition,
    TransitionAsymmetric,
    TransitionSection,
)
from gdsfactory.pdk import get_layer_tuple
from gdsfactory.typings import (
    AngleInDegrees,
    AnyComponent,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    WidthTypes,
)


def _simplify(
    points: npt.NDArray[np.floating[Any]], tolerance: float
) -> npt.NDArray[np.floating[Any]]:
    import shapely.geometry as sg

    ls = sg.LineString(points)
    ls_simple = ls.simplify(tolerance=tolerance)
    return np.asarray(ls_simple.coords)


def reflect_points(
    points: npt.NDArray[np.floating[Any]],
    p1: tuple[float, float] = (0, 0),
    p2: tuple[float, float] = (1, 0),
) -> npt.NDArray[np.float64]:
    """Reflects points across the line formed by p1 and p2.

    from https://github.com/amccaugh/phidl/pull/181

    ``points`` may be input as either single points [1,2] or array-like[N][2],
    and will return in kind.

    Args:
        points: array-like[N][2]
        p1: Coordinates of the start of the reflecting line.
        p2: Coordinates of the end of the reflecting line.

    Returns:
        A new set of points that are reflected across ``p1`` and ``p2``.
    """
    original_shape = np.shape(points)
    points = np.atleast_2d(points)
    return_single_point = len(original_shape) == 1
    p1_array = np.asarray(p1)
    p2_array = np.asarray(p2)

    line_vec = p2_array - p1_array
    line_vec_norm = np.linalg.norm(line_vec) ** 2

    # Compute reflection
    proj = np.sum(line_vec * (points - p1_array), axis=-1, keepdims=True)
    reflected_points = (
        2 * (p1_array + (p2_array - p1_array) * proj / line_vec_norm) - points
    )
    return reflected_points[0] if return_single_point else reflected_points  # type: ignore[no-any-return]


class Path(UMGeometricObject):
    """You can extrude a Path with a native cross-section to create a Component.

    Parameters:
        path: array-like[N][2], Path, or list of Paths.

    """

    def __init__(
        self,
        path: npt.NDArray[np.floating[Any]]
        | Path
        | list[tuple[float, float]]
        | None = None,
        start_angle: float | None = None,
        end_angle: float | None = None,
    ) -> None:
        """Initializes a Path.

        Args:
            path: array-like[N][2], Path, or list of Paths.
            start_angle: optional angle in degrees at the start of the path.
                Overrides the angle inferred from the points.
            end_angle: optional angle in degrees at the end of the path.
                Overrides the angle inferred from the points.
        """
        self.points: npt.NDArray[np.floating[Any]] = np.array(
            [[0, 0]], dtype=np.float64
        )
        self.start_angle: float = 0
        self.end_angle: float = 0
        self.info: dict[str, Any] = {}
        if path is not None:
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
                if len(self.points) > 1:
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
        if start_angle is not None:
            self.start_angle = mod(start_angle, 360)
        if end_angle is not None:
            self.end_angle = mod(end_angle, 360)

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
    def kcl(self) -> kf.KCLayout:
        return gf.kcl

    def transform(
        self,
        trans: kdb.Trans | kdb.DTrans | kdb.ICplxTrans | kdb.DCplxTrans,
        /,
    ) -> Any:
        """Transforms the Path in place.

        Applies a transformation as a magnification, then a mirroring at
        the x-axis, then a rotation, then a displacement, following KLayout.

        Args:
            trans: the transformation to apply.
        """
        if isinstance(trans, kdb.DCplxTrans):
            trans_ = trans
        elif isinstance(trans, kdb.DTrans):
            trans_ = kdb.DCplxTrans(trans)
        elif isinstance(trans, kdb.Trans):
            trans_ = kdb.DCplxTrans(trans.to_dtype(gf.kcl.dbu))
        else:
            trans_ = trans.to_itrans(gf.kcl.dbu)

        new_points = np.asarray(self.points, dtype=np.float64)

        if trans_.mag != 1:
            new_points = new_points * trans_.mag

        if trans_.mirror:
            new_points = new_points * np.array([1.0, -1.0])

        if trans_.angle != 0:
            angle_rad = np.radians(trans_.angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            rotation_matrix = np.array(
                [
                    [cos_angle, sin_angle],
                    [-sin_angle, cos_angle],
                ]
            )
            new_points = np.dot(new_points, rotation_matrix)

        new_points = new_points + np.array([trans_.disp.x, trans_.disp.y])

        self.points = new_points
        sign = -1 if trans_.mirror else 1
        self.start_angle = mod(sign * self.start_angle + trans_.angle, 360)
        self.end_angle = mod(sign * self.end_angle + trans_.angle, 360)

    def dbbox(self, layer: int | None = None) -> kdb.DBox:
        return kdb.DBox(*self.bbox_np().flatten())

    def ibbox(self, layer: int | None = None) -> kdb.Box:
        return kdb.Box(*map(gf.kcl.to_dbu, self.bbox_np().flatten()))

    def bbox_np(self) -> npt.NDArray[np.float64]:
        """Returns the bounding box of the Path as a numpy array."""
        return np.array(
            [
                (np.min(self.points[:, 0]), np.min(self.points[:, 1])),
                (np.max(self.points[:, 0]), np.max(self.points[:, 1])),
            ],
            dtype=np.float64,
        )

    def append(
        self,
        path: npt.NDArray[np.floating[Any]]
        | Path
        | list[Path]
        | list[tuple[float, float]],
    ) -> Path:
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
            and (np.shape(path)[1] == 2)  # type: ignore[arg-type]
        ):
            points = np.asarray(path, dtype=np.float64)
            start_angle, end_angle = 0, 0
            if len(points) > 1:
                nx1, ny1 = points[1] - points[0]
                start_angle = np.arctan2(ny1, nx1) / np.pi * 180
                nx2, ny2 = points[-1] - points[-2]
                end_angle = np.arctan2(ny2, nx2) / np.pi * 180
        elif isinstance(path, list):
            for p in path:
                self.append(p)  # type: ignore[arg-type]
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

    def offset(self, offset: float | Callable[[float], float] = 0) -> Path:
        """Offsets Path so that it follows the Path centerline plus an offset.

        The offset can either be a fixed value, or a function
        of the form my_offset(t) where t goes from 0->1

        Args:
            offset: int or float, callable. Magnitude of the offset
        """
        if offset == 0:
            points = self.points
            start_angle = self.start_angle
            end_angle = self.end_angle
        elif callable(offset):
            # Compute lengths
            dx = np.diff(self.points[:, 0])
            dy = np.diff(self.points[:, 1])
            lengths = np.cumsum(np.sqrt((dx) ** 2 + (dy) ** 2))
            lengths = np.concatenate([[0], lengths])
            # Create list of offset points and perform offset
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=offset(lengths / lengths[-1]),
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            # Numerically compute start and end angles
            tol = 1e-6
            ds = tol / lengths[-1]
            ny1 = offset(ds) - offset(0)
            start_angle = np.arctan2(-ny1, tol) / np.pi * 180 + self.start_angle
            ny2 = offset(1) - offset(1 - ds)
            end_angle = np.arctan2(-ny2, tol) / np.pi * 180 + self.end_angle
        else:
            points = self.centerpoint_offset_curve(
                self.points,
                offset_distance=cast(float, offset),  # type: ignore[redundant-cast]
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            start_angle = self.start_angle
            end_angle = self.end_angle

        self.points = points
        self.start_angle = start_angle
        self.end_angle = end_angle
        return self

    def centerpoint_offset_curve(
        self,
        points: npt.NDArray[np.floating[Any]],
        offset_distance: float | Sequence[float] | npt.NDArray[np.floating[Any]],
        start_angle: float | None = None,
        end_angle: float | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Creates a offset curve computing the centerpoint offset of x and y points.

        Args:
            points: array-like[N][2] The points to be offset.
            offset_distance: array-like[N] The distance to offset the points.
            start_angle: float or None The angle at the start of the path.
            end_angle: float or None The angle at the end of the path.

        """
        cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
        return _offset_curve_from_directions(
            points,
            offset_distance,
            cos_mid,
            sin_mid,
            sin_half,
            start_angle=start_angle,
            end_angle=end_angle,
        )

    def _parametric_offset_curve(
        self,
        points: npt.NDArray[np.floating[Any]],
        offset_distance: npt.NDArray[np.floating[Any]],
        start_angle: float | None = None,
        end_angle: float | None = None,
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
        return float(np.round(np.sum(np.sqrt((dx) ** 2 + (dy) ** 2)), 3))

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

    def __eq__(self, other: object) -> bool:
        """Check if two Path instances are equal."""
        if not isinstance(other, Path):
            return False
        return (
            np.array_equal(self.points, other.points)
            and self.start_angle == other.start_angle
            and self.end_angle == other.end_angle
        )

    def hash_geometry(self, precision: float = 1e-4) -> int:
        """Computes an SHA1 hash of the points in the Path and the start_angle and end_angle.

        Args:
            precision: Rounding precision for the the objects in the Component. For instance, \
                    a precision of 1e-2 will round a point at (0.124, 1.748) to (0.12, 1.75)

        Returns:
            str Hash result in the form of an SHA1 hex digest string.

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

    def plot(self) -> None:
        """Plot path in matplotlib.

        Example:
            ```python
            import gdsfactory as gf

            p = gf.path.euler(radius=10)
            p.plot()
            ```
        """
        import matplotlib.pyplot as plt

        plt.plot(self.points[:, 0], self.points[:, 1])
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: Literal[False] = False,
        register_cross_section: bool = False,
    ) -> Component: ...

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: Literal[True] = True,
        register_cross_section: bool = False,
    ) -> ComponentAllAngle: ...

    @overload
    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: bool = True,
        register_cross_section: bool = False,
    ) -> AnyComponent: ...

    def extrude(
        self,
        cross_section: CrossSectionSpec | None = None,
        layer: LayerSpec | None = None,
        width: float | None = None,
        simplify: float | None = None,
        all_angle: bool = False,
        register_cross_section: bool = False,
        extrusion_spec: ExtrusionSpec | None = None,
        port_names: tuple[str | None, str | None] = ("o1", "o2"),
        port_types: tuple[str, str] = ("optical", "optical"),
        add_bbox: bool = False,
        port_cross_section: bool = True,
    ) -> AnyComponent:
        """Returns a component by extruding a path with a native cross-section.

        A path can be extruded using any native cross-section returning a Component.
        The cross-section defines the layer numbers, widths and offsets.

        Args:
            cross_section: to extrude.
            layer: optional layer.
            width: optional width in um.
            simplify: Tolerance value for the simplification algorithm. \
                    All points that can be removed without changing the resulting polygon\
                    by more than the value listed here will be removed.

            all_angle: if True, the bend is drawn with a single euler curve.
            register_cross_section: if True, the cross_section factory is registered in the active PDK.
            extrusion_spec: metadata for the positional strips returned by the
                native cross-section.
            port_names: names for the main strip ports when no extrusion spec
                supplies them.
            port_types: types for the main strip ports when no extrusion spec
                supplies them.
            add_bbox: if True, add the native cross-section bounding-box layers.
            port_cross_section: if False, omit native cross-section metadata from
                generated ports for temporary legacy-compatible composition.

        Example:
            ```python
            import gdsfactory as gf

            p = gf.path.euler(radius=10)
            c = p.extrude(layer=(1, 0), width=0.5)
            c.plot()
            ```
        """
        return extrude(
            p=self,
            cross_section=cross_section,
            layer=layer,
            width=width,
            simplify=simplify,
            all_angle=all_angle,
            register_cross_section=register_cross_section,
            extrusion_spec=extrusion_spec,
            port_names=port_names,
            port_types=port_types,
            add_bbox=add_bbox,
            port_cross_section=port_cross_section,
        )

    @overload
    def extrude_transition(
        self,
        transition: Transition | TransitionAsymmetric,
        all_angle: Literal[False] = False,
        add_bbox: bool = False,
        port_names: tuple[str | None, str | None] = ("o1", "o2"),
        port_types: tuple[str, str] = ("optical", "optical"),
    ) -> Component: ...

    @overload
    def extrude_transition(
        self,
        transition: Transition | TransitionAsymmetric,
        all_angle: Literal[True] = True,
        add_bbox: bool = False,
        port_names: tuple[str | None, str | None] = ("o1", "o2"),
        port_types: tuple[str, str] = ("optical", "optical"),
    ) -> ComponentAllAngle: ...

    @overload
    def extrude_transition(
        self,
        transition: Transition | TransitionAsymmetric,
        all_angle: bool = True,
        add_bbox: bool = False,
        port_names: tuple[str | None, str | None] = ("o1", "o2"),
        port_types: tuple[str, str] = ("optical", "optical"),
    ) -> AnyComponent: ...

    def extrude_transition(
        self,
        transition: Transition | TransitionAsymmetric,
        all_angle: bool = False,
        add_bbox: bool = False,
        port_names: tuple[str | None, str | None] = ("o1", "o2"),
        port_types: tuple[str, str] = ("optical", "optical"),
    ) -> AnyComponent:
        """Extrudes a path along a transition.

        Allows different transition methods for the upper and lower edges.

        Args:
            transition: Transition or TransitionAsymmetric object describing the
                cross-sections and default transition types.
            all_angle: if True, returns a ComponentAllAngle.
            add_bbox: if True, add the first profile's native bounding-box layers.
            port_names: names for the two main transition ports.
            port_types: types for the two main transition ports.

        Returns:
            AnyComponent: The extruded component with the specified transition methods
                for each edge.
        """
        return extrude_transition(
            p=self,
            transition=transition,
            all_angle=all_angle,
            add_bbox=add_bbox,
            port_names=port_names,
            port_types=port_types,
        )

    def copy(self) -> Path:
        """Returns a copy of the Path."""
        p = Path()
        p.info = self.info.copy()
        p.points = np.array(self.points)
        p.start_angle = self.start_angle
        p.end_angle = self.end_angle
        return p

    def mirror(
        self, p1: tuple[float, float] = (0, 1), p2: tuple[float, float] = (0, 0)
    ) -> Path:
        """Mirrors the Path across the line formed between the two specified points.

        ``points`` may be input as either single points [1,2]
        or array-like[N][2], and will return in kind.

        Args:
            p1: First point of the line.
            p2: Second point of the line.
        """
        self.points = reflect_points(self.points, p1, p2)
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / np.pi
        if self.start_angle is not None:
            self.start_angle = mod(2 * angle - self.start_angle, 360)
        if self.end_angle is not None:
            self.end_angle = mod(2 * angle - self.end_angle, 360)
        return self

    def invert(self) -> Path:
        """Inverts the Path by reversing the order of its points.

        The shape of the path is unchanged, but its start becomes its end and vice versa.
        """
        self.points = self.points[::-1]
        self.start_angle, self.end_angle = (
            mod(self.end_angle + 180, 360),
            mod(self.start_angle + 180, 360),
        )
        return self


PathFactory = Callable[..., Path]
T = TypeVar("T", float, npt.NDArray[np.floating[Any]])


def _sinusoidal_transition(y1: float, y2: float) -> Callable[[T], T]:
    dy = y2 - y1

    def sine(t: T) -> T:
        return cast(
            T,
            np.add(
                y1, np.multiply(np.subtract(1, np.cos(np.multiply(np.pi, t))), dy / 2)
            ),
        )

    return sine


def _parabolic_transition(y1: float, y2: float) -> Callable[[T], T]:
    dy = y2 - y1

    def parabolic(t: T) -> T:
        return cast(T, np.add(y1, np.multiply(np.sqrt(t), dy)))

    return parabolic


def _linear_transition(y1: float, y2: float) -> Callable[[T], T]:
    dy = y2 - y1

    def linear(t: T) -> T:
        return cast(T, np.add(y1, np.multiply(t, dy)))

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
    return lambda t: y1 + (y2 - y1) * t**exp


def _resolve_transition_cross_section(cross_section: CrossSectionSpec) -> CrossSection:
    """Resolve a transition profile to a native cross-section."""
    from gdsfactory.pdk import get_cross_section

    return get_cross_section(cross_section)


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
    from scipy.integrate import odeint

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
    extrusion_spec: SymmetricExtrusionSpec | None = None,
    core_width_profile: Callable[[Any], Any] | None = None,
) -> Transition:
    """Return a transition between two symmetric native cross-sections.

    cross_section1: First native cross-section.
    cross_section2: Second native cross-section.
    width_type: 'sine', 'parabolic', 'linear' or Callable. type of width transition used \
                if any widths are different between the two input CrossSections.
    offset_type: 'sine', 'parabolic', 'linear' or Callable. Type of offset transition used \
                if any widths are different between the two input CrossSections. \
        extrusion_spec: explicit positional strip mapping when the enclosures differ.
    core_width_profile: optional actual core-width profile evaluated on
        normalized path positions. This belongs to the transition operation,
        not the native cross-section.

    """
    X1 = _resolve_transition_cross_section(cross_section1)
    X2 = _resolve_transition_cross_section(cross_section2)
    if not X1.is_symmetric() or not X2.is_symmetric():
        raise ValueError(
            "transition() only accepts symmetric native cross-sections. "
            "Use transition_asymmetric(..., extrusion_spec=...) for an "
            "asymmetric profile."
        )
    if X1.sections != X2.sections and extrusion_spec is None:
        raise ValueError(
            "Symmetric cross-sections with different enclosures require an "
            "explicit SymmetricExtrusionSpec."
        )

    return Transition(
        cross_section1=X1,
        cross_section2=X2,
        width_type=width_type,
        offset_type=offset_type,
        extrusion_spec=extrusion_spec,
        core_width_profile=core_width_profile,
    )


def transition_asymmetric(
    cross_section1: CrossSectionSpec,
    cross_section2: CrossSectionSpec,
    width_type1: WidthTypes | Callable[[float, float, float], float] = "sine",
    width_type2: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type1: WidthTypes | Callable[[float, float, float], float] = "sine",
    offset_type2: WidthTypes | Callable[[float, float, float], float] = "sine",
    extrusion_spec: AsymmetricExtrusionSpec | None = None,
    core_width_profile: Callable[[Any], Any] | None = None,
) -> TransitionAsymmetric:
    """Returns a smoothly-transitioning object between two CrossSections with asymmetric transitions.

    Args:
        cross_section1: First native cross-section.
        cross_section2: Second native cross-section.
        width_type1: transition type for lower edge width.
        width_type2: transition type for upper edge width.
        offset_type1: transition type for lower edge offset.
        offset_type2: transition type for upper edge offset.
        extrusion_spec: explicit mapping. Required when either profile is
            asymmetric.
        core_width_profile: optional actual core-width profile evaluated on
            normalized path positions.
    """
    X1 = _resolve_transition_cross_section(cross_section1)
    X2 = _resolve_transition_cross_section(cross_section2)
    if (not X1.is_symmetric() or not X2.is_symmetric()) and extrusion_spec is None:
        raise ValueError(
            "Asymmetric native cross-sections require an explicit "
            "AsymmetricExtrusionSpec."
        )

    return TransitionAsymmetric(
        cross_section1=X1,
        cross_section2=X2,
        width_type1=width_type1,
        width_type2=width_type2,
        offset_type1=offset_type1,
        offset_type2=offset_type2,
        extrusion_spec=extrusion_spec,
        core_width_profile=core_width_profile,
    )


@cell(check_instances=CheckInstances.IGNORE)
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
            component_ref.rotate(angle).move(start_pt + offset)
            next_component += spacing
        cum_dist += segment_length

    return c


def _get_extrusion_section(
    extrusion_spec: ExtrusionSpec | None,
    index: int,
    port_names: tuple[str | None, str | None],
    port_types: tuple[str, str],
) -> ExtrusionSection:
    """Return metadata for one positional native cross-section strip."""
    if extrusion_spec is not None and index < len(extrusion_spec.sections):
        return extrusion_spec.sections[index]
    if index == 0:
        return ExtrusionSection(port_names=port_names, port_types=port_types)
    return ExtrusionSection()


def _path_with_insets(p: Path, insets: tuple[float, float] | None) -> Path | None:
    """Return a path shortened by ``insets``."""
    if not insets or insets == (0, 0):
        return p

    segment_vectors = np.diff(p.points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if not len(segment_lengths) or total_length <= 0:
        return None
    if any(inset > total_length for inset in insets):
        warnings.warn(
            "Cannot apply extrusion inset because it extends beyond the path.",
            stacklevel=3,
        )
        return None

    forward = np.cumsum(segment_lengths)
    reverse = np.cumsum(segment_lengths[::-1])
    start_index = int(np.argwhere(forward >= insets[0])[0, 0])
    reverse_stop_index = int(np.argwhere(reverse >= insets[1])[0, 0])
    stop_index = len(segment_lengths) - 1 - reverse_stop_index
    start_vector = -segment_vectors[start_index]
    stop_vector = segment_vectors[stop_index]
    start_direction = start_vector / np.linalg.norm(start_vector)
    stop_direction = stop_vector / np.linalg.norm(stop_vector)
    start_remainder = forward[start_index] - insets[0]
    stop_remainder = reverse[reverse_stop_index] - insets[1]
    new_start = start_direction * start_remainder + p.points[start_index + 1]
    new_stop = stop_direction * stop_remainder + p.points[stop_index]
    points = [new_start]
    points.extend(p.points[start_index + 1 : stop_index])
    points.append(new_stop)
    return Path(np.asarray(points, dtype=np.float64))


def add_bbox_to_component(
    component: AnyComponent,
    cross_section: CrossSection,
    top: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    left: float | None = None,
) -> AnyComponent:
    """Add native cross-section bounding-box layers to ``component``."""
    from gdsfactory.add_padding import get_padding_points

    for layer, offset in cross_section.bbox_sections.items():
        points = get_padding_points(
            component=component,
            default=0,
            top=top if top is not None else offset,
            bottom=bottom if bottom is not None else offset,
            right=right if right is not None else offset,
            left=left if left is not None else offset,
        )
        component.add_polygon(points, layer=layer)
    return component


# Public spelling used by component call sites during the migration.
add_bbox = add_bbox_to_component


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: Literal[False] = False,
    register_cross_section: bool = False,
) -> Component: ...


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: Literal[True] = True,
    register_cross_section: bool = False,
) -> ComponentAllAngle: ...


@overload
def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: bool = ...,
    register_cross_section: bool = False,
) -> AnyComponent: ...


def extrude(
    p: Path,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    simplify: float | None = None,
    all_angle: bool = False,
    register_cross_section: bool = False,
    extrusion_spec: ExtrusionSpec | None = None,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    add_bbox: bool = False,
    port_cross_section: bool = True,
) -> AnyComponent:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using a native kfactory cross-section. Geometry comes
    from ``get_sections()`` and path-only metadata comes from ``extrusion_spec``.

    Args:
        p: a path is a list of points (arc, straight, euler).
        cross_section: to extrude.
        layer: optional layer to extrude.
        width: optional width to extrude.
        simplify: Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting polygon \
                by more than the value listed here will be removed.
        all_angle: if True, returns a ComponentAllAngle.
        register_cross_section: if True, registers the cross-section factory \
            used for extrusion in the global cross-section registry.
        extrusion_spec: metadata for the positional strips returned by the
            native cross-section.
        port_names: names for the main strip ports when no spec is supplied.
        port_types: types for the main strip ports when no spec is supplied.
        add_bbox: if True, add native bounding-box layers.
        port_cross_section: if False, omit native cross-section metadata from
            generated ports for temporary compatibility with composite cells.
    """
    from gdsfactory.pdk import get_cross_section, get_layer

    if (cross_section is None) == (layer is None):
        raise ValueError("Provide exactly one of 'cross_section' or 'layer'")
    if layer is not None and width is None:
        raise ValueError("When providing 'layer', 'width' must also be provided")

    if cross_section is not None:
        x = get_cross_section(cross_section)
        if width is not None and width != x.width:
            x = gf.cross_section.copy_cross_section(x, width=width)
    else:
        x = gf.cross_section.cross_section(
            width=cast("float", width),
            layer=cast("LayerSpec", layer),
        )

    c = ComponentAllAngle() if all_angle else Component()
    path_length = p.length()
    port_cross_section_value = x if port_cross_section else None

    sections = x.get_sections()

    _dir_cache: (
        tuple[
            npt.NDArray[np.floating[Any]],
            npt.NDArray[np.floating[Any]],
            npt.NDArray[np.floating[Any]],
        ]
        | None
    ) = None

    for index, native_section in enumerate(sections):
        extrusion_section = _get_extrusion_section(
            extrusion_spec,
            index,
            port_names=port_names,
            port_types=port_types,
        )
        p_sec = p
        if extrusion_section.insets and extrusion_section.insets != (0, 0):
            p_sec = _path_with_insets(p, extrusion_section.insets)
            if p_sec is None:
                continue
        path_changed = p_sec is not p
        section_min = float(native_section.section_min)
        section_max = float(native_section.section_max)
        offset_value = (section_min + section_max) / 2
        width_value = section_max - section_min
        layer = get_layer(native_section.layer)
        points = p_sec.points
        start_angle = p_sec.start_angle
        end_angle = p_sec.end_angle

        dy1 = offset_value + width_value / 2
        dy2 = offset_value - width_value / 2

        if path_changed:
            # Path was modified (insets or offset_function), compute fresh directions
            cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
            _dir_cache = None
        elif _dir_cache is not None:
            cos_mid, sin_mid, sin_half = _dir_cache
        else:
            cos_mid, sin_mid, sin_half = _compute_offset_directions(points)
            _dir_cache = (cos_mid, sin_mid, sin_half)

        points1, points2 = _apply_offsets(
            points,
            dy1,
            dy2,
            cos_mid,
            sin_mid,
            sin_half,
            start_angle=start_angle,
            end_angle=end_angle,
        )
        if isinstance(simplify, bool):
            raise ValueError("simplify argument must be a number (e.g. 1e-3) or None")

        with_simplify = extrusion_section.simplify or simplify

        if with_simplify:
            points1 = _simplify(points1, tolerance=with_simplify)
            points2 = _simplify(points2, tolerance=with_simplify)

        # Join points together
        points_poly = np.concatenate([points1, points2[::-1, :]])
        # Unchanged sections use the original path, so this preserves the old
        # per-section threshold without recomputing the same length each time.
        section_length = p_sec.length() if path_changed else path_length

        if not extrusion_section.hidden and section_length > 1e-3:
            c.add_polygon(points_poly, layer=layer)

        # Add port_names if they were specified
        if extrusion_section.port_names[0]:
            port_width = width_value
            port_orientation = (p_sec.start_angle + 180) % 360
            center = np.average([points1[0], points2[0]], axis=0)
            face = [points1[0], points2[0]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            c.add_port(
                name=extrusion_section.port_names[0],
                layer=layer,
                port_type=extrusion_section.port_types[0],
                width=port_width,
                orientation=port_orientation,
                center=(float(center[0]), float(center[1])),
                cross_section=port_cross_section_value,
                register_cross_section=register_cross_section,
            )
        if extrusion_section.port_names[1]:
            port_width = width_value
            port_orientation = (p_sec.end_angle) % 360
            center = np.average([points1[-1], points2[-1]], axis=0)
            face = [points1[-1], points2[-1]]
            face = [_rotated_delta(point, center, port_orientation) for point in face]

            c.add_port(
                name=extrusion_section.port_names[1],
                layer=layer,
                port_type=extrusion_section.port_types[1],
                width=port_width,
                center=(float(center[0]), float(center[1])),
                orientation=port_orientation,
                cross_section=port_cross_section_value,
                register_cross_section=register_cross_section,
            )

    c.info["length"] = path_length

    if add_bbox:
        add_bbox_to_component(c, x)

    for via in extrusion_spec.components_along_path if extrusion_spec else ():
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


@overload
def extrude_transition(
    p: Path,
    transition: Transition | TransitionAsymmetric,
    all_angle: Literal[False] = False,
    add_bbox: bool = False,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    port_cross_section: bool = True,
) -> Component: ...


@overload
def extrude_transition(
    p: Path,
    transition: Transition | TransitionAsymmetric,
    all_angle: Literal[True] = True,
    add_bbox: bool = False,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    port_cross_section: bool = True,
) -> ComponentAllAngle: ...


@overload
def extrude_transition(
    p: Path,
    transition: Transition | TransitionAsymmetric,
    all_angle: bool = True,
    add_bbox: bool = False,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    port_cross_section: bool = True,
) -> AnyComponent: ...


def extrude_transition(
    p: Path,
    transition: Transition | TransitionAsymmetric,
    all_angle: bool = False,
    add_bbox: bool = False,
    port_names: tuple[str | None, str | None] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    port_cross_section: bool = True,
) -> AnyComponent:
    """Extrude a path between native cross-sections.

    Symmetric profiles with identical enclosures are inferred positionally. A
    symmetric profile with a different enclosure must provide a
    SymmetricExtrusionSpec; asymmetric profiles must provide an
    AsymmetricExtrusionSpec. No named section matching is performed.
    """
    from gdsfactory.pdk import get_layer

    if not isinstance(transition, Transition | TransitionAsymmetric):
        raise TypeError(
            f"Expected Transition or TransitionAsymmetric, got {type(transition).__name__}"
        )

    x1 = _resolve_transition_cross_section(transition.cross_section1)
    x2 = _resolve_transition_cross_section(transition.cross_section2)
    is_symmetric = x1.is_symmetric() and x2.is_symmetric()

    if isinstance(transition, TransitionAsymmetric):
        width_type1 = transition.width_type1
        width_type2 = transition.width_type2
        offset_type1 = transition.offset_type1
        offset_type2 = transition.offset_type2
        extrusion_spec = transition.extrusion_spec
        core_width_profile = transition.core_width_profile
    else:
        width_type1 = transition.width_type
        width_type2 = transition.width_type
        offset_type1 = transition.offset_type
        offset_type2 = transition.offset_type
        extrusion_spec = transition.extrusion_spec
        core_width_profile = transition.core_width_profile

    if not is_symmetric and not isinstance(extrusion_spec, AsymmetricExtrusionSpec):
        raise ValueError(
            "Asymmetric native cross-sections require an explicit "
            "AsymmetricExtrusionSpec."
        )
    symmetric_spec_allowed = isinstance(
        extrusion_spec, (type(None), SymmetricExtrusionSpec)
    )
    asymmetric_spec_allowed = isinstance(
        transition, TransitionAsymmetric
    ) and isinstance(extrusion_spec, AsymmetricExtrusionSpec)
    if is_symmetric and not (symmetric_spec_allowed or asymmetric_spec_allowed):
        raise TypeError("Symmetric native profiles require SymmetricExtrusionSpec")

    def layer_key(layer: Any) -> tuple[int, int]:
        return get_layer_tuple(layer)

    def _ref_key(ref: SectionReference) -> tuple[tuple[int, int], int]:
        return layer_key(ref.layer), ref.index

    def refs(xs: Any) -> tuple[SectionReference, ...]:
        counts: dict[tuple[int, int], int] = {}
        result: list[SectionReference] = []
        for section in xs.get_sections():
            key = layer_key(section.layer)
            occurrence = counts.get(key, 0)
            counts[key] = occurrence + 1
            result.append(SectionReference(layer=key, index=occurrence))
        return tuple(result)

    refs1 = refs(x1)
    refs2 = refs(x2)
    indices1 = {_ref_key(ref): index for index, ref in enumerate(refs1)}
    indices2 = {_ref_key(ref): index for index, ref in enumerate(refs2)}

    mappings: tuple[TransitionSection, ...]
    if extrusion_spec is not None and extrusion_spec.sections:
        mappings = extrusion_spec.sections
    else:
        if not is_symmetric:
            raise ValueError(
                "Asymmetric native cross-sections require explicit strip mappings."
            )
        keys1 = list(indices1)
        keys2 = set(indices2)
        if set(keys1) != keys2:
            raise ValueError(
                "Symmetric native cross-sections have unmatched layer/section "
                "occurrences; provide SymmetricExtrusionSpec."
            )
        mappings = tuple(
            TransitionSection(
                start=refs1[indices1[key]],
                end=refs2[indices2[key]],
                extrusion=(
                    ExtrusionSection(port_names=port_names, port_types=port_types)
                    if indices1[key] == 0
                    else ExtrusionSection()
                ),
            )
            for key in keys1
        )

    def endpoint_index(
        ref: SectionReference | None, indices: dict[Any, int]
    ) -> int | None:
        if ref is None:
            return None
        try:
            return indices[_ref_key(ref)]
        except KeyError as error:
            raise ValueError(
                f"Transition section reference {ref} is not present"
            ) from error

    def transition_function(
        transition_type: WidthTypes | Callable[[float, float, float], float],
        value1: float,
        value2: float,
    ) -> Callable[[npt.NDArray[np.float64]], Any]:
        if transition_type == "linear":
            return _linear_transition(value1, value2)
        if transition_type == "sine":
            return _sinusoidal_transition(value1, value2)
        if transition_type == "parabolic":
            return _parabolic_transition(value1, value2)
        if callable(transition_type):
            return lambda t: cast(Any, transition_type(t, value1, value2))
        raise ValueError(f"Unsupported transition type {transition_type!r}")

    dx = np.diff(p.points[:, 0])
    dy = np.diff(p.points[:, 1])
    segment_lengths = np.sqrt(dx**2 + dy**2)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0:
        raise ValueError("Cannot extrude a transition along a zero-length path")
    lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)]) / total_length
    points = p.points
    c = ComponentAllAngle() if all_angle else Component()
    same_enclosure = (
        is_symmetric and x1.sections == x2.sections and extrusion_spec is None
    )

    def fixed_symmetric_bounds(
        section: Any,
        core_width: npt.NDArray[np.float64] | float,
        section_index: int,
    ) -> tuple[Any, Any]:
        """Keep an identical symmetric enclosure relative to the tapering core."""
        if section_index == 0:
            return 0.0, core_width
        half = x1.width / 2
        section_min = float(section.section_min)
        section_max = float(section.section_max)
        half_new = np.asarray(core_width) / 2
        if abs(section_min + section_max) < 2 * x1.kcl.dbu:
            d_max = max(abs(section_max) - half, 0.0)
            outer = half_new + d_max
            return -outer, outer
        if section_max <= 0:
            d_min = -section_max - half
            d_max = -section_min - half
            return -(half_new + d_max), -(half_new + d_min)
        d_min = section_min - half
        d_max = section_max - half
        return half_new + d_min, half_new + d_max

    for mapping in mappings:
        index1 = endpoint_index(mapping.start, indices1)
        index2 = endpoint_index(mapping.end, indices2)
        if index1 is None and index2 is None:
            raise ValueError("Transition mapping must contain a valid endpoint")
        section1 = x1.get_sections()[index1] if index1 is not None else None
        section2 = x2.get_sections()[index2] if index2 is not None else None
        layer1 = get_layer(section1.layer) if section1 is not None else None
        layer2 = get_layer(section2.layer) if section2 is not None else None
        if layer1 is not None and layer2 is not None and layer1 != layer2:
            raise ValueError(
                "A transition mapping cannot change physical layers; use separate "
                "start/end mappings for layers that appear or disappear."
            )
        layer = layer1 if layer1 is not None else layer2
        assert layer is not None

        bounds1 = (
            (float(section1.section_min), float(section1.section_max))
            if section1 is not None
            else None
        )
        bounds2 = (
            (float(section2.section_min), float(section2.section_max))
            if section2 is not None
            else None
        )
        center1, width1 = (
            (0.0, 0.0)
            if bounds1 is None
            else (
                (bounds1[0] + bounds1[1]) / 2,
                bounds1[1] - bounds1[0],
            )
        )
        center2, width2 = (
            (center1, 0.0)
            if bounds2 is None
            else (
                (bounds2[0] + bounds2[1]) / 2,
                bounds2[1] - bounds2[0],
            )
        )
        if bounds1 is None:
            center1 = center2
        if bounds2 is None:
            center2 = center1

        core_width_func = transition_function(width_type1, x1.width, x2.width)
        if index1 == 0 and index2 == 0:
            if core_width_profile is not None:
                core_width_values = np.asarray(core_width_profile(lengths))
                if core_width_values.ndim == 0:
                    core_width_values = np.full_like(lengths, core_width_values)
                if core_width_values.shape != lengths.shape:
                    raise ValueError(
                        "core_width_profile must return one width per normalized "
                        f"path position, got shape {core_width_values.shape} for "
                        f"{lengths.shape}."
                    )
                if not np.isclose(core_width_values[0], x1.width) or not np.isclose(
                    core_width_values[-1], x2.width
                ):
                    raise ValueError(
                        "core_width_profile endpoints must equal the start and end "
                        f"core widths ({x1.width}, {x2.width})."
                    )
                left_values = -core_width_values / 2
                right_values = core_width_values / 2
            else:
                left_func = transition_function(width_type1, width1, width2)
                right_func = transition_function(width_type2, width1, width2)
                offset_func1 = transition_function(offset_type1, center1, center2)
                offset_func2 = transition_function(offset_type2, center1, center2)
                left_values = (
                    np.asarray(offset_func1(lengths))
                    - np.asarray(left_func(lengths)) / 2
                )
                right_values = (
                    np.asarray(offset_func2(lengths))
                    + np.asarray(right_func(lengths)) / 2
                )
        elif same_enclosure and index1 is not None:
            core_width_values = np.asarray(core_width_func(lengths))
            left_values, right_values = fixed_symmetric_bounds(
                section1, core_width_values, index1
            )
        else:
            left_func = transition_function(width_type1, width1, width2)
            right_func = transition_function(width_type2, width1, width2)
            offset_func1 = transition_function(offset_type1, center1, center2)
            offset_func2 = transition_function(offset_type2, center1, center2)
            left_values = (
                np.asarray(offset_func1(lengths)) - np.asarray(left_func(lengths)) / 2
            )
            right_values = (
                np.asarray(offset_func2(lengths)) + np.asarray(right_func(lengths)) / 2
            )

        points1 = p.centerpoint_offset_curve(
            points,
            offset_distance=right_values,
            start_angle=p.start_angle,
            end_angle=p.end_angle,
        )
        points2 = p.centerpoint_offset_curve(
            points,
            offset_distance=left_values,
            start_angle=p.start_angle,
            end_angle=p.end_angle,
        )
        tolerance = mapping.extrusion.simplify
        if tolerance is not None:
            points1 = _simplify(points1, tolerance=tolerance)
            points2 = _simplify(points2, tolerance=tolerance)
        if not mapping.extrusion.hidden and width1 + width2 > 0:
            c.add_polygon(np.concatenate([points1, points2[::-1, :]]), layer=layer)

        if mapping.extrusion.port_names[0] and index1 == 0:
            c.add_port(
                name=mapping.extrusion.port_names[0],
                layer=layer1 or layer,
                port_type=mapping.extrusion.port_types[0],
                width=width1,
                orientation=(p.start_angle + 180) % 360,
                center=np.average([points1[0], points2[0]], axis=0),
                cross_section=x1 if port_cross_section else None,
            )
        if mapping.extrusion.port_names[1] and index2 == 0:
            c.add_port(
                name=mapping.extrusion.port_names[1],
                layer=layer2 or layer,
                port_type=mapping.extrusion.port_types[1],
                width=width2,
                orientation=p.end_angle % 360,
                center=np.average([points1[-1], points2[-1]], axis=0),
                cross_section=x2 if port_cross_section else None,
            )

    c.info["length"] = float(np.round(p.length(), 3))
    if add_bbox:
        add_bbox_to_component(c, x1)
    return c


def _compute_offset_directions(
    points: npt.NDArray[np.floating[Any]],
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Pre-compute direction vectors for centerpoint offset curves.

    Returns (cos_theta_mid, sin_theta_mid, sin_half_dtheta_int).
    """
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    theta = np.arctan2(dy, dx)
    theta = np.concatenate([theta[:1], theta, theta[-1:]])
    theta_mid = (np.pi + theta[1:] + theta[:-1]) / 2
    dtheta_int = np.pi + theta[:-1] - theta[1:]
    sin_half = np.sin(dtheta_int / 2)
    return np.cos(theta_mid), np.sin(theta_mid), sin_half


def _offset_curve_from_directions(
    points: npt.NDArray[np.floating[Any]],
    offset_distance: float | Sequence[float] | npt.NDArray[np.floating[Any]],
    cos_theta_mid: npt.NDArray[np.floating[Any]],
    sin_theta_mid: npt.NDArray[np.floating[Any]],
    sin_half_dtheta_int: npt.NDArray[np.floating[Any]],
    start_angle: float | None = None,
    end_angle: float | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Single offset curve from pre-computed direction vectors.

    Equivalent to calling ``centerpoint_offset_curve`` but avoids
    recomputing the direction trig.
    """
    new_points = points.copy()
    offset_array = offset_distance / sin_half_dtheta_int

    new_points[:, 0] -= offset_array * cos_theta_mid
    new_points[:, 1] -= offset_array * sin_theta_mid

    if start_angle is not None:
        sa = start_angle * np.pi / 180
        sin_sa, cos_sa = np.sin(sa), np.cos(sa)
        new_points[0, :] = points[0, :] + (
            sin_sa * offset_array[0],
            -cos_sa * offset_array[0],
        )

    if end_angle is not None:
        ea = end_angle * np.pi / 180
        sin_ea, cos_ea = np.sin(ea), np.cos(ea)
        new_points[-1, :] = points[-1, :] + (
            sin_ea * offset_array[-1],
            -cos_ea * offset_array[-1],
        )

    return new_points


def _apply_offsets(
    points: npt.NDArray[np.floating[Any]],
    offset_distance1: float | Sequence[float] | npt.NDArray[np.floating[Any]],
    offset_distance2: float | Sequence[float] | npt.NDArray[np.floating[Any]],
    cos_theta_mid: npt.NDArray[np.floating[Any]],
    sin_theta_mid: npt.NDArray[np.floating[Any]],
    sin_half_dtheta_int: npt.NDArray[np.floating[Any]],
    start_angle: float | None = None,
    end_angle: float | None = None,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    return (
        _offset_curve_from_directions(
            points,
            offset_distance1,
            cos_theta_mid,
            sin_theta_mid,
            sin_half_dtheta_int,
            start_angle=start_angle,
            end_angle=end_angle,
        ),
        _offset_curve_from_directions(
            points,
            offset_distance2,
            cos_theta_mid,
            sin_theta_mid,
            sin_half_dtheta_int,
            start_angle=start_angle,
            end_angle=end_angle,
        ),
    )


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


def _cut_path_with_ray(
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
        [
            np.array(point)
            for point in path[1:-1]
            if distances[0] < ls.project(sg.Point(point)) < distances[1]
        ]
    )
    points.append(np.array(intersections[1].coords[0]))
    return np.array(points)


# Floor on the angular resolution of an auto-computed bend, in degrees per point.
# The arc-length based npoints formula underflows for small radii, so a tight bend
# would otherwise collapse to a 2-point chord (#4557). This floor keeps it curved.
# It is bounded (<= 360 / _MAX_DEG_PER_BEND_POINT points per turn) so it can't blow
# up near angle=0, unlike the inverted 360 / abs(angle) floor removed in #4337.
_MAX_DEG_PER_BEND_POINT = 5.0


def _bend_npoints_floor(angle: float) -> int:
    """Minimum points for an auto-computed bend so it stays a curve, not a chord."""
    return math.ceil(abs(angle) / _MAX_DEG_PER_BEND_POINT) + 1


def arc(
    radius: float | None = 10.0,
    angle: float = 90,
    npoints: int | None = None,
    start_angle: float = -90,
    angular_step: float | None = None,
) -> Path:
    """Returns a radial arc.

    Args:
        radius: minimum radius of curvature.
        angle: total angle of the curve.
        npoints: Number of points used per 360 degrees. Defaults to pdk.bend_points_distance.
        start_angle: initial angle of the curve for drawing, default -90 degrees.
        angular_step: If provided, determines the angular step (in degrees) between points. \
                This overrides npoints calculation.

    Example:
        ```python
        import gdsfactory as gf

        p = gf.path.arc(radius=10, angle=45)
        p.plot()
        ```
    """
    from gdsfactory.pdk import get_active_pdk

    PDK = get_active_pdk()

    if not radius:
        raise ValueError("arc() requires a radius argument")

    if npoints is not None and angular_step is not None:
        raise ValueError(
            "arc() requires either npoints or angular_step, not both. "
            "Use angular_step for angular discretization."
        )

    if angular_step is not None:
        npoints = math.ceil(abs(angle / angular_step)) + 1
    elif not npoints:
        npoints = int(abs(angle) / 360 * radius / PDK.bend_points_distance / 2)
        npoints = max(npoints, _bend_npoints_floor(angle), 2)
    else:
        npoints = max(int(npoints), 2)

    t = np.linspace(
        start_angle * np.pi / 180, (angle + start_angle) * np.pi / 180, npoints
    )
    x = radius * np.cos(t)
    y = radius * (np.sin(t) + 1)
    points = np.array((x, y)).T * np.sign(angle)

    path = Path()
    # Manually add points & adjust start and end angles
    path.points = points
    path.start_angle = start_angle + 90
    path.end_angle = start_angle + angle + 90
    return path


_SQRT_HALF_PI: float = float(np.sqrt(np.pi / 2))
_SQRT_2_OVER_PI: float = float(np.sqrt(2 / np.pi))


def _fresnel_scipy(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Evaluate Fresnel integrals via scipy.special.fresnel.

    Args:
        t: 1-D array of normalised clothoid parameter values.

    Returns:
        Array of shape (2, len(t)): [x_coords, y_coords].
    """
    from scipy.special import fresnel

    sin_fresnel, cos_fresnel = fresnel(t * _SQRT_2_OVER_PI)
    return np.array([cos_fresnel * _SQRT_HALF_PI, sin_fresnel * _SQRT_HALF_PI])


def _fresnel(
    R0: float, s: float, num_pts: int, n_iter: int = 8
) -> npt.NDArray[np.floating]:
    """Fresnel integral using scipy.

    The n_iter parameter is accepted for compatibility but ignored.
    """
    t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
    return cast("npt.NDArray[np.floating]", np.sqrt(2) * R0 * _fresnel_scipy(t))


def _fresnel_angular(
    R0: float, s: float, num_pts: int, n_iter: int = 8
) -> npt.NDArray[np.floating]:
    """Fresnel integral with uniform angular sampling via scipy.

    The n_iter parameter is accepted for compatibility but ignored.
    """
    t_max = s / float(np.sqrt(2) * R0)
    theta_max = t_max**2 / 2
    thetas = np.linspace(0, theta_max, num_pts)
    t = np.sqrt(2 * thetas)
    return cast("npt.NDArray[np.floating]", np.sqrt(2) * R0 * _fresnel_scipy(t))


def euler(
    radius: float = 10,
    angle: float = 90,
    p: float = 0.5,
    use_eff: bool = False,
    npoints: int | None = None,
    angular_step: float | None = None,
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
        angular_step: If provided, determines the angular step (in degrees) between points. \
                This overrides npoints calculation.

    Example:
        ```python
        import gdsfactory as gf

        p = gf.path.euler(radius=10, angle=45, p=1, use_eff=True, npoints=720)
        p.plot()
        ```
    """
    from gdsfactory.pdk import get_active_pdk

    if angular_step is not None and npoints is not None:
        raise ValueError(
            "euler() requires either npoints or angular_step, not both. "
            "Use angular_step for angular discretization."
        )

    if not radius:
        raise ValueError("euler() requires a radius argument")

    if (p < 0) or (p > 1):
        raise ValueError(f"euler requires argument `p` be between 0 and 1. Got {p}")
    if p == 0:
        path = arc(radius, angle, npoints=npoints, angular_step=angular_step)
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
    sp = float(R0 * np.sqrt(p * alpha))
    # Rp = inf at alpha=0, but all usages are guarded by is_small_angle
    with np.errstate(divide="ignore"):
        Rp = R0 / np.sqrt(p * alpha)

    pdk = get_active_pdk()
    if angular_step is not None:
        npoints = math.ceil(abs(angle / angular_step)) + 1
        # For angular discretization, distribute points based on angle proportion
        euler_angle = p * angle / 2  # Angle covered by each Euler section
        arc_angle = (
            (1 - p) * angle / 2
        )  # Angle covered by arc section (half, then mirrored)
        num_pts_euler = max(2, math.ceil(euler_angle / angular_step))
        num_pts_arc = max(2, math.ceil(arc_angle / angular_step) + 1)
        npoints = (
            2 * num_pts_euler + num_pts_arc - 2
        )  # Total points (avoiding duplicates)
    else:
        if not npoints:
            npoints = abs(int(angle / 360 * radius / pdk.bend_points_distance / 2))
            npoints = max(npoints, _bend_npoints_floor(angle), 2)
        else:
            npoints = max(int(npoints), 2)
        # Use simplified form: sp/(s0/2) = 2p/(p+1), avoids 0/0 at alpha=0
        num_pts_euler = int(np.round(2 * p / (p + 1) * npoints))
        num_pts_arc = npoints - num_pts_euler

    # Ensure a minimum of 2 points for each euler/arc section
    if npoints <= 2:
        num_pts_euler = 0
        num_pts_arc = 2

    # Small angle threshold in degrees (matches arc() convention)
    is_small_angle = abs(angle) <= 1e-6

    if num_pts_euler > 0:
        if angular_step is not None:
            xbend1, ybend1 = _fresnel_angular(R0, sp, num_pts_euler)
        else:
            xbend1, ybend1 = _fresnel(R0, sp, num_pts_euler)
        xp, yp = xbend1[-1], ybend1[-1]
        # Sinc-based formulas avoid Rp*sin (inf*0 at alpha=0)
        # Rp * sin(p*a/2) = sp/2 * sinc(p*a/(2*pi))
        # Rp * (1-cos(p*a/2)) = (sp^3/8) * sinc^2(p*a/(4*pi))  [using 1-cos=2sin^2]
        # Using sinc(2x) = sinc(x)*cos(pi*x) to compute both from one sinc call
        sinc_quarter = np.sinc(p * alpha / (4 * np.pi))
        dx = xp - sp / 2 * sinc_quarter * np.cos(p * alpha / 4)
        dy = yp - (sp**3 / 8) * sinc_quarter**2
    else:
        xbend1 = ybend1 = np.asarray([], dtype=float)
        dx = 0
        dy = 0

    if not is_small_angle:
        if angular_step is not None:
            # Original angular_step behavior (different from npoints mode)
            arc_angle_section = alpha * (1 - p) / 2
            theta = np.linspace(0, arc_angle_section, num_pts_arc)
            arc_angles = theta + p * alpha / 2
        else:
            # Direct linspace replaces: s = linspace(sp, s0/2), arc_angles = (s-sp)*sqrt(p*a)/R0 + p*a/2
            arc_angles = np.linspace(p * alpha / 2, alpha / 2, num_pts_arc)
        xbend2 = Rp * np.sin(arc_angles) + dx
        ybend2 = Rp * (1 - np.cos(arc_angles)) + dy
    else:
        # Limit is 0 by L'Hopital: Rp*sin(t) ~ sin(a)/sqrt(a) -> 0
        xbend2 = np.zeros(num_pts_arc) + dx
        # Limit is 0: Rp*(1-cos(t)) ~ (1/sqrt(a))*(a^2/2) = a^(3/2)/2 -> 0
        ybend2 = np.zeros(num_pts_arc) + dy

    x = np.concatenate([xbend1, xbend2[1:]])
    y = np.concatenate([ybend1, ybend2[1:]])
    points1 = np.array([x, y]).T
    points2 = np.flipud(np.array([x, -y]).T)

    points2 = rotate_points(points2, angle - 180)
    points2 += -points2[0, :] + points1[-1, :]

    # Use [:-1] to remove duplicate junction point, but [:None] if only 1 point
    points = np.concatenate([points1[: -1 if len(points1) > 1 else None], points2])

    # Find y-axis intersection point to compute Reff
    start_angle = 180 * (angle < 0)
    end_angle = start_angle + angle

    if is_small_angle:
        # Degenerate case: curve collapses to a point, radius is infinite
        Reff = np.inf
        Rmin = np.inf
        scale = 0.0
    else:
        dy = np.tan(np.radians(end_angle - 90)) * points[-1][0]
        Reff = points[-1][1] - dy
        Rmin = Rp
        # Fix degenerate condition at angle == 180
        if np.abs(180 - angle) < 1e-3:
            Reff = points[-1][1] / 2
        scale = radius / Reff if use_eff else radius / Rmin

    points *= scale

    path = Path()

    # Manually add points & adjust start and end angles
    path.points = points
    path.start_angle = start_angle
    path.end_angle = end_angle
    path.info["Reff"] = Reff * scale if not is_small_angle else np.inf
    path.info["Rmin"] = Rmin * scale if not is_small_angle else np.inf
    if mirror:
        path.mirror((1, 0))
    return path


def _find_root_in_range(
    equation: Callable[[float], float],
    variable_range: tuple[float, float],
) -> tuple[float, float]:
    """Find a root of `equation` within the given range using Brent's method.

    Falls back to a coarse scan if the initial bracket has no sign change.

    Args:
        equation: a scalar function f(x) whose zero is sought.
        variable_range: (lower, upper) bounds for the search.

    Returns:
        (root, residual): the found root and the constraint value at it.

    Raises:
        RuntimeError: if no sign change is found within the range.
    """
    var_lo, var_hi = variable_range

    try:
        root = float(optimize.brentq(equation, var_lo, var_hi, xtol=1e-9, maxiter=500))
    except ValueError:
        x_vals = np.linspace(var_lo, var_hi, 1000)
        residuals = [equation(x) for x in x_vals]
        sign_changes = np.where(np.diff(np.sign(residuals)))[0]
        if len(sign_changes) == 0:
            raise RuntimeError(
                f"No root found in [{var_lo}, {var_hi}]. "
                "Verify that the constraint changes sign within the bracket."
            )
        root = float(
            optimize.brentq(
                equation, x_vals[sign_changes[0]], x_vals[sign_changes[0] + 1]
            )
        )

    return root, float(equation(root))


def _topic_theta_of_s(s: float, Rc: float, theta_p: float) -> float:
    """Accumulated angle along the TOP spiral section."""
    return float((4 * Rc * theta_p * s**3 - s**4) / (16 * Rc**4 * theta_p**3))


def _topic_compute_x0_y0(Rc: float, theta_p: float) -> tuple[float, float]:
    """Numerically integrate the Fresnel-like integrals for a given Rc.

    Args:
        Rc: minimum radius of curvature.
        theta_p: transition angle in radians (= p * theta_t).

    Returns:
        (x0, y0): center of the circular arc segment.
    """
    from scipy import integrate

    s_max = 2 * Rc * theta_p

    x0_int, _ = integrate.quad(
        lambda s: np.cos(_topic_theta_of_s(s, Rc, theta_p)), 0, s_max
    )
    y0_int, _ = integrate.quad(
        lambda s: np.sin(_topic_theta_of_s(s, Rc, theta_p)), 0, s_max
    )

    x0 = x0_int - Rc * np.sin(theta_p)
    y0 = y0_int + Rc * np.cos(theta_p)
    return x0, y0


def _topic_compute_top_coordinates(
    Rc: float, theta_p: float, n_points: int = 300
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute (x, y) along the TOP spiral section via Eq. 6 of the paper.

    Integrates:
        x(l) = integral_0^l cos(theta(s)) ds
        y(l) = integral_0^l sin(theta(s)) ds

    where theta(s) = (4*Rc*theta_p*s^3 - s^4) / (16 * Rc^4 * theta_p^3)
    and l ranges from 0 to 2*Rc*theta_p.

    Args:
        Rc: minimum radius of curvature (from solver).
        theta_p: transition angle in radians (= p * theta_t).
        n_points: number of sample points along the curve.

    Returns:
        x_top: x coordinates.
        y_top: y coordinates.
    """
    from scipy import integrate

    l_max = 2 * Rc * theta_p

    l_vals = np.linspace(0, l_max, n_points)
    theta_vals = np.array([_topic_theta_of_s(s, Rc, theta_p) for s in l_vals])

    x_top = integrate.cumulative_trapezoid(np.cos(theta_vals), l_vals, initial=0)
    y_top = integrate.cumulative_trapezoid(np.sin(theta_vals), l_vals, initial=0)

    return x_top, y_top


def topic(
    radius: float = 10.0, angle: float = 90.0, p: float = 0.1, npoints: int = 100
) -> Path:
    """Returns a Third Order Polynomial Interconnected Circular (TOPIC) bend, as described in this publication https://arxiv.org/html/2411.15025v1.

    The bend consists of three parts:
    a. Initial transition from straight to bend, known as TOP segment.
    b. Circular part whose center and radius are calculated analytically.
    c. Mirroring of TOP segment with respect to the bisection of the angle.

    The implementation consists of 5 parts.
    1. Define transition angle as p*angle.
    2. Calculate the center, (x0, y0), radius (Rc) and angle (angle*(1-2*p)) of the circular path.
    3. Generate TOP segment.
    4. Generate circular segment, starting from the end of TOP with center (x0, y0), radius Rc, and angle = angle(1-2*p)
    5. Generate TOP' segment by mirroring TOP with respect to the bisector of the angle.

    Args:
        radius: radius at the start and end of bend.
        angle: total angle of the curve in degrees.
        p: used to calculate the angle of the bend at the end of TOP / start of circular arc, as p*angle. It should be within [0, 0.5).
        npoints: Number of points used per 360 degrees.

    Example:
        ```python
        import gdsfactory as gf

        p = gf.path.topic(radius=10, angle=110, p=0.1, npoints=720)
        p.plot()
        ```
    """
    if p < 0.0 or p >= 0.5:
        raise ValueError(
            "The angle of bend during the transition from the TOP segment to the circular is p*angle . "
            "topic() requires the transition angle to be between 0 (circular bend) and 0.5*angle . "
        )
    if abs(angle) <= 1e-6:
        raise ValueError("The bend's total angle should be larger than 1e-6.")
    if p < 1e-4:
        topic_path = arc(radius=radius, angle=angle, npoints=npoints)
        topic_path.end_angle = angle
        topic_path.info["Rmin"] = radius
        return topic_path

    # 1. Define transition angle as p*angle.
    theta_t = np.radians(angle)
    theta_p = p * theta_t

    def constraint(Rc: float) -> float:
        """Residual of the first equation of the paper: should equal zero."""
        if Rc <= 0:
            return 1e9
        x0, y0 = _topic_compute_x0_y0(Rc, theta_p)
        return float(x0 * np.cos(theta_t / 2) + (y0 - radius) * np.sin(theta_t / 2))

    Rc, _ = _find_root_in_range(constraint, (1e-3 * radius, radius))

    x0, y0 = _topic_compute_x0_y0(Rc, theta_p)

    # Split number of points between TOP, circular and TOP' sections.
    # Circular gets the percentage of total points corresponding to its arc length / the arc length if the whole bend was circular
    n_points_circ = int(npoints * (Rc * (theta_t - 2 * theta_p)) / (radius * theta_t))
    n_points_top = max(2, (npoints - n_points_circ) // 2)

    # Add another point to circular arc if there is one left
    if 2 * n_points_top + n_points_circ == npoints - 1:
        n_points_circ += 1

    # 3. Generate TOP segment.
    x_top, y_top = _topic_compute_top_coordinates(Rc, theta_p, n_points=n_points_top)

    # 4. Generate circular segment, starting from the end of TOP with center (x0, y0), radius Rc, and angle = angle(1-2*p)
    # In order to avoid overlap between TOP's last point and circ first point, ignore the first and last points of the arc.
    theta_list = np.linspace(
        start=theta_p,
        stop=theta_t - theta_p,
        num=n_points_circ + 2,
        endpoint=True,  # n_points_circ+2 to avoid the first and last one
    )

    x_arc = np.array([x0 + Rc * np.sin(theta) for theta in theta_list[1:-1]])
    y_arc = np.array([y0 - Rc * np.cos(theta) for theta in theta_list[1:-1]])

    # 5. Generate TOP' segment by mirroring TOP with respect to the bisector of the angle.
    # The goal is to do a rotation of each point of TOP, centered to (0,radius).
    dist = np.sqrt(x_top**2 + (radius - y_top) ** 2)
    thetas = np.asin(np.clip(x_top / dist, -1, 1))
    # The first point of TOP corresponds to the last point of TOP', this is why we reverse the vectors
    x_top_prime = dist * np.sin(theta_t - thetas)
    x_top_prime = x_top_prime[::-1]
    y_top_prime = radius - dist * np.cos(theta_t - thetas)
    y_top_prime = y_top_prime[::-1]

    x_all = np.concatenate([x_top, x_arc, x_top_prime])
    y_all = np.concatenate([y_top, y_arc, y_top_prime])

    points = np.column_stack((x_all, y_all))

    topic_path = Path()
    topic_path.points = points
    topic_path.end_angle = angle
    topic_path.info["Rmin"] = Rc

    return topic_path


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
        separation: Half the radial separation between loops in um. The
            current formula is retained for compatibility with existing
            layouts, so adjacent turns are separated by ``2 * separation``.
        number_of_loops: number of loops.
        npoints: number of Points.

    Example:
        ```python
        import gdsfactory as gf

        p = gf.path.spiral_archimedean(min_bend_radius=5, separation=2, number_of_loops=3, npoints=200)
        p.plot()
        ```
    """
    theta = np.linspace(0, number_of_loops * 2 * np.pi, int(npoints))
    points = (separation / np.pi * theta + min_bend_radius)[:, None] * np.column_stack(
        (np.sin(theta), np.cos(theta))
    )
    return Path(points)


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

    tol = 1e-6
    if np.any(np.linalg.norm(normals, axis=1) < tol):
        warnings.warn(
            "Zero-length segments (duplicate consecutive points)",
            RuntimeWarning,
            stacklevel=3,
        )

    normals = (normals.T / np.linalg.norm(normals, axis=1)).T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    dtheta = dtheta - 360 * np.floor((dtheta + 180) / 360)
    return points, normals, ds, theta, dtheta


def smooth(
    points: npt.NDArray[np.floating[Any]] | Path,
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

    Example:
        ```python
        import gdsfactory as gf

        p = gf.path.smooth(([0, 0], [0, 10], [10, 10]))
        p.plot()
        ```
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
    path.append(new_points_np)
    path.rotate(float(theta[0]))
    path.move(cast("tuple[float, float]", points[0, :]))
    path.start_angle = theta[0]
    path.end_angle = theta[-1]
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
