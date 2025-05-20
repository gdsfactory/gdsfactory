import math
from warnings import warn

import numpy as np

import gdsfactory as gf
from gdsfactory.config import CONF
from gdsfactory.port import Port
from gdsfactory.routing.utils import RouteWarning


def make_error_traces(
    component: gf.Component, ports1: list[Port], ports2: list[Port], message: str
) -> None:
    """Creates a set of error traces showing the intended connectivity between ports1 and ports2.

    The specified message will be included in the RouteWarning that is raised.

    Args:
        component: the Component to add the error traces to.
        ports1: the list of starting ports.
        ports2: the list of ending ports.
        message: a message to include in the RouteWarning that is raised.

    Returns:
        A list of Routes (the error traces).
    """
    import gdsfactory as gf

    warn(message, RouteWarning, stacklevel=3)
    for port1, port2 in zip(ports1, ports2):
        path = gf.path.Path(np.array([port1.center, port2.center]))
        error_component = gf.path.extrude(path, layer=CONF.layer_error_path, width=1)
        _ = component << error_component


def is_invalid_bundle_topology(ports1: list[Port], ports2: list[Port]) -> bool:
    """Returns True if the bundle is topologically unroutable without introducing crossings.

    Args:
        ports1: the starting ports of the bundle.
        ports2: the ending ports of the bundle.

    Returns:
        True if the bundle is unroutable. False otherwise.
    """
    angle_tolerance: float = 1e-10

    # Early exit: only one route or missing orientation, always valid
    if len(ports1) < 2:
        return False

    if any(getattr(p, "orientation", None) is None for p in ports1 + ports2):
        return False

    ports_facing: list[float] = []
    center_pairs: list[
        tuple[tuple[float, float], tuple[float, float]]
    ] = []  # for intersection checking

    # Precompute all necessary quantities
    for p1, p2 in zip(ports1, ports2):
        c1: tuple[float, float] = p1.center
        c2: tuple[float, float] = p2.center

        dx: float = c2[0] - c1[0]
        dy: float = c2[1] - c1[1]
        o1: float = math.radians(p1.orientation)
        o2: float = math.radians(p2.orientation)
        dx_p1: float = math.cos(o1)
        dy_p1: float = math.sin(o1)
        dx_p2: float = math.cos(o2)
        dy_p2: float = math.sin(o2)

        dot1: float = dx * dx_p1 + dy * dy_p1
        dot2: float = -(dx * dx_p2 + dy * dy_p2)
        ports_facing.append(dot1 * dot2)

        center_pairs.append((c1, c2))

    has_intersections: bool = _any_intersection(center_pairs)

    if not has_intersections and all(s < -angle_tolerance for s in ports_facing):
        return True
    if has_intersections and all(s > angle_tolerance for s in ports_facing):
        return True

    return False


def _segment_intersects_fast(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """Fast check if 2 segments intersect (excluding colinear cases)."""

    def ccw(
        p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]
    ) -> bool:
        # Counter-clockwise test
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def _any_intersection(
    center_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> bool:
    """Returns True if any of the lines intersect (O(n^2), fast for moderate n)."""
    n: int = len(center_pairs)
    for i in range(n):
        a1, a2 = center_pairs[i]
        for j in range(i + 1, n):
            b1, b2 = center_pairs[j]
            if _segment_intersects_fast(a1, a2, b1, b2):
                return True
    return False
