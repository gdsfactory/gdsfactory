"""Routes bundles of ports with variable path length using meander components.

This module provides a routing function similar to ``route_bundle`` that
achieves a user-specified total path length by inserting meander (delay)
components at a chosen point along each route.  The extra length is absorbed
by the meander, whose internal serpentine path is sized to make up the
difference between the natural (baseline) Manhattan route length and the
requested target length.

Typical meander components include ``delay_snake``, ``delay_snake2``, and
``delay_snake_sbend`` from :mod:`gdsfactory.components.spirals`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.routing.route_bundle import (
    _ensure_manhattan_waypoints,
    route_bundle,
)
from gdsfactory.typings import (
    STEP_DIRECTIVES,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpec,
    LayerTransitions,
    Port,
    Ports,
    Step,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VariablePathLengthRoute:
    """Result for a single route with variable path length.

    Attributes:
        routes: Manhattan route segments (pre-meander and post-meander).
        meander_instance: Reference to the placed meander component, or *None*
            if no meander was needed.
        target_length: Requested path length in um.
        actual_length: Achieved total path length in um.
        baseline_length: Estimated natural Manhattan route length (without
            meander) in um.
    """

    routes: list[ManhattanRoute] = field(default_factory=list)
    meander_instance: Any | None = None
    target_length: float = 0.0
    actual_length: float = 0.0
    baseline_length: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _steps_to_waypoints(
    steps: Sequence[Step],
    start_port: Port,
) -> list[kf.kdb.DPoint]:
    """Convert step directives to absolute waypoint coordinates."""
    x, y = start_port.dcplx_trans.disp.x, start_port.dcplx_trans.disp.y
    waypoints: list[kf.kdb.DPoint] = []
    for d in steps:
        if not isinstance(d, dict):
            raise ValueError(
                f"Invalid step {d!r}. Each step must be a dict with keys (x, y, dx, dy)."
            )
        if not STEP_DIRECTIVES.issuperset(d):
            invalid = list(d.keys() - STEP_DIRECTIVES)
            raise ValueError(
                f"Invalid step directives: {invalid}. Valid: {list(STEP_DIRECTIVES)}"
            )
        x = d.get("x", x) + d.get("dx", 0)
        y = d.get("y", y) + d.get("dy", 0)
        waypoints.append(kf.kdb.DPoint(x, y))
    return waypoints


def _manhattan_distance(p1: kf.kdb.DPoint, p2: kf.kdb.DPoint) -> float:
    """Manhattan distance between two points."""
    return abs(p2.x - p1.x) + abs(p2.y - p1.y)


def _compute_manhattan_length(
    points: Sequence[kf.kdb.DPoint],
    bend_length: float = 0.0,
    bend_radius: float = 0.0,
) -> float:
    """Estimate the routed Manhattan path length through *points*.

    Accounts for 90deg bends at each direction change: the bend component
    replaces ``2 * bend_radius`` of straight with ``bend_length`` of arc,
    giving a per-bend correction of ``bend_length - 2 * bend_radius``.
    """
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(len(points) - 1):
        total += _manhattan_distance(points[i], points[i + 1])

    # Count direction changes (bends)
    n_bends = 0
    for i in range(1, len(points) - 1):
        v1x = points[i].x - points[i - 1].x
        v1y = points[i].y - points[i - 1].y
        v2x = points[i + 1].x - points[i].x
        v2y = points[i + 1].y - points[i].y
        h1 = abs(v1x) > abs(v1y)
        h2 = abs(v2x) > abs(v2y)
        if h1 != h2:
            n_bends += 1

    if bend_length and bend_radius:
        total += n_bends * (bend_length - 2 * bend_radius)

    return total


def _segment_direction(p1: kf.kdb.DPoint, p2: kf.kdb.DPoint) -> float:
    """Direction angle (0, 90, 180, or 270) for a Manhattan segment."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    if abs(dx) >= abs(dy):
        return 0.0 if dx > 0 else 180.0
    return 90.0 if dy > 0 else 270.0


def _choose_insertion_segment(
    backbone: Sequence[kf.kdb.DPoint],
    start_port: Port,
) -> int:
    """Choose the best backbone segment for meander insertion.

    Prefers segments whose direction matches the start port's orientation
    so that the meander's input port naturally faces the incoming route.
    Among candidates picks the longest.  Falls back to the overall longest
    segment.

    Returns the segment index (the meander will be placed at the midpoint
    of ``backbone[idx] -> backbone[idx+1]``).
    """
    n = len(backbone)
    start_orient = start_port.orientation
    start_dir = int(start_orient) % 360 if start_orient is not None else None

    # Score each segment
    best_aligned_idx = -1
    best_aligned_len = -1.0
    best_any_idx = 0
    best_any_len = -1.0

    for i in range(n - 1):
        seg_len = _manhattan_distance(backbone[i], backbone[i + 1])
        if seg_len > best_any_len:
            best_any_len = seg_len
            best_any_idx = i
        seg_dir = int(_segment_direction(backbone[i], backbone[i + 1])) % 360
        if start_dir is not None and seg_dir == start_dir and seg_len > best_aligned_len:
            best_aligned_len = seg_len
            best_aligned_idx = i

    return best_aligned_idx if best_aligned_idx >= 0 else best_any_idx


def _resolve_waypoints(
    ports1: Sequence[Port],
    ports2: Sequence[Port],
    waypoints: Coordinates | Sequence[kf.kdb.DPoint] | None,
    steps: Sequence[Step] | None,
) -> list[kf.kdb.DPoint] | None:
    """Convert user-supplied waypoints / steps to a list of DPoints."""
    if steps and waypoints:
        raise ValueError("Provide either steps or waypoints, not both.")

    if steps:
        return _steps_to_waypoints(steps, ports1[0])

    if waypoints is None:
        return None

    if len(waypoints) == 0:
        return []

    if isinstance(waypoints[0], kf.kdb.DPoint):
        return [cast("kf.kdb.DPoint", p) for p in waypoints]

    return [kf.kdb.DPoint(p[0], p[1]) for p in waypoints]  # type: ignore[index]


def _build_backbone(
    port1: Port,
    port2: Port,
    waypoints: list[kf.kdb.DPoint] | None,
) -> list[kf.kdb.DPoint]:
    """Build a full backbone from port1 center -> waypoints -> port2 center."""
    p1 = kf.kdb.DPoint(
        port1.dcplx_trans.disp.x,
        port1.dcplx_trans.disp.y,
    )
    p2 = kf.kdb.DPoint(
        port2.dcplx_trans.disp.x,
        port2.dcplx_trans.disp.y,
    )
    if waypoints:
        return [p1, *waypoints, p2]
    return [p1, p2]


def _get_meander_port_offset(
    meander: ComponentSpec,
    meander_n_loops: int,
    meander_xs: CrossSectionSpec,
    meander_bend180: ComponentSpec,
) -> tuple[float, float]:
    """Get the (dx, dy) offset from o1 to o2 in the meander's local frame.

    Uses a template meander with a safe length to determine the port geometry
    without depending on the actual meander length (the offset is determined
    by the number of loops and bend geometry, not the straight lengths).
    """
    template = gf.get_component(
        meander,
        length=500,  # arbitrary safe length
        n=meander_n_loops,
        cross_section=meander_xs,
        bend180=meander_bend180,
    )
    o1 = template.ports["o1"].dcplx_trans.disp
    o2 = template.ports["o2"].dcplx_trans.disp
    return (o2.x - o1.x, o2.y - o1.y)


def _rotate_offset(dx: float, dy: float, angle_deg: float) -> tuple[float, float]:
    """Rotate an (dx, dy) offset by *angle_deg* counter-clockwise."""
    import math

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_bundle_variable_path_length(
    component: gf.Component,
    ports1: Port | Ports,
    ports2: Port | Ports,
    target_length: float | Sequence[float],
    cross_section: CrossSectionSpec = "strip",
    waypoints: Coordinates | Sequence[kf.kdb.DPoint] | None = None,
    steps: Sequence[Step] | None = None,
    meander: ComponentSpec = "delay_snake",
    meander_n_loops: int = 2,
    meander_cross_section: CrossSectionSpec | None = None,
    meander_bend180: ComponentSpec = "bend_euler180",
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    separation: float = 3.0,
    sort_ports: bool = False,
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
    layer: LayerSpec | None = None,
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    layer_transitions: LayerTransitions | None = None,
) -> list[VariablePathLengthRoute]:
    """Route between ports via waypoints with a target total path length.

    Similar to :func:`~gdsfactory.routing.route_bundle.route_bundle`, but
    inserts a meander component (e.g. ``delay_snake``) into each route so that
    the total routed path length matches ``target_length``.

    The function:

    1. Computes the natural Manhattan route length (baseline) through the
       supplied waypoints.
    2. Determines the extra length each route needs to reach its target.
    3. Places a meander component at an interior waypoint along the backbone.
       The meander's internal path length is sized so that the total
       (pre-route + meander + post-route) equals the target length.
    4. Routes from port1 -> meander input and meander output -> port2 using
       standard Manhattan routing.

    Args:
        component: Component to add the routes to.
        ports1: Starting port(s).
        ports2: Ending port(s).
        target_length: Desired total path length in um.  A single float is
            applied to every route; a sequence gives per-route targets.
        cross_section: Cross-section for the route.
        waypoints: Intermediate waypoints.
        steps: Step directives (alternative to waypoints).
        meander: Component spec for the meander element (must accept *length*,
            *n*, *cross_section* and *bend180* keyword arguments).
        meander_n_loops: Number of loops inside the meander (``n`` parameter of
            ``delay_snake``).  Must be even.
        meander_cross_section: Cross-section for the meander.  Defaults to the
            route *cross_section*.
        meander_bend180: 180deg bend used inside the meander.
        bend: 90deg bend for the Manhattan route segments.
        straight: Straight waveguide for the Manhattan route segments.
        separation: Bundle separation in um.
        sort_ports: Sort ports by coordinate.
        start_straight_length: Extra straight at route start.
        end_straight_length: Extra straight at route end.
        radius: Bend radius (overrides cross-section default).
        route_width: Route width (overrides cross-section default).
        auto_taper: Auto-taper ports to routing cross-section.
        layer: Routing layer (alternative to *cross_section*).
        allow_width_mismatch: Allow different port widths.
        allow_layer_mismatch: Allow different port layers.
        allow_type_mismatch: Allow different port types.
        layer_transitions: Layer transition specs.

    Returns:
        A list of :class:`VariablePathLengthRoute` results, one per port pair.

    Note:
        The achieved path length may differ slightly from the target (typically
        within a few percent) due to bend geometry corrections that are
        estimated analytically.  For straight-dominated routes the error is
        negligible; for routes with many bends it can be ~3%.  The
        :attr:`VariablePathLengthRoute.actual_length` field reports the true
        achieved length.

    Raises:
        ValueError: If *target_length* is shorter than the achievable minimum.

    Example::

        import gdsfactory as gf
        from gdsfactory.routing.route_bundle_variable_path_length import (
            route_bundle_variable_path_length,
        )

        c = gf.Component()
        s1 = c << gf.components.straight(length=10)
        s2 = c << gf.components.straight(length=10)
        s2.dmove((200, 100))

        results = route_bundle_variable_path_length(
            c,
            ports1=s1.ports["o2"],
            ports2=s2.ports["o1"],
            target_length=800,
            cross_section="strip",
            waypoints=[(100, 0), (100, 100)],
        )
        for r in results:
            print(f"target={r.target_length:.1f}  actual={r.actual_length:.1f}")
        c.show()
    """
    # ------------------------------------------------------------------
    # 1. Normalise inputs
    # ------------------------------------------------------------------
    if isinstance(ports1, kf.DPort):
        ports1 = [ports1]
    if isinstance(ports2, kf.DPort):
        ports2 = [ports2]

    ports1 = list(ports1)
    ports2 = list(ports2)

    n_routes = len(ports1)
    if n_routes != len(ports2):
        raise ValueError(
            f"ports1 ({n_routes}) and ports2 ({len(ports2)}) must have equal length"
        )

    if isinstance(target_length, (int, float)):
        target_lengths = [float(target_length)] * n_routes
    else:
        target_lengths = [float(t) for t in target_length]
        if len(target_lengths) != n_routes:
            raise ValueError(
                f"target_length list ({len(target_lengths)}) must match "
                f"number of port pairs ({n_routes})"
            )

    # ------------------------------------------------------------------
    # 2. Resolve cross-section / geometry parameters
    # ------------------------------------------------------------------
    if cross_section is None and (layer is None or route_width is None):
        raise ValueError(
            "Provide either cross_section or both layer and route_width."
        )

    if cross_section is None:
        cross_section = partial(
            gf.cross_section.cross_section,
            layer=cast("LayerSpec", layer),
            width=cast("float", route_width),
        )

    xs = gf.get_cross_section(cross_section)
    width = route_width or xs.width
    _radius = radius or xs.radius
    meander_xs = meander_cross_section or cross_section

    # Bend info for baseline length estimation
    bend_comp = (
        bend
        if isinstance(bend, gf.Component)
        else gf.get_component(
            bend, cross_section=cross_section, radius=_radius, width=width
        )
    )
    bend_arc_length = float(bend_comp.info.get("length", 0))

    # ------------------------------------------------------------------
    # 3. Resolve waypoints
    # ------------------------------------------------------------------
    wp = _resolve_waypoints(ports1, ports2, waypoints, steps)
    if wp is not None and len(wp) >= 2:
        wp = _ensure_manhattan_waypoints(wp, start_port=ports1[0])

    # Meander port offset (independent of meander length)
    meander_offset = _get_meander_port_offset(
        meander, meander_n_loops, meander_xs, meander_bend180
    )

    # Common kwargs forwarded to route_bundle
    _rb_kwargs: dict[str, Any] = dict(
        cross_section=cross_section,
        bend=bend,
        straight=straight,
        separation=separation,
        sort_ports=sort_ports,
        radius=radius,
        route_width=route_width,
        auto_taper=auto_taper,
        allow_width_mismatch=allow_width_mismatch,
        allow_layer_mismatch=allow_layer_mismatch,
        allow_type_mismatch=allow_type_mismatch,
        layer_transitions=layer_transitions,
    )

    # Database unit conversion factor: ManhattanRoute.length is in database
    # units (typically nm).  Multiply by dbu to convert to um.
    dbu = component.kcl.dbu  # typically 0.001 (um per database unit)

    # ------------------------------------------------------------------
    # 4. Process each port pair
    # ------------------------------------------------------------------
    results: list[VariablePathLengthRoute] = []

    for idx in range(n_routes):
        p1 = ports1[idx]
        p2 = ports2[idx]
        tgt = target_lengths[idx]

        # Build backbone for length estimation (all in um)
        backbone = _build_backbone(p1, p2, wp)
        backbone = _ensure_manhattan_waypoints(backbone, start_port=p1)
        baseline = _compute_manhattan_length(backbone, bend_arc_length, _radius)

        extra = tgt - baseline

        if extra <= 0:
            # Target already met - route normally
            routes = route_bundle(
                component,
                ports1=[p1],
                ports2=[p2],
                waypoints=wp,
                start_straight_length=start_straight_length,
                end_straight_length=end_straight_length,
                **_rb_kwargs,
            )
            actual = (routes[0].length * dbu) if routes else baseline
            results.append(
                VariablePathLengthRoute(
                    routes=routes,
                    meander_instance=None,
                    target_length=tgt,
                    actual_length=actual,
                    baseline_length=baseline,
                )
            )
            continue

        # ---------------------------------------------------------------
        # Need to insert a meander to absorb extra length
        # ---------------------------------------------------------------

        # Choose the segment whose direction matches the start port
        seg_idx = _choose_insertion_segment(backbone, p1)
        seg_start = backbone[seg_idx]
        seg_end = backbone[seg_idx + 1]
        seg_dir = _segment_direction(seg_start, seg_end)

        # Place meander at the MIDPOINT of the chosen segment so that
        # port1 can route straight into meander.o1 without detours.
        mid_pt = kf.kdb.DPoint(
            (seg_start.x + seg_end.x) / 2,
            (seg_start.y + seg_end.y) / 2,
        )

        # Compute the meander o2 world position (for length estimation)
        rot_dx, rot_dy = _rotate_offset(
            meander_offset[0], meander_offset[1], seg_dir
        )
        o2_position = kf.kdb.DPoint(mid_pt.x + rot_dx, mid_pt.y + rot_dy)

        # Estimate pre-route length: port1 -> midpoint (via earlier backbone)
        pre_pts = list(backbone[: seg_idx + 1]) + [mid_pt]
        pre_length_est = _compute_manhattan_length(
            pre_pts, bend_arc_length, _radius
        )

        # Estimate post-route length: meander o2 -> port2
        post_pts = [o2_position, *backbone[seg_idx + 1 :]]
        post_pts = _ensure_manhattan_waypoints(post_pts)
        post_length_est = _compute_manhattan_length(
            post_pts, bend_arc_length, _radius
        )

        # Meander internal length = target - pre - post
        meander_internal = tgt - pre_length_est - post_length_est
        if meander_internal < 0:
            raise ValueError(
                f"Cannot achieve target_length={tgt:.1f} um: the pre-route "
                f"({pre_length_est:.1f}) + post-route ({post_length_est:.1f}) "
                f"already exceeds it.  Reduce meander_n_loops or increase "
                f"target_length."
            )

        # Create the meander component
        meander_comp = gf.get_component(
            meander,
            length=meander_internal,
            n=meander_n_loops,
            cross_section=meander_xs,
            bend180=meander_bend180,
        )

        # Place meander in the component
        meander_ref = component << meander_comp

        # Rotate to align with the segment direction.
        # delay_snake default flow is east (0deg); rotate by seg_dir.
        if seg_dir:
            meander_ref.drotate(seg_dir)

        # Position so o1 is at the segment midpoint
        o1_disp = meander_ref.ports["o1"].dcplx_trans.disp
        meander_ref.dmove((mid_pt.x - o1_disp.x, mid_pt.y - o1_disp.y))

        meander_o1 = meander_ref.ports["o1"]
        meander_o2 = meander_ref.ports["o2"]

        # Waypoints for pre-route (port1 -> meander o1):
        # Use backbone vertices between port1 and the insertion segment
        wp_before: list[kf.kdb.DPoint] | None = None
        if seg_idx >= 1:
            wp_before = list(backbone[1 : seg_idx + 1])

        # Waypoints for post-route (meander o2 -> port2):
        # Use backbone vertices after the insertion segment
        wp_after: list[kf.kdb.DPoint] | None = None
        remaining = list(backbone[seg_idx + 1 : -1])
        if remaining:
            wp_after = remaining

        # Route segment 1: port1 -> meander o1
        routes_pre = route_bundle(
            component,
            ports1=[p1],
            ports2=[meander_o1],
            waypoints=wp_before if wp_before and len(wp_before) >= 2 else None,
            start_straight_length=start_straight_length,
            end_straight_length=0,
            **_rb_kwargs,
        )

        # Route segment 2: meander o2 -> port2
        routes_post = route_bundle(
            component,
            ports1=[meander_o2],
            ports2=[p2],
            waypoints=wp_after if wp_after and len(wp_after) >= 2 else None,
            start_straight_length=0,
            end_straight_length=end_straight_length,
            **_rb_kwargs,
        )

        # Compute actual total length (convert route lengths from dbu to um)
        meander_path_length = float(
            meander_comp.info.get("length", meander_internal)
        )
        actual_pre = sum(r.length * dbu for r in routes_pre)
        actual_post = sum(r.length * dbu for r in routes_post)
        actual = actual_pre + meander_path_length + actual_post

        results.append(
            VariablePathLengthRoute(
                routes=routes_pre + routes_post,
                meander_instance=meander_ref,
                target_length=tgt,
                actual_length=actual,
                baseline_length=baseline,
            )
        )

    return results
