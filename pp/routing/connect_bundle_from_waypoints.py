import numpy as np

import pp
from pp.components import bend_circular
from pp.components import taper as taper_factory
from pp.components import waveguide
from pp.config import TAPER_LENGTH, WG_EXPANDED_WIDTH
from pp.routing.manhattan import remove_flat_angles, round_corners
from pp.routing.utils import get_list_ports_angle


def _is_vertical(segment, tol=1e-5):
    p0, p1 = segment
    return abs(p0[0] - p1[0]) < tol


def _is_horizontal(segment, tol=1e-5):
    p0, p1 = segment
    return abs(p0[1] - p1[1]) < tol


def _segment_sign(s):
    p0, p1 = s
    if _is_vertical(s):
        return np.sign(p1[1] - p0[1])

    if _is_horizontal(s):
        return np.sign(p1[0] - p0[0])


def get_ports_x_or_y_distances(list_ports, ref_point):
    if not list_ports:
        return []

    angle = get_list_ports_angle(list_ports)
    x0 = ref_point[0]
    y0 = ref_point[1]
    if angle in [0, 180]:
        xys = [p.y - y0 for p in list_ports]
        xs = [round(p.x, 5) for p in list_ports]
        if len(set(xs)) > 1:
            raise ValueError("List ports with angle 0/180 should all have the same x")
    else:
        xys = [p.x - x0 for p in list_ports]
        ys = [round(p.y, 5) for p in list_ports]
        if len(set(ys)) > 1:
            raise ValueError("List ports with angle 90/270 should all have the same y")
    return xys


def _distance(port1, port2):
    if hasattr(port1, "x"):
        x1, y1 = port1.x, port1.y
    else:
        x1, y1 = port1[0], port1[1]

    if hasattr(port2, "x"):
        x2, y2 = port2.x, port2.y
    else:
        x2, y2 = port2[0], port2[1]

    dx = x1 - x2
    dy = y1 - y2

    return np.sqrt(dx ** 2 + dy ** 2)


def connect_bundle_waypoints(
    start_ports,
    end_ports,
    way_points,
    straight_factory=waveguide,
    taper_factory=taper_factory,
    bend_factory=bend_circular,
    bend_radius=10.0,
    auto_sort=True,
    **kwargs
):
    """
    Args:
        start_ports: list of ports
        end_ports: list of ports
        way_points: list of points defining a route

    """
    if len(end_ports) != len(start_ports):
        raise ValueError(
            "Number of start ports should match number of end ports.\
        Got {} {}".format(
                len(start_ports), len(end_ports)
            )
        )

    for p in start_ports:
        p.angle = int(p.angle) % 360

    for p in end_ports:
        p.angle = int(p.angle) % 360

    start_angle = start_ports[0].orientation
    end_angle = end_ports[0].orientation

    # Sort the ports such that the bundle connect the correct corresponding ports.

    angles_to_sorttypes = {
        (0, 180): ("Y", "Y"),
        (0, 90): ("Y", "X"),
        (0, 0): ("Y", "-Y"),
        (0, 270): ("Y", "-X"),
        (90, 0): ("X", "Y"),
        (90, 90): ("X", "-X"),
        (90, 180): ("X", "-Y"),
        (90, 270): ("X", "X"),
        (180, 90): ("Y", "-X"),
        (180, 0): ("Y", "Y"),
        (180, 270): ("Y", "X"),
        (180, 180): ("Y", "-Y"),
        (270, 90): ("X", "X"),
        (270, 270): ("X", "-X"),
        (270, 0): ("X", "-Y"),
        (270, 180): ("X", "Y"),
    }

    dict_sorts = {
        "X": lambda p: p.x,
        "Y": lambda p: p.y,
        "-X": lambda p: -p.x,
        "-Y": lambda p: -p.y,
    }
    key = (start_angle, end_angle)
    sp_st, ep_st = angles_to_sorttypes[key]
    start_port_sort = dict_sorts[sp_st]
    end_port_sort = dict_sorts[ep_st]

    if auto_sort:
        start_ports.sort(key=start_port_sort)
        end_ports.sort(key=end_port_sort)

    routes = _generate_manhattan_bundle_waypoints(
        start_ports, end_ports, way_points, **kwargs
    )

    bends90 = [bend_factory(radius=bend_radius, width=p.width) for p in start_ports]

    if taper_factory:
        if callable(taper_factory):
            taper = taper_factory(
                length=TAPER_LENGTH,
                width1=start_ports[0].width,
                width2=WG_EXPANDED_WIDTH,
                layer=start_ports[0].layer,
            )
        else:
            # In this case the taper is a fixed cell
            taper = taper_factory
    else:
        taper = None
    connections = [
        round_corners(pts, bend90, straight_factory, taper=taper)
        for pts, bend90 in zip(routes, bends90)
    ]
    return connections


def snap_route_to_end_point_x(route, x):
    y1, y2 = [p[1] for p in route[-2:]]
    return route[:-2] + [(x, y1), (x, y2)]


def snap_route_to_end_point_y(route, y):
    x1, x2 = [p[0] for p in route[-2:]]
    return route[:-2] + [(x1, y), (x2, y)]


def _generate_manhattan_bundle_waypoints(
    start_ports, end_ports, backbone_route, **kwargs
):
    """
    Args:
        start_ports: list of start ports. Should all be facing in the same direction
        end_ports: list of end ports. Should all be facing in the same direction
        route: going from one point somewhere within the start_ports bank to
        another point within the end_ports bank
    """
    backbone_route = remove_flat_angles(backbone_route)

    way_segments = [(p0, p1) for p0, p1 in zip(backbone_route[:-1], backbone_route[1:])]
    offsets_start = get_ports_x_or_y_distances(start_ports, backbone_route[0])

    start_angle = start_ports[0].orientation
    if start_angle in [90, 270]:
        offsets_start = [-_d for _d in offsets_start]

    end_angle = end_ports[0].orientation

    def _displace_segment_copy(s, a, sh=1, sv=1):
        sign_seg = _segment_sign(s)
        if _is_horizontal(s):
            dp = (0, sh * sign_seg * a)
        elif _is_vertical(s):
            dp = (sv * sign_seg * a, 0)
        else:
            raise ValueError("Segment should be manhattan, got {}".format(s))

        displaced_seg = [np.array(p) + dp for p in s]
        return displaced_seg

    def _displace_segment_copy_group1(s, a):
        return _displace_segment_copy(s, a, sh=1, sv=-1)

    def _intersection(s1, s2):
        if _is_horizontal(s1) and _is_vertical(s2):
            sh, sv = s1, s2
        elif _is_horizontal(s2) and _is_vertical(s1):
            sh, sv = s2, s1
        else:

            if _is_horizontal(s1):
                s1_dir = "h"
            elif _is_vertical(s1):
                s1_dir = "v"
            else:
                s1_dir = "u"

            if _is_horizontal(s2):
                s2_dir = "h"
            elif _is_vertical(s2):
                s2_dir = "v"
            else:
                s2_dir = "u"

            raise ValueError(
                "s1 / s2 should be h/v or v/h. Got \
            {} {} {} {}".format(
                    s1_dir, s2_dir, s1, s2
                )
            )
        return (sv[0][0], sh[0][1])

    N = len(start_ports)
    routes = []
    _make_segment = _displace_segment_copy_group1
    for i in range(N):
        prev_seg_sep = offsets_start[i]
        route = []

        for j, seg in enumerate(way_segments):
            seg_sep = prev_seg_sep
            d_seg = _make_segment(seg, seg_sep)

            if j == 0:
                start_point = d_seg[0]
            else:
                prev_seg = way_segments[j - 1]
                tmp_seg = _make_segment(prev_seg, prev_seg_sep)
                start_point = _intersection(d_seg, tmp_seg)

            route += [start_point]

            # If last point before the ports, adjust the separation to the end ports
            if j == len(way_segments) - 1:
                end_point = end_ports[i].position
                route += [end_point]

                if end_angle in [0, 180]:
                    route = snap_route_to_end_point_y(route, end_point[1])
                else:
                    route = snap_route_to_end_point_x(route, end_point[0])

        routes += [route]

    return routes


@pp.cell
def test_connect_bundle_waypoints():
    return test_connect_bundle_waypointsA()


@pp.cell
def test_connect_bundle_waypointsA():
    import pp
    from pp.component import Port

    xs1 = np.arange(10) * 5 - 500.0

    N = xs1.size
    ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 75]) + 500.0

    ports1 = [Port("A_{}".format(i), (xs1[i], 0), 0.5, 90) for i in range(N)]
    ports2 = [Port("B_{}".format(i), (0, ys2[i]), 0.5, 180) for i in range(N)]

    top_cell = pp.Component()
    p0 = ports1[0].position + (22, 0)
    way_points = [
        p0,
        p0 + (0, 100),
        p0 + (200, 100),
        p0 + (200, -200),
        p0 + (0, -200),
        p0 + (0, -350),
        p0 + (400, -350),
        (p0[0] + 400, ports2[-1].y),
        ports2[-1].position,
    ]

    elements = connect_bundle_waypoints(ports1, ports2, way_points)
    top_cell.add(elements)

    return top_cell


@pp.cell
def test_connect_bundle_waypointsB():
    import pp
    from pp.component import Port

    ys1 = np.array([0, 5, 10, 15, 30, 40, 50, 60]) + 0.0
    ys2 = np.array([0, 10, 20, 30, 70, 90, 110, 120]) + 500.0
    N = ys1.size

    ports1 = [Port("A_{}".format(i), (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [Port("B_{}".format(i), (500, ys2[i]), 0.5, 180) for i in range(N)]

    p0 = ports1[0].position + (0, 22.5)

    top_cell = pp.Component()
    way_points = [
        p0,
        p0 + (200, 0),
        p0 + (200, -200),
        p0 + (400, -200),
        (p0[0] + 400, ports2[0].y),
        ports2[0].position,
    ]

    elements = connect_bundle_waypoints(ports1, ports2, way_points)
    top_cell.add(elements)

    return top_cell


@pp.cell
def test_connect_bundle_waypointsC():
    import pp
    from pp.component import Port

    ys1 = np.array([0, 5, 10, 15, 20, 60, 70, 80, 120, 125])
    ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 65]) - 500.0
    N = ys1.size

    ports1 = [Port("A_{}".format(i), (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [Port("B_{}".format(i), (600, ys2[i]), 0.5, 180) for i in range(N)]

    top_cell = pp.Component()
    way_points = [
        ports1[0].position,
        ports1[0].position + (200, 0),
        ports1[0].position + (200, -200),
        ports1[0].position + (400, -200),
        (ports1[0].x + 400, ports2[0].y),
        ports2[0].position,
    ]

    elements = connect_bundle_waypoints(ports1, ports2, way_points)
    top_cell.add(elements)

    return top_cell


@pp.cell
def test_connect_bundle_waypointsD():
    import pp
    from pp.component import Port

    ys1 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 75]) + 100.0
    ys2 = np.array([0, -5, -10, -20, -25, -30, -40, -55, -60, -75]) + 500.0
    N = ys1.size

    ports1 = [Port("A_{}".format(i), (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [Port("B_{}".format(i), (0, ys2[i]), 0.5, 0) for i in range(N)]

    def _mean_y(ports):
        return np.mean([p.y for p in ports])

    yc1 = _mean_y(ports1)
    yc2 = _mean_y(ports2)

    top_cell = pp.Component()
    way_points = [(0, yc1), (200, yc1), (200, yc2), (0, yc2)]

    elements = connect_bundle_waypoints(ports1, ports2, way_points)
    top_cell.add(elements)

    return top_cell


if __name__ == "__main__":
    cell = test_connect_bundle_waypointsD()
    pp.show(cell)
