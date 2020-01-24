import numpy as np
import pp
from pp import Port
from pp.routing.connect_bundle_from_waypoints import connect_bundle_waypoints


@pp.autoname
def test_connect_bundle_waypoints():
    return test_connect_bundle_waypointsA()


@pp.autoname
def test_connect_bundle_waypointsA():

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


@pp.autoname
def test_connect_bundle_waypointsB():

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


@pp.autoname
def test_connect_bundle_waypointsC():

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


@pp.autoname
def test_connect_bundle_waypointsD():

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
