import numpy as np
from phidl.device_layout import _rotate_points

from pp.routing.connect import connect_strip_way_points
from pp.routing.manhattan import generate_manhattan_waypoints


def _groups(ports, cut, axis="X"):
    if axis == "Y":
        group1 = [p for p in ports if p.x <= cut]
        group2 = [p for p in ports if p.x > cut]
    else:
        group1 = [p for p in ports if p.y <= cut]
        group2 = [p for p in ports if p.y > cut]

    return group1, group2


def _transform_port(
    point, orientation, origin=(0, 0), rotation=None, x_reflection=False
):
    new_point = np.array(point)
    new_orientation = orientation

    if x_reflection:
        new_point[1] = -new_point[1]
        new_orientation = -orientation
    if rotation is not None:
        new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
        new_orientation += rotation
    if origin is not None:
        new_point = new_point + np.array(origin)
    new_orientation = new_orientation % 360

    return new_point, new_orientation


def _transform_ports(ports, rotation, origin=(0, 0), x_reflection=False):
    ports_transformed = []
    for p in ports:
        new_port = p._copy(new_uid=False)
        new_midpoint, new_orientation = _transform_port(
            p.midpoint, p.orientation, origin, rotation, x_reflection
        )
        new_port.midpoint = new_midpoint
        new_port.new_orientation = new_orientation
        ports_transformed.append(new_port)

    return ports_transformed


def corner_bundle(
    start_ports,
    end_ports,
    route_filter=connect_strip_way_points,
    separation=5.0,
    **kwargs
):
    r"""
    Args:
        start_ports: list of start ports
        end_ports: list of end ports
        route_filter: filter to apply to the manhattan waypoints
            e.g `connect_strip_way_points` for deep etch strip waveguide
    Returns:
        `[route_filter(r) for r in routes]` where routes is a list of lists of coordinates
        e.g with default `connect_strip_way_points`, returns a list of elements which can be added to a component


    ::

        Bend-types

        90 Deg bend
                   A1 A2      AN
                   | |  ...  |
                   | |      /
                  / /      /
                 / /      /
          B1 ----/ /    _/
          B2 -----/     /
                       /
            ...      _/
          BN --------/


        270 Deg bend
            /------------\
           /              \
          /  /-------\     \
         /  /         \     \
        |  /   /--\    |    |
        | /  /   |     | ...|
        | | /    B1    B2   BN
        | | |
        | \ \--- A1
        |  ----- A2
        |    ...
         \------ AN


    Connect banks of ports with either 90Deg or 270Deg angle between them
    """

    routes = corner_bundle_route(
        start_ports,
        end_ports,
        routing_func=generate_manhattan_waypoints,
        separation=separation,
        **kwargs
    )

    return [route_filter(r) for r in routes]


def corner_bundle_route(
    start_ports,
    end_ports,
    routing_func=generate_manhattan_waypoints,
    separation=5.0,
    **kwargs
):

    nb_ports = len(start_ports)
    connections = []

    for p in start_ports:
        p.angle = p.angle % 360

    for p in end_ports:
        p.angle = p.angle % 360

    assert len(end_ports) == nb_ports
    assert (
        len(set([p.angle for p in start_ports])) <= 1
    ), "All start ports should have identical angle"

    assert (
        len(set([p.angle for p in end_ports])) <= 1
    ), "All end ports should have identical angle"

    a_start = start_ports[0].angle
    a_end = end_ports[0].angle

    da = a_end - a_start
    assert (
        da
    ) % 180 == 90, "corner_bundle can \
    only route port banks between orthogonal axises. Got angles of {} and {}\
    ".format(
        a_start, a_end
    )

    # Rotate all ports to be in the configuration where start_angle = 0

    origin = start_ports[0].midpoint
    start_ports_transformed = _transform_ports(
        start_ports, rotation=-a_start, origin=origin
    )
    end_ports_transformed = _transform_ports(
        end_ports, rotation=-a_start, origin=origin
    )

    # a_end_tr = end_ports_transformed[0].angle % 360

    ys = [p.y for p in start_ports_transformed]
    ye = [p.y for p in end_ports_transformed]

    xs = [p.x for p in start_ports_transformed]
    xe = [p.x for p in end_ports_transformed]

    are_above = max(ys) < min(ye)
    are_below = min(ys) > max(ye)

    are_right = max(xs) < min(xe)
    are_left = min(xs) > max(xe)

    assert are_above or are_below, "corner_bundle - ports should be below or above"
    assert are_right or are_left, "corner_bundle - ports should be left or right"

    start_sort_type = ["Y", "-X", "-Y", "X"]

    da = da % 360

    if da == 270:
        da = -90

    is_routable_270 = ((are_below and are_left) or (are_above and are_right)) and (
        da == -90
    )
    is_routable_90 = ((are_below and are_right) or (are_above and are_left)) and (
        da == 90
    )
    assert (
        is_routable_270 or is_routable_90
    ), "Ports not routable with corner_bundle: \
    da={}; are_below={}; are_above={}; are_left={}; are_right={}. \
    Consider applying a U turn first and then to to the 90Deg or 270Deg connection".format(
        da, are_below, are_above, are_left, are_right
    )

    if da > 0:
        end_sort_type = ["Y", "-X", "-Y", "X"]
    else:
        end_sort_type = ["-Y", "X", "Y", "-X"]

    start_angle_sort_index = a_start // 90
    end_angle_sort_index = a_end // 90

    # flip sort for a few specific cases:
    if are_above and are_left and da == 90:
        start_angle_sort_index = start_angle_sort_index + 2
        start_angle_sort_index = start_angle_sort_index % 4

    if are_below and are_right and da == 90:
        end_angle_sort_index = end_angle_sort_index + 2
        end_angle_sort_index = end_angle_sort_index % 4

    start_angle_sort_type = start_sort_type[start_angle_sort_index]
    end_angle_sort_type = end_sort_type[end_angle_sort_index]

    type2key = {
        "X": lambda p: p.x,
        "Y": lambda p: p.y,
        "-X": lambda p: -p.x,
        "-Y": lambda p: -p.y,
    }

    # print(a_start, a_end, start_angle_sort_type, end_angle_sort_type)
    start_ports.sort(key=type2key[start_angle_sort_type])
    end_ports.sort(key=type2key[end_angle_sort_type])

    i = 0
    for p1, p2 in zip(start_ports, end_ports):
        conn = routing_func(
            p1, p2, start_straight=i * separation, end_straight=i * separation, **kwargs
        )
        connections += [conn]
        i += 1

    return connections
