""" route bundles of port (river routing)
"""

from typing import Callable, List, Optional
from numpy import float64, ndarray
import numpy as np
from pp.routing.connect import connect_strip
from pp.routing.connect import connect_elec_waypoints
from pp.routing.connect import connect_strip_way_points
from pp.routing.manhattan import generate_manhattan_waypoints

from pp.routing.u_groove_bundle import u_bundle_indirect
from pp.routing.u_groove_bundle import u_bundle_direct
from pp.routing.corner_bundle import corner_bundle
from pp.routing.path_length_matching import path_length_matched_points
from pp.name import autoname
from pp.component import ComponentReference, Component
from pp.port import Port
from pp.config import conf

METAL_MIN_SEPARATION = 10.0
BEND_RADIUS = conf.tech.bend_radius


def connect_bundle(
    start_ports,
    end_ports,
    route_filter=connect_strip_way_points,
    separation=5.0,
    bend_radius=BEND_RADIUS,
    extension_length=0,
    **kwargs,
):
    """Connects bundle of ports using river routing.
    Chooses the correct u_bundle to use based on port angles

    Args:
        start_ports should all be facing in the same direction
        end_ports should all be facing in the same direction
        route_filter: function to connect
        separation: waveguide separation
        bend_radius: for the routes
        extension_length: adds waveguide extension

    """
    # Accept dict ot list
    if isinstance(start_ports, dict):
        start_ports = list(start_ports.values())

    if isinstance(end_ports, dict):
        end_ports = list(end_ports.values())

    nb_ports = len(start_ports)
    for p in start_ports:
        p.angle = int(p.angle) % 360

    for p in end_ports:
        p.angle = int(p.angle) % 360

    assert len(end_ports) == nb_ports
    assert (
        len(set([p.angle for p in start_ports])) <= 1
    ), "All start port angles should be the same"

    assert (
        len(set([p.angle for p in end_ports])) <= 1
    ), "All end port angles should be the same"

    # Ensure the correct bend radius is used
    def _route_filter(*args, **kwargs):
        kwargs["bend_radius"] = bend_radius
        return route_filter(*args, **kwargs)

    params = {
        "start_ports": start_ports,
        "end_ports": end_ports,
        "route_filter": _route_filter,
        "separation": separation,
        "bend_radius": bend_radius,
    }

    start_angle = start_ports[0].angle
    end_angle = end_ports[0].angle

    start_axis = "X" if start_angle in [0, 180] else "Y"
    end_axis = "X" if end_angle in [0, 180] else "Y"

    x_start = np.mean([p.x for p in start_ports])
    x_end = np.mean([p.x for p in end_ports])

    y_start = np.mean([p.y for p in start_ports])
    y_end = np.mean([p.y for p in end_ports])

    if start_axis == end_axis:
        if (
            start_angle == 0
            and end_angle == 180
            and x_start < x_end
            or start_angle == 180
            and end_angle == 0
            and x_start > x_end
            or start_angle == 90
            and end_angle == 270
            and y_start < y_end
            or start_angle == 270
            and end_angle == 90
            and y_start > y_end
        ):
            return link_ports(**params, **kwargs)

        elif start_angle == end_angle:
            return u_bundle_direct(**params, **kwargs)

        elif end_angle == (start_angle + 180) % 360:
            params["extension_length"] = extension_length
            return u_bundle_indirect(**params, **kwargs)
        else:
            raise NotImplementedError("This should never happen")

    else:
        return corner_bundle(**params, **kwargs)
        raise NotImplementedError("Routing along different axis not implemented yet")


def get_port_x(port: Port) -> float64:
    return port.midpoint[0]


def get_port_y(port):
    return port.midpoint[1]


def get_port_width(port):
    return port.width


def are_decoupled(x1, x1p, x2, x2p, sep=METAL_MIN_SEPARATION):
    if x2p + sep > x1:
        return False
    if x2 < x1p + sep:
        return False
    if x2 < x1p - sep:
        return False
    return True


def link_ports(
    start_ports: List[Port],
    end_ports: List[Port],
    separation: float = 5.0,
    route_filter: Callable = connect_strip_way_points,
    **routing_params,
) -> List[ComponentReference]:
    """Semi auto-routing for two lists of ports.

    Args:
        ports1: first list of ports
        ports2: second list of ports
        separation: minimum separation between two waveguides
        axis: specifies "X" or "Y"
              X (resp. Y) -> indicates that the ports should be sorted and
             compared using the X (resp. Y) axis
        bend_radius: If unspecified, attempts to get it from the waveguide definition of the first port in ports1
        route_filter: filter to apply to the manhattan waypoints
            e.g `connect_strip_way_points` for deep etch strip waveguide
        end_straight_offset: offset to add at the end of each waveguide
        sort_ports: * True -> sort the ports according to the axis.
                    * False -> no sort applied
        compute_array_separation_only: If True, returns the min distance which should be used between the two arrays instead of returning the connectors. Useful for budgeting space before instantiating other components.

    Returns:
        `[route_filter(r) for r in routes]` where routes is a list of lists of coordinates
        e.g with default `connect_strip_way_points`, returns a list of elements which can be added to a component


    The routing assumes manhattan routing between the different ports.
    The strategy is to modify `start_straight` and `end_straight` for each
    waveguide such that waveguides do not collide.

    .. code::

        1             X    X     X  X X  X
        |-----------|    |     |  | |  |-----------------------|
        |          |-----|     |  | |---------------|          |
        |          |          ||  |------|          |          |
        2 X          X          X          X          X          X


    start: at the top
    end: at the bottom

    The general strategy is:
    Group tracks which would collide together and apply the following method
    on each group:

        if x2 >= x1, increase ``end_straight``
            (as seen on the right 3 ports)
        otherwise, decrease ``end_straight``
            (as seen on the first 2 ports)

    We deal with negative end_straight by doing at the end
        end_straights = end_straights - min(end_straights)

    This method deals with different metal track/wg/wire widths too.

    """

    routes = link_ports_routes(
        start_ports,
        end_ports,
        separation=separation,
        route_filter=generate_manhattan_waypoints,
        **routing_params,
    )

    return [route_filter(route, **routing_params) for route in routes]


def link_ports_routes(
    start_ports: List[Port],
    end_ports: List[Port],
    separation: float,
    bend_radius: float = BEND_RADIUS,
    route_filter: Callable = generate_manhattan_waypoints,
    sort_ports: bool = True,
    end_straight_offset: Optional[float] = None,
    compute_array_separation_only: bool = False,
    tol: float = 0.00001,
    **kwargs,
) -> List[ndarray]:
    """
    Args:
        start_ports: list of ports
        end_ports: list of ports
        separation: route spacing
        bend_radius: for route
        route_filter: Function used to connect two ports. Should be like `connect_strip`
        sort_ports: if True sort ports
        end_straight_offset: adds a straigth
    """
    ports1 = start_ports
    ports2 = end_ports
    if not ports1 and not ports2:
        return []

    if len(ports1) == 0 or len(ports2) == 0:
        print("WARNING! Not linking anything, empty list of ports")
        return []

    if start_ports[0].angle in [0, 180]:
        axis = "X"
    else:
        axis = "Y"

    if len(ports1) == 1 and len(ports2) == 1:
        if end_straight_offset:
            kwargs["end_straight"] = end_straight_offset
        return [
            route_filter(
                ports1[0],
                ports2[0],
                start_straight=0.05,
                bend_radius=bend_radius,
                **kwargs,
            )
        ]

    elems = []

    ## Contains end_straight of tracks which need to be adjusted together
    end_straights_in_group = []

    ## Once a group is finished, all the lengths are appended to end_straights
    end_straights = []

    ## Axis along which we sort the ports
    if axis in ["X", "x"]:
        f_key1 = get_port_y
        # f_key2 = get_port_y
    else:
        f_key1 = get_port_x
        # f_key2 = get_port_x

    # if sort_ports:
    #     ports1.sort(key=f_key1)
    #     ports2.sort(key=f_key2)

    ports2_by1 = {p1: p2 for p1, p2 in zip(ports1, ports2)}
    if sort_ports:
        ports1.sort(key=f_key1)
        ports2 = [ports2_by1[p1] for p1 in ports1]

    ## Keep track of how many ports should be routed together
    number_o_connectors_in_group = 0

    if axis in ["X", "x"]:
        x1_prev = get_port_y(ports1[0])
        x2_prev = get_port_y(ports2[0])
        y0 = get_port_x(ports2[0])
        y1 = get_port_x(ports1[0])
    else:  # X axis
        x1_prev = get_port_x(ports1[0])
        x2_prev = get_port_x(ports2[0])
        y0 = get_port_y(ports2[0])
        y1 = get_port_y(ports1[0])

    s = sign(y0 - y1)
    curr_end_straight = 0

    end_straight_offset = end_straight_offset or 15.0

    Le = end_straight_offset

    has_close_x_ports = False
    close_ports_thresh = 2 * bend_radius + 1.0

    ## First pass - loop on all the ports to find the tentative end_straights

    _w = get_port_width

    for i in range(len(ports1)):
        if axis in ["X", "x"]:
            x1 = get_port_y(ports1[i])
            x2 = get_port_y(ports2[i])
            y = get_port_x(ports2[i])
        else:
            x1 = get_port_x(ports1[i])
            x2 = get_port_x(ports2[i])
            y = get_port_y(ports2[i])

        dx = abs(x2 - x1)
        if dx < close_ports_thresh:
            has_close_x_ports = True

        """
        Compute the metal separation to use. This depends on the adjacent
        metal track widths
        """
        if i != len(ports1) - 1 and i != 0:
            # Deal with any track which is not on the edge
            max_width = max(_w(ports1[i + 1]), _w(ports1[i - 1]))
            curr_sep = 0.5 * (_w(ports1[i]) + max_width) + separation

        elif i == 0:
            # Deal with start edge case
            curr_sep = separation + 0.5 * (_w(ports1[0]) + _w(ports1[1]))

        elif i == len(ports1) - 1:
            # Deal with end edge case
            curr_sep = separation + 0.5 * (_w(ports1[-2]) + _w(ports1[-1]))

        if are_decoupled(x2, x2_prev, x1, x1_prev, sep=curr_sep):
            """
            If this metal track does not impact the previous one, then
            start a new group.
            """
            L = min(end_straights_in_group)
            end_straights += [max(x - L, 0) + Le for x in end_straights_in_group]

            # Start new group
            end_straights_in_group = []
            curr_end_straight = 0
            number_o_connectors_in_group = 0

        else:
            if x2 >= x1:
                curr_end_straight += curr_sep
            else:
                curr_end_straight -= curr_sep

        end_straights_in_group.append(curr_end_straight + (y - y0) * s)
        number_o_connectors_in_group += 1

        x1_prev = x1
        x2_prev = x2

    # Append the last group
    L = min(end_straights_in_group)

    end_straights += [max(x - L, 0) + Le for x in end_straights_in_group]

    if compute_array_separation_only:
        # If there is no port too close to each other in x, then there are
        # only two bends per route
        if not has_close_x_ports:
            return max(end_straights) + 2 * bend_radius
        else:
            return max(end_straights) + 4 * bend_radius

    ## Second pass - route the ports pairwise
    N = len(ports1)
    for i in range(N):
        if axis in ["X", "x"]:
            x1 = get_port_y(ports1[i])
            x2 = get_port_y(ports2[i])
        else:
            x1 = get_port_x(ports1[i])
            x2 = get_port_x(ports2[i])

        dx = abs(x2 - x1)

        # If both ports are aligned, we just need a straight line
        if dx < tol:
            elems += [
                route_filter(
                    ports1[i],
                    ports2[i],
                    start_straight=0,
                    end_straight=end_straights[i],
                    bend_radius=bend_radius,
                    **kwargs,
                )
            ]  #

        # Annoying case where it is too tight for direct manhattan routing
        elif dx < close_ports_thresh:

            a = close_ports_thresh + abs(dx)
            prt = ports1[i]
            angle = prt.angle
            dp_w = (0, -a) if axis in ["X", "x"] else (-a, 0)
            dp_e = (0, a) if axis in ["X", "x"] else (a, 0)
            do_step_aside = False
            if i == 0:
                ## If westest port, then we can safely step on the west further

                ## First check whether we have to step

                dx2 = ports1[i + 1].x - prt.x
                req_x = 2 * bend_radius + ports2[i].x - ports1[i].x

                if dx2 < req_x:
                    do_step_aside = True
                dp = dp_w

            elif i == N - 1:
                ## If eastest port, then we can safely step on the east further

                ## First check whether we have to step

                dx2 = prt.x - ports1[i - 1].x
                req_x = 2 * bend_radius + ports1[i].x - ports2[i].x

                if dx2 < req_x:
                    do_step_aside = True

                dp = dp_e

            else:
                ## Otherwise find closest port and escape where/if space permit

                dx1 = prt.x - ports1[i - 1].x
                dx2 = ports1[i + 1].x - prt.x
                do_step_aside = True
                if dx2 > dx1:
                    dp = dp_e
                else:
                    dp = dp_w

                ## If there is not enough space to step away, put a warning.
                ## This requires inspection on the mask. Raising an error
                ## would likely make it harder to debug. Here we will see
                ## a DRC error or an unwanted crossing on the mask.

                if max(dx1, dx2) < a:
                    print(
                        "WARNING - high risk of collision in routing. \
                    Ports too close to each other."
                    )
            _route = []
            if do_step_aside:
                tmp_port = prt.move_polar_copy(2 * bend_radius + 1.0, angle)
                tmp_port.move(dp)
                _route += [
                    route_filter(
                        prt, tmp_port.flip(), bend_radius=bend_radius, **kwargs
                    )
                ]

            else:
                tmp_port = prt
            # print(
            #     "STEPPING",
            #     ports1[i].position,
            #     tmp_port.position,
            #     ports2[i].position,
            # )
            _route += [
                route_filter(
                    tmp_port,
                    ports2[i],
                    start_straight=0.05,
                    end_straight=end_straights[i],
                    bend_radius=bend_radius,
                    **kwargs,
                )
            ]

            elems += _route
        # Usual case
        else:
            elems += [
                route_filter(
                    ports1[i],
                    ports2[i],
                    start_straight=0.05,
                    end_straight=end_straights[i],
                    bend_radius=bend_radius,
                    **kwargs,
                )
            ]
    return elems


def generate_waypoints_connect_bundle(*args, **kwargs):
    """
    returns a list of waypoints for each path generated with link_ports
    """
    return connect_bundle(*args, route_filter=lambda x, **params: x, **kwargs)


def compute_ports_max_displacement(start_ports, end_ports):
    if start_ports[0].angle in [0, 180]:
        a1 = [p.y for p in start_ports]
        a2 = [p.y for p in end_ports]
    else:
        a1 = [p.x for p in start_ports]
        a2 = [p.x for p in end_ports]

    return max(abs(max(a1) - min(a2)), abs(min(a1) - max(a2)))


def connect_bundle_path_length_match(
    ports1,
    ports2,
    separation=30.0,
    end_straight_offset=None,
    bend_radius=BEND_RADIUS,
    extra_length=0,
    nb_loops=1,
    modify_segment_i=-2,
    route_filter=connect_strip_way_points,
    **kwargs,
):
    """
    Args:
        ports1: list of ports
        ports2: list of ports
        separation: 30.0
        end_straight_offset
        bend_radius: BEND_RADIUS
        extra_length: distance added to all path length compensation. Useful is we want to add space for extra taper on all branches
        nb_loops: number of extra loops added in the path
        modify_segment_i: index of the segment which accomodates the new turns default is next to last segment
        route_filter: connect_strip_way_points
        **kwargs: extra arguments for inner call to generate_waypoints_connect_bundle

    Returns:
        [route_filter(l) for l in list_of_waypoints]

    """
    extra_length = extra_length / 2
    kwargs["separation"] = separation

    # Heuristic to get a correct default end_straight_offset to leave
    # enough space for path-length compensation

    if end_straight_offset is None:
        if modify_segment_i == -2:
            end_straight_offset = (
                compute_ports_max_displacement(ports1, ports2) / (2 * nb_loops)
                + separation
                + extra_length
            )
        else:
            end_straight_offset = 0

    kwargs["end_straight_offset"] = end_straight_offset
    kwargs["bend_radius"] = bend_radius
    list_of_waypoints = generate_waypoints_connect_bundle(ports1, ports2, **kwargs)

    list_of_waypoints = path_length_matched_points(
        list_of_waypoints,
        extra_length=extra_length,
        bend_radius=bend_radius,
        nb_loops=nb_loops,
        modify_segment_i=modify_segment_i,
    )
    return [route_filter(waypoints) for waypoints in list_of_waypoints]


def link_electrical_ports(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = METAL_MIN_SEPARATION,
    bend_radius: float = 0.0001,
    link_dummy_ports=False,
    route_filter: Callable = connect_elec_waypoints,
    **kwargs,
) -> List[ComponentReference]:
    """Connect bundle of electrical ports

    Args:
        ports1: first list of ports
        ports2: second list of ports
        separation: minimum separation between two waveguides
        axis: specifies "X" or "Y"
              X (resp. Y) -> indicates that the ports should be sorted and
             compared using the X (resp. Y) axis
        bend_radius: If unspecified, attempts to get it from the waveguide definition of the first port in ports1
        route_filter: filter to apply to the manhattan waypoints
            e.g `connect_strip_way_points` for deep etch strip waveguide

        end_straight_offset: offset to add at the end of each waveguide
        sort_ports: * True -> sort the ports according to the axis.
                    * False -> no sort applied
        compute_array_separation_only: If True, returns the min distance which should be used between the two arrays instead of returning the connectors. Useful for budgeting space before instantiating other components.

    Returns:
        list of references of the electrical routes
    """
    if link_dummy_ports:
        new_ports1 = ports1
        new_ports2 = ports2

    else:

        def is_dummy(port):
            return hasattr(port, "is_dummy") and port.is_dummy

        to_keep1 = [not is_dummy(p) for p in ports1]
        to_keep2 = [not is_dummy(p) for p in ports2]
        to_keep = [x * y for x, y in zip(to_keep1, to_keep2)]

        new_ports1 = [p for i, p in enumerate(ports1) if to_keep[i]]
        new_ports2 = [p for i, p in enumerate(ports2) if to_keep[i]]

    return link_ports(
        new_ports1,
        new_ports2,
        separation,
        bend_radius=bend_radius,
        route_filter=route_filter,
        **kwargs,
    )


def link_optical_ports(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 5.0,
    route_filter: Callable = connect_strip_way_points,
    bend_radius: float = BEND_RADIUS,
    **kwargs,
) -> List[ComponentReference]:
    """connect bundle of optical ports"""
    return link_ports(
        ports1,
        ports2,
        separation,
        bend_radius=bend_radius,
        route_filter=route_filter,
        **kwargs,
    )


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


def get_min_spacing(
    ports1: List[Port],
    ports2: List[Port],
    sep: float = 5.0,
    sort_ports: bool = True,
    radius: float = BEND_RADIUS,
) -> float:
    """
    Returns the minimum amount of spacing required to create a given fanout
    """

    if ports1[0].angle in [0, 180]:
        axis = "X"
    else:
        axis = "Y"

    j = 0
    min_j = 0
    max_j = 0
    if sort_ports:
        if axis in ["X", "x"]:
            ports1.sort(key=get_port_y)
            ports2.sort(key=get_port_y)
        else:
            ports1.sort(key=get_port_x)
            ports2.sort(key=get_port_x)

    for i, _ in enumerate(ports1):
        if axis in ["X", "x"]:
            x1 = get_port_y(ports1[i])
            x2 = get_port_y(ports2[i])
        else:
            x1 = get_port_x(ports1[i])
            x2 = get_port_x(ports2[i])
        if x2 >= x1:
            j += 1
        else:
            j -= 1
        if j < min_j:
            min_j = j
        if j > max_j:
            max_j = j
    j = 0

    return (max_j - min_j) * sep + 2 * radius + 1.0


def link_optical_ports_no_grouping(
    ports1,
    ports2,
    sep=5.0,
    route_filter=connect_strip,
    radius=BEND_RADIUS,
    start_straight=None,
    end_straight=None,
    sort_ports=True,
):
    """
    Returns a list of route elements
    Compared to link_ports, this function does not do any grouping.
    It is not as smart for the routing, but it can fall back on arclinarc
    connection if needed. We can also specify longer start_straight and end_straight

    Semi auto routing for optical ports
    The routing assumes manhattan routing between the different ports.
    The strategy is to modify ``start_straight`` and ``end_straight`` for each
    waveguide such that waveguides do not collide.


    We want to connect something like this:

    ::
         2             X    X     X  X X  X
           |-----------|    |     |  | |  |-----------------------|
           |          |-----|     |  | |---------------|          |
           |          |          ||  |------|          |          |
         1 X          X          X          X          X          X

    ``start`` is at the bottom
    ``end`` is at the top

    The general strategy is:

    if x2 < x1, decrease ``start straight``, and increase ``end_straight``
        (as seen on left two ports)
    otherwise, decrease ``start_straight``, and increase ``end_straight``
        (as seen on the last 3 right ports)

    Args:
        ports1: first list of optical ports
        ports2: second list of optical ports
        axis:   specifies "X" or "Y" direction along which the port is going
        route_filter:   ManhattanExpandedWgConnector or ManhattanWgConnector or any other connector function with the same input
        radius:         bend radius. If unspecified, uses the default radius
        start_straight: offset on the starting length before the first bend
        end_straight:   offset on the ending length after the last bend
        sort_ports:     True -> sort the ports according to the axis. False -> no sort applied

    Returns:
        a list of elements containing the connecting waveguides

    """

    if ports1[0].angle in [0, 180]:
        axis = "X"
    else:
        axis = "Y"

    elems = []
    j = 0

    # min and max offsets needed for avoiding collisions between waveguides
    min_j = 0
    max_j = 0

    if sort_ports:
        # Sort ports according to X or Y
        if axis in ["X", "x"]:
            ports1.sort(key=get_port_y)
            ports2.sort(key=get_port_y)
        else:
            ports1.sort(key=get_port_x)
            ports2.sort(key=get_port_x)

    # Compute max_j and min_j
    for i in range(len(ports1)):
        if axis in ["X", "x"]:
            x1 = ports1[i].position[1]
            x2 = ports2[i].position[1]
        else:
            x1 = ports1[i].position[0]
            x2 = ports2[i].position[0]
        if x2 >= x1:
            j += 1
        else:
            j -= 1
        if j < min_j:
            min_j = j
        if j > max_j:
            max_j = j
    j = 0

    if start_straight is None:
        start_straight = 0.2

    if end_straight is None:
        end_straight = 0.2

    start_straight += max_j * sep
    end_straight += -min_j * sep

    # Do case with wire direct if the ys are close to each other
    for i, _ in enumerate(ports1):

        if axis in ["X", "x"]:
            x1 = ports1[i].position[1]
            x2 = ports2[i].position[1]
        else:
            x1 = ports1[i].position[0]
            x2 = ports2[i].position[0]

        s_straight = start_straight - j * sep
        e_straight = j * sep + end_straight

        if radius is None:
            elems += [
                route_filter(
                    ports1[i],
                    ports2[i],
                    start_straight=s_straight,
                    end_straight=e_straight,
                )
            ]
        else:
            elems += [
                route_filter(
                    ports1[i],
                    ports2[i],
                    start_straight=s_straight,
                    end_straight=e_straight,
                    bend_radius=radius,
                )
            ]

        if x2 >= x1:
            j += 1
        else:
            j -= 1
    return elems


@autoname
def test_connect_bundle():

    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]

    pitch = 127.0
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]

    top_ports = [Port("top_{}".format(i), (xs_top[i], 0), 0.5, 270) for i in range(N)]

    bottom_ports = [
        Port("bottom_{}".format(i), (xs_bottom[i], -400), 0.5, 90) for i in range(N)
    ]

    top_cell = Component(name="connect_bundle")
    routes = connect_bundle(top_ports, bottom_ports)
    top_cell.add(routes)
    lengths = [
        1180.4159265358978,
        1063.4159265358978,
        946.4159265358978,
        899.4159265358979,
        782.4159265358979,
        665.4159265358979,
        558.4159265358979,
        441.41592653589794,
        438.41592653589794,
        555.4159265358979,
        672.4159265358979,
        794.4159265358979,
        916.415926535898,
        1038.415926535898,
    ]
    for route, length in zip(routes, lengths):
        # print(route.parent.length)
        assert np.isclose(route.parent.length, length)
    return top_cell


@autoname
def test_connect_corner(N=6, config="A"):
    d = 10.0
    sep = 5.0
    top_cell = Component(name="connect_corner")

    if config in ["A", "B"]:
        a = 100.0
        ports_A_TR = [
            Port("A_TR_{}".format(i), (d, a / 2 + i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_TL = [
            Port("A_TL_{}".format(i), (-d, a / 2 + i * sep), 0.5, 180) for i in range(N)
        ]
        ports_A_BR = [
            Port("A_BR_{}".format(i), (d, -a / 2 - i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_BL = [
            Port("A_BL_{}".format(i), (-d, -a / 2 - i * sep), 0.5, 180)
            for i in range(N)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port("B_TR_{}".format(i), (a / 2 + i * sep, d), 0.5, 90) for i in range(N)
        ]
        ports_B_TL = [
            Port("B_TL_{}".format(i), (-a / 2 - i * sep, d), 0.5, 90) for i in range(N)
        ]
        ports_B_BR = [
            Port("B_BR_{}".format(i), (a / 2 + i * sep, -d), 0.5, 270) for i in range(N)
        ]
        ports_B_BL = [
            Port("B_BL_{}".format(i), (-a / 2 - i * sep, -d), 0.5, 270)
            for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in ["C", "D"]:
        a = N * sep + 2 * d
        ports_A_TR = [
            Port("A_TR_{}".format(i), (a, d + i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_TL = [
            Port("A_TL_{}".format(i), (-a, d + i * sep), 0.5, 180) for i in range(N)
        ]
        ports_A_BR = [
            Port("A_BR_{}".format(i), (a, -d - i * sep), 0.5, 0) for i in range(N)
        ]
        ports_A_BL = [
            Port("A_BL_{}".format(i), (-a, -d - i * sep), 0.5, 180) for i in range(N)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port("B_TR_{}".format(i), (d + i * sep, a), 0.5, 90) for i in range(N)
        ]
        ports_B_TL = [
            Port("B_TL_{}".format(i), (-d - i * sep, a), 0.5, 90) for i in range(N)
        ]
        ports_B_BR = [
            Port("B_BR_{}".format(i), (d + i * sep, -a), 0.5, 270) for i in range(N)
        ]
        ports_B_BL = [
            Port("B_BL_{}".format(i), (-d - i * sep, -a), 0.5, 270) for i in range(N)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    if config in ["A", "C"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            routes = connect_bundle(ports1, ports2)
            top_cell.add(routes)

    elif config in ["B", "D"]:
        for ports1, ports2 in zip(ports_A, ports_B):
            routes = connect_bundle(ports2, ports1)
            top_cell.add(routes)

    return top_cell


@autoname
def test_connect_bundle_udirect(dy=200, angle=270):

    xs1 = [-100, -90, -80, -55, -35, 24, 0] + [200, 210, 240]

    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, angle) for i in range(N)]

        ports2 = [
            Port("bottom_{}".format(i), (dy, xs2[i]), 0.5, angle) for i in range(N)
        ]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, angle) for i in range(N)]

        ports2 = [
            Port("bottom_{}".format(i), (xs2[i], dy), 0.5, angle) for i in range(N)
        ]

    top_cell = Component(name="connect_bundle_udirect")
    routes = connect_bundle(ports1, ports2)
    lengths = [
        237.4359265358979,
        281.4359265358979,
        336.4359265358979,
        376.4359265358979,
        421.4359265358979,
        451.4359265358979,
        481.4359265358979,
        271.4359265358979,
    ]

    for route, length in zip(routes, lengths):
        # print(route.parent.length)
        assert np.isclose(route.parent.length, length)

    top_cell.add(routes)
    return top_cell


@autoname
def test_connect_bundle_u_indirect(dy=-200, angle=180):
    xs1 = [-100, -90, -80, -55, -35] + [200, 210, 240]
    axis = "X" if angle in [0, 180] else "Y"

    pitch = 10.0
    N = len(xs1)
    xs2 = [50 + i * pitch for i in range(N)]

    a1 = angle
    a2 = a1 + 180

    if axis == "X":
        ports1 = [Port("top_{}".format(i), (0, xs1[i]), 0.5, a1) for i in range(N)]

        ports2 = [Port("bottom_{}".format(i), (dy, xs2[i]), 0.5, a2) for i in range(N)]

    else:
        ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]

        ports2 = [Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    top_cell = Component("connect_bundle_u_indirect")
    routes = connect_bundle(ports1, ports2)
    lengths = [
        341.41592653589794,
        341.41592653589794,
        341.41592653589794,
        326.41592653589794,
        316.41592653589794,
        291.41592653589794,
        291.41592653589794,
        311.41592653589794,
    ]

    for route, length in zip(routes, lengths):
        # print(route.parent.length)
        assert np.isclose(route.parent.length, length)

    top_cell.add(routes)

    return top_cell


@autoname
def test_facing_ports():
    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    top_cell = Component("test_facing_ports")
    routes = connect_bundle(ports1, ports2)
    top_cell.add(routes)
    lengths = [
        671.4159265358979,
        481.41592653589794,
        291.41592653589794,
        291.41592653589794,
        291.41592653589794,
        276.41592653589794,
        626.4159265358979,
        401.41592653589794,
        401.41592653589794,
        381.41592653589794,
        251.41592653589794,
        391.41592653589794,
    ]
    for route, length in zip(routes, lengths):
        # print(route.parent.length)
        assert np.isclose(route.parent.length, length)

    return top_cell


def demo_connect_bundle():
    """ combines all the connect_bundle tests """

    y = 400.0
    x = 500
    y0 = 900
    dy = 200.0
    c = Component("connect_bundle")
    for j, s in enumerate([-1, 1]):
        for i, angle in enumerate([0, 90, 180, 270]):
            c2 = test_connect_bundle_u_indirect(dy=s * dy, angle=angle)
            c2ref = c2.ref(position=(i * x, j * y))
            c.add(c2ref)

            c2 = test_connect_bundle_udirect(dy=s * dy, angle=angle)
            c2ref = c2.ref(position=(i * x, j * y + y0))
            c.add(c2ref)

    for i, config in enumerate(["A", "B", "C", "D"]):
        c2 = test_connect_corner(config=config)
        c2ref = c2.ref(position=(i * x, 1700))
        c.add(c2ref)

    c2 = test_facing_ports()
    c2ref = c2.ref(position=(800, 1820))
    c.add(c2ref)

    return c


def test_connect_bundle_small():
    import pp

    c = pp.Component()
    c1 = c << pp.c.mmi2x2()
    c2 = c << pp.c.mmi2x2()
    c2.move((100, 40))
    routes = connect_bundle(
        [c1.ports["E0"], c1.ports["E1"]],
        [c2.ports["W0"], c2.ports["W1"]],
        bend_radius=5,
    )
    for route in routes:
        # print(route.parent.length)
        assert np.isclose(route.parent.length, 100.25796326794897)
    c.add(routes)
    return c


if __name__ == "__main__":
    import pp

    # c = demo_connect_bundle()
    # c = test_connect_bundle_small()
    # c = test_facing_ports()
    # c = test_connect_bundle_u_indirect()
    # c = test_connect_bundle_udirect()
    c = test_connect_bundle()
    pp.show(c)
