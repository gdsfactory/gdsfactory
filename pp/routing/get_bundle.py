"""Routes bundles of ports (river routing).
"""
from typing import Callable, List, Optional, Union, cast

import numpy as np
from numpy import ndarray

import pp
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.straight import straight
from pp.components.taper import taper as taper_function
from pp.config import TECH
from pp.cross_section import get_waveguide_settings
from pp.port import Port
from pp.routing.corner_bundle import corner_bundle
from pp.routing.get_route import (
    get_route,
    get_route_from_waypoints,
    get_route_from_waypoints_electrical,
)
from pp.routing.manhattan import generate_manhattan_waypoints
from pp.routing.path_length_matching import path_length_matched_points
from pp.routing.sort_ports import get_port_x, get_port_y
from pp.routing.sort_ports import sort_ports as sort_ports_function
from pp.routing.u_groove_bundle import u_bundle_direct, u_bundle_indirect
from pp.types import ComponentFactory, Number, Route

METAL_MIN_SEPARATION = TECH.metal_spacing


def get_bundle(
    ports1: List[Port],
    ports2: List[Port],
    route_filter: Callable = get_route_from_waypoints,
    separation: float = 5.0,
    extension_length: float = 0.0,
    bend_factory: ComponentFactory = bend_euler,
    sort_ports: bool = True,
    end_straight_offset: float = 0.0,
    start_straight: float = 0.01,
    waveguide: str = "strip",
    **waveguide_settings,
) -> List[Route]:
    """Connects bundle of ports using river routing.
    Chooses the correct u_bundle to use based on port angles

    Args:
        ports1: should all be facing in the same direction
        ports2: should all be facing in the same direction
        route_filter: function to connect
        separation: straight separation
        extension_length: adds straight extension
        bend_factory:
        waveguide
        waveguide_settings: waveguide_settings

    """
    # Accept dict or list
    if isinstance(ports1, Port):
        ports1 = [ports1]

    if isinstance(ports2, Port):
        ports2 = [ports2]

    if isinstance(ports1, dict):
        ports1 = list(ports1.values())

    if isinstance(ports2, dict):
        ports2 = list(ports2.values())

    for p in ports1:
        p.angle = int(p.angle) % 360

    for p in ports2:
        p.angle = int(p.angle) % 360

    assert len(ports1) == len(
        ports2
    ), f"ports1={len(ports1)} and ports2={len(ports2)} must be equal"

    ports1 = cast(List[Port], ports1)
    ports2 = cast(List[Port], ports2)

    waveguide_settings = get_waveguide_settings(waveguide, **waveguide_settings)

    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    start_port_angles = set([p.angle for p in ports1])
    if len(start_port_angles) > 1:
        raise ValueError(
            "All start port angles should be the same", f"Got {start_port_angles}"
        )

    params = {
        "ports1": ports1,
        "ports2": ports2,
        "route_filter": route_filter,
        "separation": separation,
        "bend_factory": bend_factory,
        "end_straight_offset": end_straight_offset,
        "start_straight": start_straight,
    }

    start_angle = ports1[0].angle
    end_angle = ports2[0].angle

    start_axis = "X" if start_angle in [0, 180] else "Y"
    end_axis = "X" if end_angle in [0, 180] else "Y"

    x_start = np.mean([p.x for p in ports1])
    x_end = np.mean([p.x for p in ports2])

    y_start = np.mean([p.y for p in ports1])
    y_end = np.mean([p.y for p in ports2])

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
            # print('link_ports')
            return link_ports(**params, **waveguide_settings)

        elif start_angle == end_angle:
            # print('u_bundle_direct')
            return u_bundle_direct(**params, **waveguide_settings)

        elif end_angle == (start_angle + 180) % 360:
            # print('u_bundle_indirect')
            params["extension_length"] = extension_length
            return u_bundle_indirect(**params, **waveguide_settings)
        else:
            raise NotImplementedError("This should never happen")

    else:
        # print('corner_bundle')
        # return link_ports(**params, **waveguide_settings)
        return corner_bundle(
            ports1=ports1,
            ports2=ports2,
            separation=separation,
            waveguide=waveguide,
            **waveguide_settings,
        )


def get_port_width(port: Port) -> Union[float, int]:
    return port.width


def are_decoupled(
    x1: Number,
    x1p: Number,
    x2: Number,
    x2p: Number,
    sep: float = METAL_MIN_SEPARATION,
) -> bool:
    if x2p + sep > x1:
        return False
    if x2 < x1p + sep:
        return False
    if x2 < x1p - sep:
        return False
    return True


def link_ports(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 5.0,
    end_straight_offset: float = 0.0,
    start_straight: float = 0.0,
    bend_factory: ComponentFactory = bend_euler,
    route_filter: Callable = get_route_from_waypoints,
    waveguide: str = "strip",
    sort_ports: bool = True,
    **waveguide_settings,
) -> List[Route]:
    r"""Semi auto-routing for two lists of ports.

    Args:
        ports1: first list of ports
        ports2: second list of ports
        separation: minimum separation between two straights
        axis: specifies "X" or "Y"
              X (resp. Y) -> indicates that the ports should be sorted and
             compared using the X (resp. Y) axis
        route_filter: filter to apply to the manhattan waypoints
            e.g `get_route_from_waypoints` for deep etch strip straight
        end_straight_offset: offset to add at the end of each straight
        sort_ports: * True -> sort the ports according to the axis.
                    * False -> no sort applied

    Returns:
        `[route_filter(r) for r in routes]` list of lists of coordinates
        e.g with default `get_route_from_waypoints`,
        returns a list of elements which can be added to a component


    The routing assumes manhattan routing between the different ports.
    The strategy is to modify `start_straight` and `end_straight` for each
    straight such that straights do not collide.

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
    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    routes = link_ports_routes(
        ports1,
        ports2,
        separation=separation,
        bend_factory=bend_factory,
        waveguide=waveguide,
        end_straight_offset=end_straight_offset,
        start_straight=start_straight,
        sort_ports=sort_ports,
        **waveguide_settings,
    )
    return [
        route_filter(
            route,
            bend_factory=bend_factory,
            waveguide=waveguide,
            **waveguide_settings,
        )
        for route in routes
    ]


def link_ports_routes(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 30,
    sort_ports: bool = True,
    end_straight_offset: float = 0.0,
    tol: float = 0.00001,
    start_straight: float = 0.0,
    waveguide: str = "strip",
    **waveguide_settings,
) -> List[ndarray]:
    """Returns route coordinates List

    Args:
        ports1: list of starting ports
        ports2: list of end ports
        separation: route spacing
        sort_ports: if True sort ports
        end_straight_offset: adds a straigth
        tol: tolerance
        start_straight: length of straight
        waveguide
        waveguide_settings: waveguide_settings
    """

    if not ports1 and not ports2:
        return []

    assert len(ports1) == len(
        ports2
    ), f"ports1={len(ports1)} and ports2={len(ports2)} must be equal"

    if len(ports1) == 0 or len(ports2) == 0:
        print(f"WARNING! ports1={ports1} or ports2={ports2} are empty")
        return []

    if ports1[0].angle in [0, 180]:
        axis = "X"
    else:
        axis = "Y"

    if len(ports1) == 1 and len(ports2) == 1:
        return [
            generate_manhattan_waypoints(
                ports1[0],
                ports2[0],
                start_straight=start_straight,
                end_straight=end_straight_offset,
                waveguide=waveguide,
                **waveguide_settings,
            )
        ]

    elems = []

    # Contains end_straight of tracks which need to be adjusted together
    end_straights_in_group = []

    # Once a group is finished, all the lengths are appended to end_straights
    end_straights = []

    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    # Keep track of how many ports should be routed together
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

    # First pass - loop on all the ports to find the tentative end_straights
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

        # Compute the metal separation to use. This depends on the adjacent metal
        # track widths
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
            # If this metal track does not impact the previous one, then start a new
            # group.
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

    # Second pass - route the ports pairwise
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
                generate_manhattan_waypoints(
                    ports1[i],
                    ports2[i],
                    start_straight=start_straight,
                    end_straight=end_straights[i],
                    waveguide=waveguide,
                    **waveguide_settings,
                )
            ]

        else:
            elems += [
                generate_manhattan_waypoints(
                    ports1[i],
                    ports2[i],
                    start_straight=start_straight,
                    end_straight=end_straights[i],
                    waveguide=waveguide,
                    **waveguide_settings,
                )
            ]
    return elems


def compute_ports_max_displacement(ports1: List[Port], ports2: List[Port]) -> Number:
    if ports1[0].angle in [0, 180]:
        a1 = [p.y for p in ports1]
        a2 = [p.y for p in ports2]
    else:
        a1 = [p.x for p in ports1]
        a2 = [p.x for p in ports2]

    return max(abs(max(a1) - min(a2)), abs(min(a1) - max(a2)))


def get_bundle_path_length_match(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 30.0,
    end_straight_offset: Optional[float] = None,
    extra_length: float = 0.0,
    nb_loops: int = 1,
    modify_segment_i: int = -2,
    bend_factory: ComponentFactory = bend_euler,
    straight_factory: Callable = straight,
    taper_factory: Optional[Callable] = taper_function,
    start_straight: float = 0.0,
    route_filter: Callable = get_route_from_waypoints,
    waveguide: str = "strip",
    **waveguide_settings,
) -> List[Route]:
    """Returns list of routes that are path length matched.

    Args:
        ports1: list of ports
        ports2: list of ports
        separation: 30.0
        end_straight_offset
        extra_length: distance added to all path length compensation.
            Useful is we want to add space for extra taper on all branches
        nb_loops: number of extra loops added in the path
        modify_segment_i: index of the segment that accomodates the new turns
            default is next to last segment
        route_filter: get_route_from_waypoints
        bend_factory: for bends
        waveguide
        waveguide_settings: waveguide_settings

    Tips:

    - If path length matches the wrong segments, change `modify_segment_i` arguments.
    - Adjust `nb_loops` to avoid too short or too long segments
    - Adjust `separation` and `end_straight_offset` to avoid compensation collisions


    .. plot::
      :include-source:

      import pp

      c = pp.Component("path_length_match_sample")

      dy = 2000.0
      xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]
      pitch = 100.0
      N = len(xs1)
      xs2 = [-20 + i * pitch for i in range(N)]

      a1 = 90
      a2 = a1 + 180
      ports1 = [pp.Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
      ports2 = [pp.Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

      routes = pp.routing.get_bundle_path_length_match(
          ports1, ports2, extra_length=44
      )
      for route in routes:
          c.add(route['references'])
      c.plot()

    """
    extra_length = extra_length / 2

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

    list_of_waypoints = link_ports_routes(
        ports1=ports1,
        ports2=ports2,
        separation=separation,
        end_straight_offset=end_straight_offset,
        start_straight=start_straight,
        waveguide=waveguide,
        **waveguide_settings,
    )

    list_of_waypoints = path_length_matched_points(
        list_of_waypoints,
        extra_length=extra_length,
        bend_factory=bend_factory,
        nb_loops=nb_loops,
        modify_segment_i=modify_segment_i,
        waveguide=waveguide,
        **waveguide_settings,
    )
    return [
        route_filter(
            waypoints,
            bend_factory=bend_factory,
            straight_factory=straight_factory,
            taper_factory=taper_factory,
            waveguide=waveguide,
            **waveguide_settings,
        )
        for waypoints in list_of_waypoints
    ]


def link_electrical_ports(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = METAL_MIN_SEPARATION,
    link_dummy_ports: bool = False,
    route_filter: Callable = get_route_from_waypoints_electrical,
    **kwargs,
) -> List[Route]:
    """Connect bundle of electrical ports

    Args:
        ports1: first list of ports
        ports2: second list of ports
        separation: minimum separation between two straights
        axis: specifies "X" or "Y"
              X (resp. Y) -> indicates that the ports should be sorted and
             compared using the X (resp. Y) axis
        route_filter: filter to apply to the manhattan waypoints
            e.g `get_route_from_waypoints` for deep etch strip straight

        end_straight_offset: offset to add at the end of each straight
        sort_ports: * True -> sort the ports according to the axis.
                    * False -> no sort applied

    Returns:
        list of references of the electrical routes
    """

    assert len(ports1) == len(
        ports2
    ), f"ports1={len(ports1)} and ports2={len(ports2)} must be equal"

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
        route_filter=route_filter,
        **kwargs,
    )


def link_optical_ports(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 5.0,
    route_filter: Callable = get_route_from_waypoints,
    sort_ports: bool = True,
    **waveguide_settings,
) -> List[Route]:
    """connect bundle of optical ports"""
    return link_ports(
        ports1,
        ports2,
        separation,
        route_filter=route_filter,
        sort_ports=sort_ports,
        **waveguide_settings,
    )


def sign(x: Number) -> int:
    if x > 0:
        return 1
    else:
        return -1


def get_min_spacing(
    ports1: List[Port],
    ports2: List[Port],
    sep: float = 5.0,
    radius: float = 5.0,
    sort_ports: bool = True,
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
    ports1: List[Port],
    ports2: List[Port],
    sep: float = 5.0,
    route_filter: Callable = get_route,
    start_straight: Optional[float] = None,
    end_straight: Optional[float] = None,
    sort_ports: bool = True,
    waveguide: str = "strip",
    **waveguide_settings,
) -> List[Route]:
    r"""Returns a list of route elements.

    Compared to link_ports, this function does not do any grouping.
    It is not as smart for the routing, but it can fall back on arclinarc
    connection if needed. We can also specify longer start_straight and end_straight

    Semi auto routing for optical ports
    The routing assumes manhattan routing between the different ports.
    The strategy is to modify ``start_straight`` and ``end_straight`` for each
    straight such that straights do not collide.


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
        axis: specifies "X" or "Y" direction along which the port is going
        route_filter: ManhattanExpandedWgConnector or ManhattanWgConnector
            or any other connector function with the same input
        radius: bend radius. If unspecified, uses the default radius
        start_straight: offset on the starting length before the first bend
        end_straight: offset on the ending length after the last bend
        sort_ports: True -> sort the ports according to the axis. False -> no sort applied

    Returns:
        a list of routes the connecting straights

    """

    if ports1[0].angle in [0, 180]:
        axis = "X"
    else:
        axis = "Y"

    elems = []
    j = 0

    # min and max offsets needed for avoiding collisions between straights
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

        elems += [
            route_filter(
                ports1[i],
                ports2[i],
                start_straight=s_straight,
                end_straight=e_straight,
                waveguide=waveguide,
                **waveguide_settings,
            )
        ]

        if x2 >= x1:
            j += 1
        else:
            j -= 1
    return elems


@pp.cell_with_validator
def test_get_bundle_small() -> Component:
    import pp

    c = pp.Component()
    c1 = c << pp.components.mmi2x2()
    c2 = c << pp.components.mmi2x2()
    c2.move((100, 40))
    routes = get_bundle(
        [c1.ports["E0"], c1.ports["E1"]],
        [c2.ports["W0"], c2.ports["W1"]],
        radius=5,
    )
    for route in routes:
        # print(route.length)
        assert np.isclose(route.length, 111.3)
        c.add(route.references)
    return c


if __name__ == "__main__":

    # c = test_connect_corner(None, check=False)
    # c = test_get_bundle_small()
    c = test_get_bundle_small()
    # c = test_facing_ports()
    # c = test_get_bundle_u_indirect()
    # c = test_get_bundle_udirect()
    # c = test_connect_corner()

    c.show()
