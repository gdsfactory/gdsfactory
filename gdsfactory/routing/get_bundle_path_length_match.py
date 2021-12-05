"""Routes bundles of ports (river routing).
"""
from typing import Callable, List, Optional

from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.get_bundle import (
    _get_bundle_waypoints,
    compute_ports_max_displacement,
)
from gdsfactory.routing.get_route import get_route_from_waypoints
from gdsfactory.routing.path_length_matching import path_length_matched_points
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Route


def get_bundle_path_length_match(
    ports1: List[Port],
    ports2: List[Port],
    separation: float = 30.0,
    end_straight_length: Optional[float] = None,
    extra_length: float = 0.0,
    nb_loops: int = 1,
    modify_segment_i: int = -2,
    bend: ComponentFactory = bend_euler,
    straight: Callable = straight,
    taper: Optional[Callable] = taper_function,
    start_straight_length: float = 0.0,
    route_filter: Callable = get_route_from_waypoints,
    sort_ports: bool = True,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> List[Route]:
    """Returns list of routes that are path length matched.

    Args:
        ports1: list of ports
        ports2: list of ports
        separation: between the loops
        end_straight_length: if None tries to determine it
        extra_length: distance added to all path length compensation.
            Useful is we want to add space for extra taper on all branches
        nb_loops: number of extra loops added in the path
        modify_segment_i: index of the segment that accomodates the new turns
            default is next to last segment
        bend: for bends
        straight: for straights
        taper:
        start_straight_length:
        route_filter: get_route_from_waypoints
        sort_ports: sorts ports before routing
        cross_section: factory
        **kwargs: cross_section settings

    Tips:

    - If path length matches the wrong segments, change `modify_segment_i` arguments.
    - Adjust `nb_loops` to avoid too short or too long segments
    - Adjust `separation` and `end_straight_offset` to avoid compensation collisions


    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.Component("path_length_match_sample")

      dy = 2000.0
      xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]
      pitch = 100.0
      N = len(xs1)
      xs2 = [-20 + i * pitch for i in range(N)]

      a1 = 90
      a2 = a1 + 180
      ports1 = [gf.Port(f"top_{i}", (xs1[i], 0), 0.5, a1) for i in range(N)]
      ports2 = [gf.Port(f"bottom_{i}", (xs2[i], dy), 0.5, a2) for i in range(N)]

      routes = gf.routing.get_bundle_path_length_match(
          ports1, ports2, extra_length=44
      )
      for route in routes:
          c.add(route.references)
      c.plot()

    """
    extra_length = extra_length / 2

    # Heuristic to get a correct default end_straight_offset to leave
    # enough space for path-length compensation
    if sort_ports:
        ports1, ports2 = sort_ports_function(ports1, ports2)

    if end_straight_length is None:
        if modify_segment_i == -2:
            end_straight_length = (
                compute_ports_max_displacement(ports1, ports2) / (2 * nb_loops)
                + separation
                + extra_length
            )
        else:
            end_straight_length = 0

    list_of_waypoints = _get_bundle_waypoints(
        ports1=ports1,
        ports2=ports2,
        separation=separation,
        end_straight_length=end_straight_length,
        start_straight_length=start_straight_length,
        cross_section=cross_section,
        **kwargs,
    )

    list_of_waypoints = path_length_matched_points(
        list_of_waypoints,
        extra_length=extra_length,
        bend=bend,
        nb_loops=nb_loops,
        modify_segment_i=modify_segment_i,
        cross_section=cross_section,
        **kwargs,
    )
    return [
        route_filter(
            waypoints,
            bend=bend,
            straight=straight,
            taper=taper,
            cross_section=cross_section,
            **kwargs,
        )
        for waypoints in list_of_waypoints
    ]


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.components.straight_array(spacing=50)
    c2 = c << gf.components.straight_array(spacing=5)
    c2.movex(200)
    c1.y = 0
    c2.y = 0

    routes = gf.routing.get_bundle_path_length_match(
        c1.get_ports_list(orientation=0),
        c2.get_ports_list(orientation=180),
        end_straight_length=0,
        start_straight_length=0,
        separation=50,
        layer=(2, 0),
    )

    for route in routes:
        c.add(route.references)
    c.show()
