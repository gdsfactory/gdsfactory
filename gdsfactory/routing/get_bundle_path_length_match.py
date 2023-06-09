"""Routes bundles of ports (river routing)."""
from __future__ import annotations

from typing import List

from gdsfactory.config import logger
from gdsfactory.port import Port
from gdsfactory.routing.get_bundle import get_bundle
from gdsfactory.typings import (
    Route,
)


def get_bundle_path_length_match(
    ports1: List[Port],
    ports2: List[Port],
    extra_length: float = 0.0,
    nb_loops: int = 1,
    modify_segment_i: int = -2,
    **kwargs,
) -> List[Route]:
    """Returns list of routes that are path length matched.

    TODO: remove.

    Args:
        ports1: list of ports.
        ports2: list of ports.
        separation: between the loops.
        end_straight_length: if None tries to determine it.
        extra_length: distance added to all path length compensation.
            Useful is we want to add space for extra taper on all branches.
        nb_loops: number of extra loops added in the path.
        modify_segment_i: index of the segment that accommodates the new turns
            default is next to last segment.
        bend: for bends.
        straight: for straights.
        taper: spec.
        start_straight_length: in um.
        route_filter: get_route_from_waypoints.
        sort_ports: sorts ports before routing.
        cross_section: factory.
        kwargs: cross_section settings.

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
      ports1 = [gf.Port(name=f"top_{i}", center=(xs1[i], +0), width=0.5, orientation=a1, layer="WG") for i in range(N)]
      ports2 = [gf.Port(name=f"bot_{i}", center=(xs2[i], dy), width=0.5, orientation=a2, layer="WG") for i in range(N)]

      routes = gf.routing.get_bundle_path_length_match(ports1, ports2, extra_length=44)
      for route in routes:
          c.add(route.references)

      gf.config.set_plot_options(show_subports=False)
      c.plot()

    """
    logger.warning(
        "get_bundle_path_length_match is deprecated and will be removed!. Use get_bundle instead, which can also do path_length_match",
        DeprecationWarning,
    )
    return get_bundle(
        ports1=ports1,
        ports2=ports2,
        path_length_match_loops=nb_loops,
        path_length_match_extra_length=extra_length,
        path_length_match_modify_segment_i=modify_segment_i,
    )


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
    c.show(show_ports=True)
