"""Routes bundles of ports (river routing)."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as _straight
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.routing.path_length_matching import path_length_matched_points
from gdsfactory.routing.route_bundle import (
    _route_bundle_waypoints,
    compute_ports_max_displacement,
)
from gdsfactory.routing.route_single import route_single
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    MultiCrossSectionAngleSpec,
    Route,
)


def route_bundle_path_length_match(
    component,
    ports1: list[Port],
    ports2: list[Port],
    separation: float = 30.0,
    end_straight_length: float | None = None,
    extra_length: float = 0.0,
    nb_loops: int = 1,
    modify_segment_i: int = -2,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = _straight,
    taper: ComponentSpec | None = taper_function,
    start_straight_length: float = 0.0,
    sort_ports: bool = True,
    cross_section: CrossSectionSpec | MultiCrossSectionAngleSpec = strip,
    enforce_port_ordering: bool = True,
    **kwargs,
) -> list[Route]:
    """Returns list of routes that are path length matched.

    Args:
        component: to add the routes to.
        ports1: list of ports.
        ports2: list of ports.
        separation: between the loops.
        end_straight_length: if None tries to determine it.
        extra_length: distance added to all path length compensation. Useful is we want to add space for extra taper on all branches.
        nb_loops: number of extra loops added in the path.
        modify_segment_i: index of the segment that accommodates the new turns default is next to last segment.
        bend: for bends.
        straight: for straights.
        taper: spec.
        start_straight_length: in um.
        route_filter: route_single_from_waypoints.
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

      routes = gf.routing.route_bundle_path_length_match(ports1, ports2, extra_length=44)
      for route in routes:
          c.add(route.references)

      gf.config.set_plot_options(show_subports=False)
      c.plot()

    """
    extra_length /= 2
    cross_section = gf.get_cross_section(cross_section)
    cross_section = cross_section.copy(**kwargs)
    radius = cross_section.radius

    # Heuristic to get a correct default end_straight_offset to leave
    # enough space for path-length compensation
    if sort_ports:
        ports1, ports2 = sort_ports_function(
            ports1, ports2, enforce_port_ordering=enforce_port_ordering
        )

    if end_straight_length is None:
        if modify_segment_i == -2:
            end_straight_length = (
                compute_ports_max_displacement(ports1, ports2) / (2 * nb_loops)
                + separation
                + extra_length
            )
        else:
            end_straight_length = 0

    separation_dbu = round(separation / component.kcl.dbu)
    end_straight_length_dbu = round(end_straight_length / component.kcl.dbu)
    start_straight_length_dbu = round(start_straight_length / component.kcl.dbu)
    radius_dbu = round(radius / component.kcl.dbu)

    list_of_waypoints = _route_bundle_waypoints(
        ports1=ports1,
        ports2=ports2,
        separation_dbu=separation_dbu,
        end_straight_length_dbu=end_straight_length_dbu,
        start_straight_length_dbu=start_straight_length_dbu,
        radius_dbu=radius_dbu,
    )

    list_of_waypoints = path_length_matched_points(
        list_of_waypoints,
        extra_length=extra_length,
        bend=bend,
        nb_loops=nb_loops,
        modify_segment_i=modify_segment_i,
        cross_section=cross_section,
    )

    for port1, port2, waypoints in zip(ports1, ports2, list_of_waypoints):
        route_single(
            port1=port1,
            port2=port2,
            waypoints=waypoints,
            bend=bend,
            straight=straight,
            taper=taper,
            cross_section=cross_section,
            end_straight_length=end_straight_length,
            start_straight_length=start_straight_length,
        )


if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.straight_array(spacing=50)
    c2 = c << gf.components.straight_array(spacing=5)
    c2.d.movex(200)
    c1.y = 0
    c2.y = 0

    routes = gf.routing.route_bundle_path_length_match(
        component=c,
        ports1=gf.port.get_ports_list(c1.ports, orientation=0),
        ports2=gf.port.get_ports_list(c2.ports, orientation=180),
        end_straight_length=0,
        start_straight_length=0,
        separation=50,
    )

    c.show()
