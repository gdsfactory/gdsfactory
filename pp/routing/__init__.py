""" Add optical and electrical routes
"""

from pp.routing.add_electrical_pads import add_electrical_pads
from pp.routing.add_electrical_pads_shortest import add_electrical_pads_shortest
from pp.routing.add_electrical_pads_top import add_electrical_pads_top
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.connect import (
    connect_elec_waypoints,
    connect_strip,
    connect_strip_way_points,
)
from pp.routing.connect_bundle import (
    connect_bundle,
    connect_bundle_path_length_match,
    link_electrical_ports,
    link_optical_ports,
    link_optical_ports_no_grouping,
)
from pp.routing.manhattan import round_corners, route_manhattan
from pp.routing.repackage import package_optical2x2
from pp.routing.route_fiber_single import route_fiber_single
from pp.routing.route_ports_to_side import route_elec_ports_to_side, route_ports_to_side
from pp.routing.route_south import route_south

route_factory = dict(
    optical=connect_strip_way_points, electrical=connect_elec_waypoints,
)

link_factory = dict(
    link_ports=connect_bundle,
    link_ports_path_length_match=connect_bundle_path_length_match,
    link_electrical_waypoints=connect_elec_waypoints,
    link_optical_waypoints=connect_strip_way_points,
    link_optical_no_grouping=link_optical_ports_no_grouping,
)

__all__ = [
    "add_electrical_pads",
    "add_electrical_pads_shortest",
    "add_electrical_pads_top",
    "add_fiber_array",
    "add_fiber_array",
    "add_fiber_single",
    "connect_bundle",
    "connect_bundle_path_length_match",
    "connect_strip",
    "connect_strip_way_points",
    "link_electrical_ports",
    "link_optical_ports",
    "link_optical_ports_no_grouping",
    "link_factory",
    "package_optical2x2",
    "round_corners",
    "route_elec_ports_to_side",
    "route_fiber_single",
    "route_manhattan",
    "route_ports_to_side",
    "route_ports_to_side",
    "route_factory",
    "route_south",
]


if __name__ == "__main__":
    print(__all__)
