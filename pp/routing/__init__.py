"""
Route optical and electrical waveguides
"""

from pp.routing.connect_bundle import (
    connect_bundle,
    link_electrical_ports,
    link_optical_ports,
)
from pp.routing.connect import connect_strip, connect_strip_way_points
from pp.routing.connect_component import add_io_optical
from pp.routing.manhattan import round_corners, route_manhattan
from pp.routing.repackage import package_optical2x2
from pp.routing.route_fiber_single import route_fiber_single
from pp.routing.route_ports_to_side import route_elec_ports_to_side, route_ports_to_side
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.add_electrical_pads import add_electrical_pads
from pp.routing.add_electrical_pads_top import add_electrical_pads_top
from pp.routing.add_electrical_pads_shortest import add_electrical_pads_shortest

__all__ = [
    "add_io_optical",
    "add_fiber_array",
    "connect_bundle",
    "connect_strip",
    "connect_strip_way_points",
    "link_electrical_ports",
    "link_optical_ports",
    "package_optical2x2",
    "round_corners",
    "route_elec_ports_to_side",
    "route_fiber_single",
    "route_manhattan",
    "route_ports_to_side",
    "route_ports_to_side",
    "add_electrical_pads",
    "add_electrical_pads_top",
    "add_electrical_pads_shortest",
    "add_fiber_array",
    "add_fiber_single",
]

if __name__ == "__main__":
    print(__all__)
